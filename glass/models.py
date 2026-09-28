"""GLASS models: a pooled crystal encoder, a slot decoder, and a latent velocity flow."""

import math

import torch
import torch.nn.functional as F
from torch import nn

NUM_TYPES = 95  # Type 0 marks an empty slot; 1-94 are atomic numbers.


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: (silu(xW_1) * xW_3) W_2."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SelfAttentionLayer(nn.Module):
    """Pre-norm scaled-dot-product attention with a SwiGLU feed-forward block."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, int(8 / 3 * dim))

    def forward(self, x, mask=None):
        batch, atoms, dim = x.shape

        qkv = (
            self.qkv(self.norm1(x))
            .reshape(batch, atoms, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        h = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
        x = x + self.out(h.transpose(1, 2).reshape(batch, atoms, dim))

        return x + self.mlp(self.norm2(x))


def coordinate_features(coords, bands):
    fractional = coords.remainder(1)
    # Integer harmonics make both values and derivatives periodic under f -> f + k.
    # Evaluate phases in float32 even when the surrounding transformer uses bf16.
    frequencies = torch.arange(1, bands + 1, device=coords.device, dtype=torch.float32)
    phase = 2 * math.pi * fractional.float()[..., None] * frequencies
    return torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)


class Encoder(nn.Module):
    """Permutation-invariant encoder: atom tokens, self-attention, attention pooling."""

    def __init__(self, dim, latent_dim, depth, heads, coord_fourier_bands):
        super().__init__()
        self.coord_fourier_bands = coord_fourier_bands
        self.type_emb = nn.Embedding(NUM_TYPES, dim)
        self.coord_emb = nn.Sequential(
            nn.Linear(6 * coord_fourier_bands, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.cell_emb = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.layers = nn.ModuleList(
            SelfAttentionLayer(dim, heads) for _ in range(depth)
        )

        self.norm = nn.LayerNorm(dim)
        self.pool_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool_kv = nn.Linear(dim, 2 * dim)
        self.latent_proj = nn.Linear(dim, latent_dim)

    def forward(self, types, coords, padding, cell):
        # Lattice features are broadcast to every atom token.
        x = self.type_emb(types) + self.cell_emb(cell)[:, None]
        x = x + self.coord_emb(coordinate_features(coords, self.coord_fourier_bands))
        mask = (~padding)[:, None, None, :]

        for layer in self.layers:
            x = layer(x, mask)

        key, value = self.pool_kv(self.norm(x))[:, None].chunk(2, dim=-1)
        query = self.pool_query[:, None].expand(len(x), -1, -1, -1)
        pooled = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        return self.latent_proj(pooled.reshape(len(x), -1))


class Decoder(nn.Module):
    """Learned slot queries, each concatenated with the latent, decode one atom each."""

    def __init__(self, nmax, dim, latent_dim, depth, heads, slot_dim):
        super().__init__()
        self.nmax = nmax
        self.slot_proj = nn.Linear(slot_dim + latent_dim, dim)
        self.queries = nn.Embedding(nmax, slot_dim)

        self.layers = nn.ModuleList(
            SelfAttentionLayer(dim, heads) for _ in range(depth)
        )

        # Required before fractional wrapping: large bf16 outputs otherwise collapse to integers.
        self.norm = nn.LayerNorm(dim)
        self.coord_head = nn.Linear(dim, 3)
        self.type_head = nn.Linear(dim, NUM_TYPES)

        self.cell_head = nn.Sequential(
            nn.Linear(latent_dim, dim), nn.GELU(), nn.Linear(dim, 6)
        )

    def forward(self, z):
        queries = self.queries.weight[None].expand(len(z), -1, -1)
        latent = z[:, None].expand(-1, self.nmax, -1)
        slots = self.slot_proj(torch.cat((queries, latent), dim=-1))

        for layer in self.layers:
            slots = layer(slots)

        with torch.autocast(slots.device.type, enabled=False):
            slots = self.norm(slots.float())
            coords = self.coord_head(slots).remainder(1)
            cell = self.cell_head(z.float())

        return coords, self.type_head(slots), cell


class Autoencoder(nn.Module):
    def __init__(
        self,
        nmax,
        dim,
        latent_dim,
        enc_depth,
        dec_depth,
        heads,
        slot_dim,
        coord_fourier_bands,
    ):
        super().__init__()
        self.encoder = Encoder(dim, latent_dim, enc_depth, heads, coord_fourier_bands)
        self.decoder = Decoder(nmax, dim, latent_dim, dec_depth, heads, slot_dim)

    def forward(self, types, coords, padding, cell):
        return self.decoder(self.encoder(types, coords, padding, cell))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
        phase = t.reshape(-1, 1) * freq
        return torch.cat((phase.sin(), phase.cos()), -1)


class FlowBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm, self.time = nn.LayerNorm(hidden), nn.Linear(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.SiLU(), nn.Linear(4 * hidden, hidden)
        )

    def forward(self, x, time):
        return x + self.mlp(self.norm(x) + self.time(time))


class LatentFlow(nn.Module):
    """Time-conditioned residual MLP predicting the latent velocity x1 - x0."""

    def __init__(self, latent_dim, hidden, depth):
        super().__init__()
        self.latent_dim = latent_dim
        self.time = nn.Sequential(
            TimeEmbedding(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.input = nn.Linear(latent_dim, hidden)

        self.blocks = nn.ModuleList(FlowBlock(hidden) for _ in range(depth))
        self.norm, self.output = nn.LayerNorm(hidden), nn.Linear(hidden, latent_dim)

    def forward(self, x, t):
        # ODE time stays in [0, 1]; sinusoidal features use the [0, 1000] scale.
        time = self.time(t * 1000)
        x = self.input(x)

        for block in self.blocks:
            x = block(x, time)

        return self.output(self.norm(x))

    @torch.no_grad()
    def sample(self, count, steps, generator):
        """Integrate from Gaussian noise to a standardized latent with midpoint steps."""
        device = generator.device
        x = torch.randn(count, self.latent_dim, device=device, generator=generator)
        dt = 1 / steps

        for step in range(steps):
            t0 = torch.full((count, 1), step * dt, device=device)
            mid = self(x, t0)
            x = x + dt * self(x + 0.5 * dt * mid, t0 + 0.5 * dt)

        return x
