"""Differentiable coordinate and lattice geometry shared across atomic models and losses."""

from __future__ import annotations

import torch

from glass.matching import image_shifts, nearest_pairwise_image


def gram(cell: torch.Tensor) -> torch.Tensor:
    """Six lattice tokens (log lengths, angle cosines) -> Gram matrix."""
    a, b, c = cell[..., :3].clamp(-5, 5).exp().unbind(-1)
    ca, cb, cg = cell[..., 3:].clamp(-0.9999, 0.9999).unbind(-1)

    return torch.stack(
        (
            a * a,
            a * b * cg,
            a * c * cb,
            a * b * cg,
            b * b,
            b * c * ca,
            a * c * cb,
            b * c * ca,
            c * c,
        ),
        -1,
    ).reshape(*cell.shape[:-1], 3, 3)


def periodic_delta(
    a: torch.Tensor, b: torch.Tensor, metric: torch.Tensor
) -> torch.Tensor:
    """Return fractional displacement from the nearest image of b to a.

    Broadcast a and b over atoms/pairs; metric is the cell Gram matrix [B, 3, 3].
    Choose a cell offset s in {-1, 0, 1}³ minimizing Cartesian distance, then
    return a - (b + s). E.g. 0.95 - 0.05 wraps from +0.90 to -0.10 in a cubic
    cell. Only the image choice is discrete: coordinate gradients still flow
    through a-b. The 27-image search is not exact for arbitrary skew cells.
    """
    displacement = a - b
    pairwise = a.ndim == 4 and a.shape[2] == 1 and b.ndim == 4 and b.shape[1] == 1

    if pairwise:
        # The GPU kernel selects an image without storing all 27 candidates.
        image_shift = nearest_pairwise_image(a.squeeze(2), b.squeeze(1), metric)
    else:
        shifts = image_shifts(a)
        candidates = displacement.unsqueeze(-2) - shifts
        squared_distance = torch.einsum(
            "b...ki,bij,b...kj->b...k", candidates, metric, candidates
        )
        image_shift = shifts[squared_distance.argmin(-1)]

    # The integer image choice is discrete; gradients flow through the coordinates.
    return displacement - image_shift


def predicted_cell_rmsd(pred, pred_cell, target, cell, real):
    """Cartesian RMSD for assigned atoms in their respective, unscaled cells.

    Both row-vector cells use a lower-triangular orientation and the supplied
    origin. Search the 27 adjacent predicted images for each target atom.
    Invalid cells yield NaN; callers must report their number separately.
    Computed in float64 on the CPU, since Apple GPUs do not support float64.
    """
    pred, pred_cell, target, cell, real = (
        x.cpu() for x in (pred, pred_cell, target, cell, real)
    )

    def lattice(tokens):
        lengths = tokens[..., :3].double().exp()
        ca, cb, cg = tokens[..., 3:].double().unbind(-1)
        one = torch.ones_like(ca)
        angles = torch.stack((one, cg, cb, cg, one, ca, cb, ca, one), -1)
        metric = angles.reshape(-1, 3, 3) * lengths[:, :, None] * lengths[:, None, :]
        matrix, info = torch.linalg.cholesky_ex(metric)
        valid = (info == 0) & torch.isfinite(matrix).all(dim=(-1, -2))
        return matrix, valid

    predicted_lattice, predicted_valid = lattice(pred_cell)
    target_lattice, target_valid = lattice(cell)
    if not target_valid.all():
        raise ValueError("Invalid target lattice in reconstruction evaluation")

    target_cartesian = target.double() @ target_lattice
    predicted_images = pred.double().unsqueeze(-2) + image_shifts(pred.double())
    predicted_cartesian = torch.einsum(
        "bnki,bij->bnkj", predicted_images, predicted_lattice
    )
    squared_error = (
        (predicted_cartesian - target_cartesian.unsqueeze(-2)).square().sum(-1).amin(-1)
    )
    rmsd = (squared_error.masked_fill(~real, 0).sum(1) / real.sum(1)).sqrt()
    return rmsd.masked_fill(~predicted_valid, float("nan"))
