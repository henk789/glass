"""Periodic matching costs and minimum-cost atom assignments.

Coordinates are fractional; the cell Gram matrix converts displacements to
squared Cartesian distances. CUDA kernels avoid materializing all 27 periodic
images or moving assignment costs to the CPU. CPU paths use PyTorch and SciPy.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU installations need no Triton.
    triton = tl = None


def image_shifts(reference):
    """Return the 27 offsets in {-1, 0, 1}³, on reference's device and dtype.

    An offset s identifies the image b + s of a fractional position b. Offsets
    are lexicographically ordered, so argmin chooses the first image in a tie.
    """
    axis = torch.arange(-1, 2, device=reference.device, dtype=reference.dtype)
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(
        27, 3
    )


if triton is not None:

    @triton.jit
    def _periodic_distance_kernel(
        target_ptr,
        pred_ptr,
        metric_ptr,
        cost_ptr,
        stride_tb,
        stride_tn,
        stride_td,
        stride_pb,
        stride_pn,
        stride_pd,
        stride_mb,
        stride_mr,
        stride_mc,
        stride_cb,
        stride_cn,
        stride_cm,
        n_targets,
        n_preds,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Compute a tile of pair distances, keeping only the best of 27 images.

        The symmetric Gram matrix needs six entries. Expanding dᵀ metric d
        here avoids storing a [batch, targets, slots, images, 3] tensor.
        """
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < n_targets
        mask_n = offs_n < n_preds

        m_ptr = metric_ptr + pid_b * stride_mb
        m00 = tl.load(m_ptr + 0 * stride_mr + 0 * stride_mc)
        m01 = tl.load(m_ptr + 0 * stride_mr + 1 * stride_mc)
        m02 = tl.load(m_ptr + 0 * stride_mr + 2 * stride_mc)
        m11 = tl.load(m_ptr + 1 * stride_mr + 1 * stride_mc)
        m12 = tl.load(m_ptr + 1 * stride_mr + 2 * stride_mc)
        m22 = tl.load(m_ptr + 2 * stride_mr + 2 * stride_mc)

        t_base = target_ptr + pid_b * stride_tb + offs_m[:, None] * stride_tn
        t0 = tl.load(t_base + 0 * stride_td, mask=mask_m[:, None], other=0.0)
        t1 = tl.load(t_base + 1 * stride_td, mask=mask_m[:, None], other=0.0)
        t2 = tl.load(t_base + 2 * stride_td, mask=mask_m[:, None], other=0.0)

        p_base = pred_ptr + pid_b * stride_pb + offs_n[None, :] * stride_pn
        p0 = tl.load(p_base + 0 * stride_pd, mask=mask_n[None, :], other=0.0)
        p1 = tl.load(p_base + 1 * stride_pd, mask=mask_n[None, :], other=0.0)
        p2 = tl.load(p_base + 2 * stride_pd, mask=mask_n[None, :], other=0.0)

        r0 = t0 - p0
        r1 = t1 - p1
        r2 = t2 - p2

        min_dist = tl.full((BLOCK_M, BLOCK_N), float("inf"), dtype=tl.float32)

        for is0 in tl.static_range(-1, 2):
            s0 = float(is0)
            c0 = r0 - s0

            for is1 in tl.static_range(-1, 2):
                s1 = float(is1)
                c1 = r1 - s1

                for is2 in tl.static_range(-1, 2):
                    s2 = float(is2)
                    c2 = r2 - s2

                    dist = (
                        c0 * c0 * m00
                        + c1 * c1 * m11
                        + c2 * c2 * m22
                        + 2.0 * (c0 * c1 * m01 + c0 * c2 * m02 + c1 * c2 * m12)
                    )
                    min_dist = tl.minimum(min_dist, dist)

        c_out = (
            cost_ptr
            + pid_b * stride_cb
            + offs_m[:, None] * stride_cn
            + offs_n[None, :] * stride_cm
        )
        tl.store(c_out, min_dist, mask=(mask_m[:, None] & mask_n[None, :]))

    @triton.jit
    def _nearest_image_kernel(
        a_ptr,
        b_ptr,
        metric_ptr,
        image_out_ptr,
        stride_ab,
        stride_an,
        stride_ad,
        stride_bb,
        stride_bm,
        stride_bd,
        stride_mb,
        stride_mr,
        stride_mc,
        stride_ib,
        stride_in,
        stride_im,
        stride_id,
        n_a,
        n_b,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Search one tile of atom pairs and store the winning integer offsets.

        Strict improvement preserves the first image when distances tie,
        matching PyTorch argmin's lexicographic image order.
        """
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < n_a
        mask_n = offs_n < n_b

        m_ptr = metric_ptr + pid_b * stride_mb
        m00 = tl.load(m_ptr + 0 * stride_mr + 0 * stride_mc)
        m01 = tl.load(m_ptr + 0 * stride_mr + 1 * stride_mc)
        m02 = tl.load(m_ptr + 0 * stride_mr + 2 * stride_mc)
        m11 = tl.load(m_ptr + 1 * stride_mr + 1 * stride_mc)
        m12 = tl.load(m_ptr + 1 * stride_mr + 2 * stride_mc)
        m22 = tl.load(m_ptr + 2 * stride_mr + 2 * stride_mc)

        a_base = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_an
        a0 = tl.load(a_base + 0 * stride_ad, mask=mask_m[:, None], other=0.0)
        a1 = tl.load(a_base + 1 * stride_ad, mask=mask_m[:, None], other=0.0)
        a2 = tl.load(a_base + 2 * stride_ad, mask=mask_m[:, None], other=0.0)

        b_base = b_ptr + pid_b * stride_bb + offs_n[None, :] * stride_bm
        b0 = tl.load(b_base + 0 * stride_bd, mask=mask_n[None, :], other=0.0)
        b1 = tl.load(b_base + 1 * stride_bd, mask=mask_n[None, :], other=0.0)
        b2 = tl.load(b_base + 2 * stride_bd, mask=mask_n[None, :], other=0.0)

        r0 = a0 - b0
        r1 = a1 - b1
        r2 = a2 - b2

        min_dist = tl.full((BLOCK_M, BLOCK_N), float("inf"), dtype=tl.float32)
        best_shift0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        best_shift1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        best_shift2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for is0 in tl.static_range(-1, 2):
            s0 = float(is0)
            c0 = r0 - s0

            for is1 in tl.static_range(-1, 2):
                s1 = float(is1)
                c1 = r1 - s1

                for is2 in tl.static_range(-1, 2):
                    s2 = float(is2)
                    c2 = r2 - s2

                    dist = (
                        c0 * c0 * m00
                        + c1 * c1 * m11
                        + c2 * c2 * m22
                        + 2.0 * (c0 * c1 * m01 + c0 * c2 * m02 + c1 * c2 * m12)
                    )

                    update = dist < min_dist
                    min_dist = tl.where(update, dist, min_dist)
                    best_shift0 = tl.where(update, s0, best_shift0)
                    best_shift1 = tl.where(update, s1, best_shift1)
                    best_shift2 = tl.where(update, s2, best_shift2)

        image_base = (
            image_out_ptr
            + pid_b * stride_ib
            + offs_m[:, None] * stride_in
            + offs_n[None, :] * stride_im
        )
        mask2d = mask_m[:, None] & mask_n[None, :]

        tl.store(image_base + 0 * stride_id, best_shift0, mask=mask2d)
        tl.store(image_base + 1 * stride_id, best_shift1, mask=mask2d)
        tl.store(image_base + 2 * stride_id, best_shift2, mask=mask2d)

    @triton.jit
    def _assignment_kernel(
        cost_ptr, out_ptr, nrow_ptr, n, m, bn: tl.constexpr, bm: tl.constexpr
    ):
        """Solve one sample's rectangular assignment by shortest augmenting paths.

        Each real target row must occupy a distinct prediction column. Insert
        rows one at a time: find a cheapest path to a free column, then reroute
        the assignments along that path. The row/column potentials u/v track
        reduced costs; predecessor records the path for rerouting.
        Only the first nrow rows are real; remaining rows are padding.
        """
        b = tl.program_id(0)
        nrow = tl.load(nrow_ptr + b)
        rows, cols = tl.arange(0, bn), tl.arange(0, bm)
        col_ok = cols < m
        base = b * n * m

        u = tl.zeros((bn,), tl.float32)
        v = tl.zeros((bm,), tl.float32)
        column_for_row = tl.full((bn,), -1, tl.int32)
        row_for_column = tl.full((bm,), -1, tl.int32)

        # Insert each real target row, rerouting prior assignments along the augmenting path.
        for new_row in range(nrow):
            i, free_column = new_row, -1
            path_cost = tl.zeros((), tl.float32)
            visited_rows, visited_columns = rows < 0, cols < 0
            column_distance = tl.full((bm,), float("inf"), tl.float32)
            predecessor = tl.full((bm,), -1, tl.int32)
            row_distance = tl.zeros((bn,), tl.float32)

            while free_column < 0:
                visited_rows |= rows == i
                ui = tl.sum(tl.where(rows == i, u, 0.0))
                row = tl.load(
                    cost_ptr + base + i * m + cols, mask=col_ok, other=float("inf")
                )
                candidate = path_cost + row - ui - v

                update = (candidate < column_distance) & ~visited_columns & col_ok
                column_distance = tl.where(update, candidate, column_distance)
                predecessor = tl.where(update, i, predecessor)

                candidate = tl.where(
                    ~visited_columns & col_ok, column_distance, float("inf")
                )
                path_cost = tl.min(candidate, axis=0)
                j = tl.argmin(candidate, axis=0).to(tl.int32)
                visited_columns |= cols == j
                owner = tl.sum(tl.where(cols == j, row_for_column, 0))

                if owner < 0:
                    free_column = j
                else:
                    row_distance = tl.where(rows == owner, path_cost, row_distance)
                    i = owner

            # Update dual potentials only for vertices reached by the search.
            u = tl.where(rows == new_row, u + path_cost, u)
            u = tl.where(
                visited_rows & (rows != new_row) & (column_for_row >= 0),
                u + path_cost - row_distance,
                u,
            )
            v = tl.where(visited_columns, v - path_cost + column_distance, v)

            # Follow predecessors from the free column back to the inserted row.
            j, previous = free_column, -1

            while previous != new_row:
                previous = tl.sum(tl.where(cols == j, predecessor, 0))
                row_for_column = tl.where(cols == j, previous, row_for_column)
                old = tl.sum(tl.where(rows == previous, column_for_row, 0))
                column_for_row = tl.where(rows == previous, j, column_for_row)
                j = old

        tl.store(out_ptr + b * n + rows, column_for_row, mask=rows < nrow)


def _rows_to_slots(
    slot_of_atom: torch.Tensor, n_real: torch.Tensor, slots: int
) -> torch.Tensor:
    """Invert atom->slot assignments into slot->atom gather indices.

    Real atoms occupy the first n_real rows. Unmatched slots point to the first
    padding row (index n_real); a full sample has no unmatched slots. The extra
    scratch column absorbs padded atoms so they cannot overwrite real matches.
    """
    batch, atoms = slot_of_atom.shape
    atom_ids = torch.arange(atoms, device=slot_of_atom.device)[None].expand(batch, -1)
    real = atom_ids < n_real[:, None]

    scratch = torch.empty(
        batch, slots + 1, device=slot_of_atom.device, dtype=torch.long
    )
    scratch[:, :slots] = n_real.clamp(max=slots - 1)[:, None]
    scratch.scatter_(1, torch.where(real, slot_of_atom, slots), atom_ids)
    return scratch[:, :slots]


@torch.no_grad()
def hungarian(cost: torch.Tensor, n_real: torch.Tensor) -> torch.Tensor:
    """Match every real target atom to a distinct prediction slot at minimum cost.

    cost has shape [batch, padded targets, slots]; n_real counts the leading real
    targets in each sample. Return [batch, slots] target indices for gathering
    coordinates/types into prediction order. Unmatched slots gather the first
    padding target. For example, atoms 0->slot 2 and 1->slot 0 give [1, 2, 0],
    where target 2 is padding. The discrete assignment has no gradient.
    """
    batch, atoms, slots = cost.shape

    if triton is not None and cost.is_cuda:
        cost = cost.contiguous().float()
        # Nonfinite costs can prevent an augmenting-path search from terminating.
        # The assertion runs on CUDA, including during graph replay, without a CPU sync.
        torch._assert_async(torch.isfinite(cost).all(), "Nonfinite Hungarian costs")
        rows = torch.empty(batch, atoms, device=cost.device, dtype=torch.int32)

        _assignment_kernel[(batch,)](
            cost,
            rows,
            n_real.to(torch.int32),
            atoms,
            slots,
            bn=triton.next_power_of_2(atoms),
            bm=triton.next_power_of_2(slots),
            num_warps=4,
        )

        slot_of_atom = rows.long()
    else:
        cpu = cost.detach().cpu().numpy()
        rows = np.zeros((batch, atoms), dtype=np.int64)

        for b, n in enumerate(n_real.cpu().tolist()):
            rows[b, :n] = linear_sum_assignment(cpu[b, :n])[1]

        slot_of_atom = torch.from_numpy(rows).to(cost.device)

    return _rows_to_slots(slot_of_atom, n_real, slots)


@torch.no_grad()
def periodic_squared_distance(target, pred, metric):
    """Return [batch, targets, slots] minimum squared Cartesian distances.

    target and pred are fractional positions [batch, atoms, 3]; metric is the
    cell Gram matrix [batch, 3, 3]. For each pair, minimize dᵀ metric d over
    d = target - (pred + s), with s in {-1, 0, 1}³. This local image search is
    not exact for arbitrary skew cells. Costs are FP32 and have no gradient;
    reconstruction uses geometry.periodic_delta for coordinate gradients.
    """

    # Casting a sliced/expanded tensor may change strides; cast before taking them.
    target, pred, metric = target.float(), pred.float(), metric.float()
    batch, targets, _ = target.shape
    slots = pred.shape[1]

    if triton is not None and target.is_cuda:
        distance = torch.empty(batch, targets, slots, device=target.device)
        grid = (batch, triton.cdiv(targets, 32), triton.cdiv(slots, 32))

        _periodic_distance_kernel[grid](
            target,
            pred,
            metric,
            distance,
            *target.stride(),
            *pred.stride(),
            *metric.stride(),
            *distance.stride(),
            targets,
            slots,
            BLOCK_M=32,
            BLOCK_N=32,
        )

        return distance

    shifts = image_shifts(target)
    candidates = target[:, :, None, None] - pred[:, None, :, None] - shifts
    distance = torch.einsum("bnmki,bij,bnmkj->bnmk", candidates, metric, candidates)
    return distance.min(-1).values


@torch.no_grad()
def nearest_pairwise_image(
    a: torch.Tensor, b: torch.Tensor, metric: torch.Tensor
) -> torch.Tensor:
    """Choose which image of b is closest to each a in Cartesian distance.

    a/b contain fractional positions [batch, atoms, 3]; metric is the cell Gram
    matrix [batch, 3, 3]. Return offsets s of shape [batch, Na, Nb, 3] minimizing
    (a - b - s)ᵀ metric (a - b - s), for s in {-1, 0, 1}³. These are integer
    cell offsets stored as FP32, not displacement vectors. For example, in a
    cubic cell a_x=0.95 and b_x=0.05 choose s_x=1, giving displacement -0.10.
    Image selection has no gradient; subtract s from a-b in the calling code.
    """

    a, b, metric = a.float(), b.float(), metric.float()

    batch, targets, _ = a.shape
    slots = b.shape[1]

    if triton is not None and a.is_cuda:
        image = torch.empty(batch, targets, slots, 3, device=a.device)
        grid = (batch, triton.cdiv(targets, 32), triton.cdiv(slots, 32))

        _nearest_image_kernel[grid](
            a,
            b,
            metric,
            image,
            *a.stride(),
            *b.stride(),
            *metric.stride(),
            *image.stride(),
            targets,
            slots,
            BLOCK_M=32,
            BLOCK_N=32,
        )

        return image

    shifts = image_shifts(a)
    displacement = a[:, :, None] - b[:, None]
    candidates = displacement.unsqueeze(-2) - shifts
    distance = torch.einsum("b...ki,bij,b...kj->b...k", candidates, metric, candidates)
    return shifts[distance.argmin(-1)]
