"""Relax QMOF150 structures with eSEN-30M-OAM + D3(BJ) dispersion in batched TorchSim.

Run with the relaxation environment (python -m eval.setup esen):
    cache/environments/esen/bin/python -m eval.relax --input samples.pt --out relaxed/

LBFGS with a Frechet cell filter; convergence includes cell forces. The output
directory holds one CIF per relaxed structure and a manifest recording every
failure, so it can be passed to the other evaluators as --input.
"""

import argparse
import gc
import json

import numpy as np
import torch
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

from eval.common import read_structures
from eval.setup import CACHE

MODEL = CACHE / "models/esen_30m_oam.pt"
FMAX, STEPS, MAXSTEP, BATCH_SIZE = 0.02, 200, 0.1, 32


def esen_d3_model(model_path, device):
    """eSEN-30M-OAM with D3(BJ) dispersion as a TorchSim model."""
    from ase.units import Bohr
    from fairchem.core import OCPCalculator
    from torch_dftd.dftd3_xc_params import get_dftd3_default_params
    from torch_dftd.nn.dftd3_module import DFTD3Module
    from torch_geometric.data import Batch, Data
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torch_nl_n2

    class ESEND3(ModelInterface):
        def __init__(self):
            super().__init__()
            self.trainer = OCPCalculator(
                checkpoint_path=model_path, cpu=device == "cpu", seed=0
            ).trainer
            self.trainer.model.requires_grad_(False)
            self._device = torch.device(device)
            self._dtype = torch.float32
            self._compute_forces = True
            self._compute_stress = True
            self.dispersion = DFTD3Module(
                get_dftd3_default_params("bj", "pbe", old=False),
                cutoff=40 * Bohr,
                cnthr=40 * Bohr,
                abc=False,
                dtype=self.dtype,
                bidirectional=True,
                cutoff_smoothing="none",
            ).to(self.device)

        def forward(self, state, **_):
            cell = state.row_vector_cell.detach()
            fractional = torch.einsum(
                "ni,nij->nj", state.positions.detach(), cell.inverse()[state.system_idx]
            )
            wrapped = torch.einsum(
                "ni,nij->nj", fractional.remainder(1), cell[state.system_idx]
            )
            sizes = state.n_atoms_per_system.tolist()
            batch = Batch.from_data_list(
                [
                    Data(
                        pos=pos.clone(),
                        cell=lattice[None].clone(),
                        atomic_numbers=numbers,
                        natoms=len(pos),
                        fixed=torch.zeros_like(numbers),
                        tags=torch.zeros_like(numbers),
                    )
                    for pos, lattice, numbers in zip(
                        wrapped.split(sizes), cell, state.atomic_numbers.split(sizes)
                    )
                ]
            )
            prediction = self.trainer.predict(batch, per_image=False, disable_tqdm=True)
            output = {
                "energy": prediction["energy"].detach().reshape(-1),
                "forces": prediction["forces"].detach(),
                "stress": prediction["stress"].detach().reshape(-1, 3, 3),
            }
            edges, edge_system, shifts = torch_nl_n2(
                wrapped,
                cell,
                state.pbc,
                cell.new_tensor(self.dispersion.cutoff),
                state.system_idx,
            )

            # Homogeneous strain differentiates both positions and periodic images.
            # Its energy derivative / volume has ASE's tensile-positive stress sign.
            with torch.enable_grad():
                positions = wrapped.detach().requires_grad_(True)
                strain = torch.zeros_like(cell, requires_grad=True)
                deformation = torch.eye(3, device=self.device) + strain
                strained_cell = cell @ deformation
                strained_positions = torch.einsum(
                    "ni,nij->nj", positions, deformation[state.system_idx]
                )
                shift_positions = torch.einsum(
                    "ei,eij->ej", shifts.to(self.dtype), strained_cell[edge_system]
                )
                energy = self.dispersion.calc_energy_batch(
                    state.atomic_numbers,
                    strained_positions,
                    edges,
                    cell=strained_cell,
                    pbc=state.pbc,
                    shift_pos=shift_positions,
                    batch=state.system_idx,
                    batch_edge=edge_system,
                    damping="bj",
                )
                position_gradient, strain_gradient = torch.autograd.grad(
                    energy.sum(), (positions, strain)
                )

            output["energy"] += energy.detach()
            output["forces"] -= position_gradient.detach()
            output["stress"] += (
                strain_gradient.detach() / cell.det().abs()[:, None, None]
            )
            return output

    return ESEND3()


def unstable(output, state):
    """Nonfinite energy, force, or stress, or a degenerate cell."""
    energies = output["energy"]
    force_max = torch.zeros(
        state.n_systems, device=energies.device, dtype=energies.dtype
    )

    force_max.scatter_reduce_(
        0,
        state.system_idx,
        torch.linalg.vector_norm(output["forces"], dim=1),
        reduce="amax",
    )
    bad = ~torch.isfinite(energies) | ~torch.isfinite(force_max)
    if "stress" in output:
        bad |= ~torch.isfinite(output["stress"]).all(dim=(-2, -1))
    bad |= ~torch.isfinite(state.cell).all(dim=(-2, -1)) | (state.cell.det().abs() == 0)
    return bad


def optimize_batch(atoms, calculator):
    import torch_sim as ts

    force_convergence = ts.generate_force_convergence_fn(FMAX, include_cell_forces=True)
    state = ts.io.atoms_to_state(
        atoms, device=calculator.device, dtype=calculator.dtype
    )
    initial = calculator(state)
    initially_bad = unstable(initial, state)

    good = torch.nonzero(~initially_bad).flatten().tolist()

    if not good:
        return initial, good, None

    state = state[good]

    def converged(current, last_energy=None):
        output = {
            "energy": current.energy,
            "forces": current.forces,
            "stress": current.stress,
        }
        return force_convergence(current, last_energy) | unstable(output, current)

    from torch_sim.optimizers.lbfgs import lbfgs_init, lbfgs_step

    final = ts.optimize(
        state,
        calculator,
        optimizer=(lbfgs_init, lbfgs_step),
        convergence_fn=converged,
        max_steps=STEPS,
        steps_between_swaps=1,
        init_kwargs={
            "cell_filter": ts.CellFilter.frechet,
            "alpha": 70.0,
            "step_size": 1.0,
        },
        max_step=MAXSTEP,
    )

    return initial, good, final


def relax(source, out, device="cuda"):
    import torch_sim as ts

    calculator = esen_d3_model(str(MODEL), device)
    force_convergence = ts.generate_force_convergence_fn(FMAX, include_cell_forces=True)
    candidates = [
        (
            identifier,
            num_atoms,
            AseAtomsAdaptor.get_atoms(structure) if structure is not None else None,
            error,
        )
        for identifier, num_atoms, structure, error in read_structures(source)
    ]
    records = [
        {"id": identifier, "num_atoms": num_atoms, "error": error, "converged": False}
        for identifier, num_atoms, _, error in candidates
    ]
    out.mkdir(parents=True)

    pending = [
        index for index, (_, _, atoms, _) in enumerate(candidates) if atoms is not None
    ]
    batch_size, completed, out_of_memory = BATCH_SIZE, 0, False
    while completed < len(pending):
        if out_of_memory:
            # Release the failed graphs only once the exception traceback is gone.
            gc.collect()
            torch.cuda.empty_cache()
            out_of_memory = False
        indices = pending[completed : completed + batch_size]
        atoms = [candidates[index][2] for index in indices]
        initial_positions = [item.positions.copy() for item in atoms]

        try:
            initial, good, final = optimize_batch(atoms, calculator)
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True
            if len(indices) == 1:
                records[indices[0]]["error"] = (
                    "OutOfMemory: single structure exceeds GPU memory"
                )
                completed += 1
            else:
                batch_size = max(1, len(indices) // 2)
            continue

        for local in set(range(len(indices))) - set(good):
            records[indices[local]]["error"] = (
                "UnstableStructure: nonfinite energy, force, stress or invalid cell"
            )

        if final is not None:
            final_output = {
                "energy": final.energy,
                "forces": final.forces,
                "stress": final.stress,
            }
            bad = unstable(final_output, final).cpu().numpy()
            converged = force_convergence(final, None).cpu().numpy()
            final_atoms = ts.io.state_to_atoms(final)

            for final_local, original_local in enumerate(good):
                record = records[indices[original_local]]
                if bad[final_local]:
                    record["error"] = (
                        "UnstableStructure: Unphysical force or energy during relaxation"
                    )
                    continue

                relaxed = final_atoms[final_local]
                displacement = relaxed.positions - initial_positions[original_local]
                write(out / f"{record['id']}.cif", relaxed, format="cif")
                record.update(
                    file=f"{record['id']}.cif",
                    converged=bool(converged[final_local]),
                    initial_energy_eV=float(initial["energy"][original_local]),
                    energy_eV=float(final.energy[final_local]),
                    steps=int(final.n_iter[final_local]),
                    # Fixed atom correspondence; includes the cell deformation.
                    cartesian_displacement_rmsd=float(
                        np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
                    ),
                )

        completed += len(indices)
        print(f"relaxed {completed}/{len(pending)}", flush=True)

    protocol = {
        "model": str(MODEL),
        "fmax": FMAX,
        "steps": STEPS,
        "maxstep": MAXSTEP,
        "optimizer": "TorchSim LBFGS, Frechet cell filter",
    }
    (out / "manifest.json").write_text(
        json.dumps(
            {"protocol": protocol, "complete": True, "records": records}, indent=2
        )
        + "\n"
    )
    return out


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", dest="source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    print(relax(**vars(parser.parse_args())))
