import os
import h5py
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    # Specify the folders where your .h5 files are located and where you want the .npz files saved
    input_folder = None
    output_folder = None
    os.makedirs(output_folder, exist_ok=True)

    # List the names of your 8 .h5 files
    h5_files = {
        1: 'ani_gdb_s01.h5',
        2: 'ani_gdb_s02.h5',
        3: 'ani_gdb_s03.h5',
        4: 'ani_gdb_s04.h5',
        5: 'ani_gdb_s05.h5',
        6: 'ani_gdb_s06.h5',
        7: 'ani_gdb_s07.h5',
        8: 'ani_gdb_s08.h5'
    }

    atom2num = {
        'H': 1,
        'C': 6,
        'N': 7,
        'O': 8
    }

    self_energies = {
        1: -0.500607632585,  # Hydrogen
        6: -37.8302333826,   # Carbon
        7: -54.5680045287,   # Nitrogen
        8: -75.0362229210    # Oxygen
    }

    atomic_weights = {
        1: 1.00794,   # Hydrogen
        6: 12.0107,   # Carbon
        7: 14.0067,   # Nitrogen
        8: 15.9994,   # Oxygen
    }

    max_atom_count = 26
    padded_distances = np.full((max_atom_count, max_atom_count), -999., dtype=np.float32)

    for heavy_atoms, file_name in h5_files.items():
        input_path = os.path.join(input_folder, file_name)

        with h5py.File(input_path, 'r') as h5f:
            for key, value in tqdm(h5f[f"gdb11_s0{heavy_atoms}"].items()):
                coords = np.array(value['coordinates'])
                distances = np.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], axis=-1)

                atom_types = [atom.decode('utf-8') for atom in value['species']]
                atom_types = np.array([atom2num[atom] for atom in atom_types])

                molecule_info = np.zeros((coords.shape[0], coords.shape[1], distances.shape[-1] * 2), dtype=np.float32)
                molecule_info[:, :, ::2] = distances
                molecule_info[:, :, 1::2] = atom_types


                padded_molecule_info = np.zeros((coords.shape[0], coords.shape[1], max_atom_count * 2), dtype=np.float32)
                padded_molecule_info[:, :, :molecule_info.shape[-1]] = molecule_info
                padded_molecule_info[:, :, molecule_info.shape[-1]::2] = padded_distances[:molecule_info.shape[1], molecule_info.shape[1]:]

                # Get molecule self energy
                self_energy = sum([self_energies[atom] for atom in atom_types])

                data = {
                    'mask': atom_types,
                    'molecule_info': padded_molecule_info,
                    'energy': (np.array(value['energies']) - self_energy) * 627.509,
                }


                file_name = key.split('_')[1].replace('-', '_') + ".npz"
                output_path = os.path.join(output_folder, file_name)
                np.savez(output_path, **data)
