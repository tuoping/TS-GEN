import torch
from .rigid_utils import Rigid
from .residue_constants import restype_order
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
from mdgen.residue_constants import restype_order, restype_atom37_mask

class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1):
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres = self.df.seqres[name]
        else:
            name = self.args.overfit_peptide
            seqres = name

        if self.args.atlas:
            i = np.random.randint(1, 4)
            full_name = f"{name}_R{i}"
        else:
            full_name = name
        arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
        arr_cond = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}_cond_{self.args.suffix}.npy', 'r')
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]
            arr_cond = arr_cond[::self.args.frame_interval]
        
        frame_start = np.random.choice(np.arange(arr.shape[0] - self.args.num_frames))
        if self.args.overfit_frame:
            frame_start = 0
        end = frame_start + self.args.num_frames
        # arr = np.copy(arr[frame_start:end]) * 10 # convert to angstroms
        arr = np.copy(arr[frame_start:end]).astype(np.float32) # / 10.0 # convert to nm
        arr_cond = np.copy(arr_cond[frame_start:end]).astype(np.float32)
        if self.args.copy_frames:
            arr[1:] = arr[0]
            arr_cond[1:] = arr_cond[0]

        # arr should be in ANGSTROMS
        frames = atom14_to_frames(torch.from_numpy(arr))
        seqres = np.array([restype_order[c] for c in seqres])
        aatype = torch.from_numpy(seqres)[None].expand(self.args.num_frames, -1)
        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()
        
        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)
        
        if self.args.no_frames:
            return {
                'name': full_name,
                'frame_start': frame_start,
                'atom37': atom37,
                'seqres': seqres,
                'mask': restype_atom37_mask[seqres], # (L,)
            }
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        
        torsion_mask = torsion_mask[0]
        
        if self.args.atlas:
            if L > self.args.crop:
                start = np.random.randint(0, L - self.args.crop + 1)
                torsions = torsions[:,start:start+self.args.crop]
                frames = frames[:,start:start+self.args.crop]
                seqres = seqres[start:start+self.args.crop]
                mask = mask[start:start+self.args.crop]
                torsion_mask = torsion_mask[start:start+self.args.crop]
                
            
            elif L < self.args.crop:
                pad = self.args.crop - L
                frames = Rigid.cat([
                    frames, 
                    Rigid.identity((self.args.num_frames, pad), requires_grad=False, fmt='rot_mat')
                ], 1)
                mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
                seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
                torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)], 1)
                torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])

        return {
            'name': full_name,
            'frame_start': frame_start,
            'torsions': torsions,
            'torsion_mask': torsion_mask,
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres,
            'mask': mask, # (L,)
            "x_cond": torch.from_numpy(arr_cond[:2]).float(),
            "x_cond_mask": torch.ones(2, dtype=torch.float32),
        }


import os
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
import ase.io

from copy import deepcopy

def remove_element(atoms, element=[]):
    """
    Remove atoms of a specific element from the Atoms object.
    """
    atomic_numbers = atoms.get_atomic_numbers()
    new_atomic_numbers = deepcopy(atomic_numbers)
    for i in range(len(atomic_numbers)):
        if atomic_numbers[i]==24:
            new_atomic_numbers[i] = 1
        elif atomic_numbers[i]==27:
            new_atomic_numbers[i] = 2
        elif atomic_numbers[i]==28:
            new_atomic_numbers[i] = 3
        else:
            raise Exception("Unrecognized type", atomic_numbers[i])
    atoms.set_atomic_numbers(new_atomic_numbers)
    indices_to_remove = [i for i, n in enumerate(atoms.get_atomic_numbers()) if n in element]
    del atoms[indices_to_remove]

from ase.geometry.geometry import get_distances

class EquivariantTransformerDataset_CrCoNi(torch.utils.data.Dataset):
    def __init__(self, traj_dirname, cutoff, num_frames=None, stage="train"):
        self.max_num_edges = 4000
        self.cutoff = cutoff
        self.traj_filenames = []
        self.traj_initial = []
        for u1 in range(5):
            for k in range(100):
                if stage == "train":
                    criterion = (k%3 <= 1)
                    if criterion:
                        self.traj_filenames.append(os.path.join(traj_dirname, f"dataset-{u1*100+k}.pt"))
                        self.traj_initial.append(os.path.join(traj_dirname, f"initial-{u1*100+k}.xyz"))
                elif stage == "val":
                    criterion = (k%3 > 1)
                    if criterion:
                        self.traj_filenames.append(os.path.join(traj_dirname, f"dataset-{u1*100+k}.pt"))
                        self.traj_initial.append(os.path.join(traj_dirname, f"initial-{u1*100+k}.xyz"))
                elif stage == "save":
                    self.traj_filenames.append(os.path.join(traj_dirname, f"testing-{u1}-{k}.extxyz"))
                else:
                    raise Exception(f"Wrong stage str {stage}")
        self.num_frames = num_frames
        self.stage = stage
    
    def __len__(self):
        return len(self.traj_filenames)
    
    def __getitem__(self, idx, random_starting_point=False):
        idx = idx % len(self.traj_filenames)
        if self.stage == "save":
            atoms_list = ase.io.read(self.traj_filenames[idx], index=":")
            num_atoms = len(atoms_list[0])
            for atoms in atoms_list: 
                remove_element(atoms)
                atoms.wrap()
                if len(atoms) != num_atoms:
                    print("Traj filename", self.traj_filenames[idx])
                    raise Exception("Atoms length mismatch", len(atoms), num_atoms)

            mask = torch.ones((num_atoms,), dtype=torch.float32)
            v_mask = torch.ones((num_atoms, 3), dtype=torch.float32)               

            # Onehot encoder for atom type
            unique_numbers = np.concatenate([np.unique(atoms.numbers) for atoms in atoms_list])        
            atom_encoder = OneHotEncoder(sparse_output=False)
            atom_encoder.fit(unique_numbers.reshape(-1, 1))
            start_i_traj = 0
            end_i_traj = len(atoms_list)
            dataset = []
            for atoms in atoms_list[start_i_traj:end_i_traj]:
                inv_cell = np.linalg.pinv(np.array(atoms.cell))
                z = atom_encoder.transform(atoms.numbers.reshape(-1, 1))
                padded_z = np.zeros((num_atoms, 5))
                padded_z[:, :z.shape[1]] = z
                num_atoms = len(atoms)

                data = Data(
                    z          = torch.tensor(padded_z,               dtype=torch.float32),
                    pos        = torch.tensor(atoms.positions - np.ones(3)*0.5 @ atoms.cell, dtype=torch.float32),
                    frac_pos        = torch.tensor(atoms.positions @ inv_cell - np.ones(3)*0.5, dtype=torch.float32),
                    cell       = torch.tensor(np.array(atoms.cell), dtype=torch.float32),
                    freq = torch.tensor(atoms.info["freq"], dtype=torch.float32),
                    E_barrier = torch.tensor(atoms.info["E_barrier"], dtype=torch.float32),
                    E_now = torch.tensor(atoms.info["E_now"], dtype=torch.float32),
                    E_next = torch.tensor(atoms.info["E_next"], dtype=torch.float32),
                    disp = torch.tensor(atoms.arrays["disp"], dtype=torch.float32),
                    num_atoms = torch.tensor(num_atoms, dtype=torch.long),
                )
                dataset.append(data.clone())
            if not os.path.exists("data/CrCoNi_data_posdisp/"):
                os.makedirs("data/CrCoNi_data_posdisp/")
            torch.save(dataset, f'data/CrCoNi_data_posdisp/dataset-{idx}.pt')
            ase.io.write(f"data/CrCoNi_data_posdisp/initial-{idx}.xyz", atoms_list[0])
            return len(dataset)
        else:
            dataset = torch.load(self.traj_filenames[idx], weights_only=False)
            if random_starting_point:
                start_i_traj = np.random.randint(0, len(dataset), 1)[0]
            else:
                start_i_traj = 0
            if self.num_frames is None:
                self.num_frames = len(dataset)
            end_i_traj = min(start_i_traj+self.num_frames, len(dataset))

            num_atoms = dataset[0].num_atoms
            mask = torch.ones((num_atoms,), dtype=torch.float32)
            v_mask = torch.ones((num_atoms, 3), dtype=torch.float32)   

            dataset = dataset[start_i_traj:end_i_traj]
            return {
                "name": "CrCoNi",
                "species": torch.stack([data.z for data in dataset]),
                "x": torch.stack([data.pos for data in dataset]),
                "cell": torch.stack([data.cell for data in dataset]),
                "num_atoms": torch.stack([data.num_atoms for data in dataset]),
                "freq": torch.stack([data.freq for data in dataset]),
                "E_barrier": torch.stack([data.E_barrier for data in dataset]),
                "E_now": torch.stack([data.E_now for data in dataset]),
                "E_next": torch.stack([data.E_next for data in dataset]),
                "disp": torch.stack([data.disp for data in dataset]),
                "mask": mask,
                "v_mask": v_mask
            }

            
        

class GemnetDataset_CrCoNi(torch.utils.data.Dataset):
    def __init__(self, traj_dirname, cutoff, stage="train"):
        self.cutoff = cutoff
        self.traj_filenames = []
        for u1 in range(5):
            for k in range(100):
                if stage == "train":
                    criterion = (k%3 <= 1)
                    if criterion:
                        self.traj_filenames.append(os.path.join(traj_dirname, f"training-{u1}-{k}.extxyz"))
                elif stage == "val":
                    criterion = (k%3 > 1)
                    if criterion:
                        self.traj_filenames.append(os.path.join(traj_dirname, f"testing-{u1}-{k}.extxyz"))
    
    def __len__(self):
        return len(self.traj_filenames)
    
    def __getitem__(self, idx):
        idx = idx % len(self.traj_filenames)
        atoms_list = ase.io.read(self.traj_filenames[idx], index=":")
        num_atoms = len(atoms_list[0])
        for atoms in atoms_list: 
            remove_element(atoms)
            atoms.wrap()
            if len(atoms) != num_atoms:
                print("Traj filename", self.traj_filenames[idx])
                raise Exception("Atoms length mismatch", len(atoms), num_atoms)
        raise RuntimeError("GemnetDataset_CrCoNi is not implemented yet")    
        