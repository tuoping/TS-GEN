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
        elif atomic_numbers[i]==18:
            new_atomic_numbers[i] = 4
        else:
            raise Exception("Unrecognized type", atomic_numbers[i])
    atoms.set_atomic_numbers(new_atomic_numbers)
    indices_to_remove = [i for i, n in enumerate(atoms.get_atomic_numbers()) if n in element]
    del atoms[indices_to_remove]


from ase import Atoms
from ase.geometry.geometry import get_distances

def extract_positions_by_element(atoms, element_symbol):
    """
    Extracts the positions of atoms with a specific element symbol from an ASE Atoms object.

    Parameters:
    - atoms: ASE Atoms object
    - element_symbol: Symbol of the element (e.g., 'O' for oxygen)

    Returns:
    - positions: Numpy array of shape (N, 3) with the positions of the selected element
    """
    # Extract indices of atoms with the specified element
    element_indices = [i for i, atom in enumerate(atoms) if atom.symbol == element_symbol]

    # Extract the positions of those atoms
    positions = atoms.positions[element_indices]

    return positions


def calculate_rdf_pair(
    positions_a, 
    positions_b, 
    volume, 
    r_max, 
    bin_width, 
    cell=None, 
    pbc=None
):
    """
    Calculate the radial distribution function (RDF) between two sets
    of particles (A and B) in a (possibly) periodic cubic box.

    Parameters
    ----------
    positions_a : (N_a, 3) array_like
        Coordinates of element A.
    positions_b : (N_b, 3) array_like
        Coordinates of element B.
    volume : float
        Volume of the simulation box (for normalization). Assumed cubic,
        but only the volume is used here.
    r_max : float
        Maximum distance for the RDF calculation.
    bin_width : float
        Width of each RDF bin.
    cell : array_like of shape (3,) or (3,3), optional
        Box dimensions. If pbc is used, must provide either:
          - (3,) for orthorhombic boxes
          - (3,3) for full cell vectors
    pbc : (3,) of bool, optional
        Which directions are periodic (e.g., [True, True, True]).

    Returns
    -------
    r_values : (num_bins,) ndarray
        Midpoints of each radial bin.
    g_r : (num_bins,) ndarray
        The radial distribution function g(r).
    integral_g_r : (num_bins,) ndarray
        Cumulative coordination number up to distance r, normalized by
        (number_density * number_of_A_particles).
    """
    # Convert to arrays
    positions_a = np.asarray(positions_a)
    positions_b = np.asarray(positions_b)

    num_particles_a = len(positions_a)
    num_particles_b = len(positions_b)

    # Broadcast differences: shape => (N_a, N_b, 3)
    dr = positions_a[:, None, :] - positions_b[None, :, :]

    # If periodic boundary conditions are specified, apply minimal image convention
    if pbc is not None and cell is not None:
        # Handle the case of a (3,) cell (orthorhombic box)
        # or a (3,3) cell (general triclinic box).
        cell = np.asarray(cell)
        
        if cell.shape == (3,):  # Orthorhombic
            for dim in range(3):
                if pbc[dim]:
                    length = cell[dim]
                    dr[:, :, dim] -= length * np.round(dr[:, :, dim] / length)
        elif cell.shape == (3, 3):  # Triclinic or general
            # Solve for integer shifts n that minimize the distance in each dimension:
            #   dr_corrected = dr - n * cell_vectors
            # For an explanation, see references on minimal image in triclinic cells.
            #
            # A simple approximate approach is to project dr onto each cell vector,
            # round, and subtract. For large systems, a fully robust approach may need
            # more advanced logic. This snippet assumes cell is invertible:
            inv_cell = np.linalg.inv(cell)
            # Convert positions to fractional coords, round, shift back
            frac_shift = np.round(dr @ inv_cell)
            dr -= frac_shift @ cell
        else:
            raise ValueError("cell must be shape (3,) or (3,3) if pbc is provided.")

    # Compute the Euclidean distances
    distances = np.sqrt(np.sum(dr**2, axis=-1))

    # Filter out zero-distances (if any) and distances beyond r_max
    valid_mask = (distances > 0.0) & (distances < r_max)
    distances = distances[valid_mask]

    # Bin the distances
    num_bins = int(np.ceil(r_max / bin_width))
    rdf_hist, bin_edges = np.histogram(
        distances, bins=num_bins, range=(0, r_max)
    )
    
    # Radial bin centers (midpoints)
    r_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Volume of spherical shells: shell_volume = 4/3 pi ( (r+dr)^3 - r^3 )
    # We'll use bin_edges for a more accurate shell volume:
    shell_volumes = (4.0 / 3.0) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

    # Compute the "ideal" histogram = density * shell_volume * number_of_A
    number_density = num_particles_b / volume  # number density of B
    ideal_counts = shell_volumes * number_density * num_particles_a

    # RDF is ratio of actual to ideal
    g_r = rdf_hist / ideal_counts

    # For cumulative coordination number (integral RDF):
    # 1) compute cumulative sum of rdf_hist
    cdf_hist = np.cumsum(rdf_hist)
    # 2) at each bin i, total counts so far is cdf_hist[i]
    # 3) normalize by ( number_density * num_particles_a )
    integral_g_r = cdf_hist / (number_density * num_particles_a)

    return r_values, g_r, integral_g_r


class EquivariantTransformerDataset_CrCoNi(torch.utils.data.Dataset):
    def __init__(self, traj_dirname, cutoff, num_frames=None, random_starting_point=True, localmask=False, sim_condition=True, stage="train"):
        temperature = 300
        self.kT = temperature*8.617*10**-5
        self.max_num_edges = 4000
        self.cutoff = cutoff
        self.traj_filenames = []
        self.traj_initial = []
        self.traj_rdf = []
        self.traj_act_space = []
        self.idx_sources = []
        LSS_reward_pool = []
        for u1 in range(5):
            for k in range(100):
                idx = u1*100+k
                if stage == "train":
                    criterion = (k%3 <= 1)
                elif stage == "val":
                    criterion = (k%3 > 1)
                elif stage == "save":
                    self.traj_filenames.append(os.path.join(traj_dirname, f"testing-{u1}-{k}.extxyz"))
                    criterion = False
                else:
                    raise Exception(f"Wrong stage str {stage}")
                if criterion:
                    self.traj_filenames.append(os.path.join(traj_dirname, f"dataset-{u1*100+k}.pt"))
                    self.traj_rdf.append(os.path.join(traj_dirname, f"RDF-{u1*100+k}.pt"))
                    self.traj_initial.append(os.path.join(traj_dirname, f"initial-{u1*100+k}.xyz"))
                    self.traj_act_space.append(os.path.join(traj_dirname, f"act_space-{u1}-{k}.txt"))
                    self.idx_sources.append(idx)
                    # _dataset = torch.load(self.traj_filenames[-1], weights_only=False)
                    # LSS_reward_pool.append(torch.stack([data.E_now for data in _dataset]))

        self.num_frames = num_frames
        self.stage = stage
        self.localmask = localmask
        self.random_starting_point = random_starting_point
        self.sim_condition = sim_condition
        # self.LSS_reward_pool = torch.concat(LSS_reward_pool, dim=0)
        # self.partition = torch.logsumexp(-self.LSS_reward_pool, dim=0)
    
    def __len__(self):
        return len(self.traj_filenames)
    
    def __getitem__(self, idx):
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
            dataset_g_r= []
            for atoms in atoms_list:
                r_, g_r, integral_g_r = calculate_rdf_pair(atoms.positions, atoms.positions, atoms.get_volume(), self.cutoff, 0.1, cell=atoms.cell, pbc=True)
                dataset_g_r.append(torch.from_numpy(np.stack([r_, g_r])))
            torch.save(dataset_g_r, f'data/CrCoNi_data/RDF-{idx}.pt')

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
            if not os.path.exists("data/CrCoNi_data/"):
                os.makedirs("data/CrCoNi_data/")
            torch.save(dataset, f'data/CrCoNi_data/dataset-{idx}.pt')
            ase.io.write(f"data/CrCoNi_data/initial-{idx}.xyz", atoms_list[0])
            return len(dataset)
        else:
            idx_source = self.idx_sources[idx]
            _dataset = torch.load(self.traj_filenames[idx], weights_only=False)
            # _RDF = torch.load(self.traj_rdf[idx], weights_only=False)
            # assert _RDF[0].shape == (2,35)
            act_space = torch.from_numpy(np.loadtxt(self.traj_act_space[idx])).to(torch.long)
            LSS_reward_pool = torch.stack([data.E_now for data in _dataset])
            if self.random_starting_point:
                start_i_traj = np.random.randint(0, len(_dataset)-self.num_frames-1, 1)[0]
            else:
                start_i_traj = 0
            if self.num_frames is None:
                self.num_frames = len(_dataset)
            end_i_traj = start_i_traj+self.num_frames
            dataset = _dataset[start_i_traj:end_i_traj]
            dataset_next = _dataset[start_i_traj+1:end_i_traj+1]
            num_atoms = dataset[0].num_atoms
            if self.sim_condition:
                TKS_reward = torch.stack([-data.E_barrier+data.freq*self.kT for data in dataset])  # T
            else:
                LSS_reward = torch.stack([data.E_now for data in dataset]) # T

            x = torch.stack([data.pos for data in dataset])
            T,L,_ = x.shape
            # log_mask = -LSS_reward - self.partition
            ### Normalize over each trajectory
            if self.sim_condition:
                log_mask = -TKS_reward
            else:
                log_mask = -LSS_reward - torch.logsumexp(-LSS_reward_pool, dim=0)
            _mask = torch.exp(log_mask)[:,None] # T,L
            _v_mask = _mask.unsqueeze(-1).expand(-1,-1,3) # T,L,3
            _h_mask = _mask.unsqueeze(-1).expand(-1,-1,5) # T,L,5
            if self.localmask:
                # disp_mask = (torch.stack([data.disp for data in dataset]).norm(dim=-1)>1).unsqueeze(-1)
                mask = torch.ones([T,L])
                v_mask = torch.ones([T,L,3])
                h_mask = torch.ones([T,L,5])
                for i_traj in range(start_i_traj, end_i_traj):
                    disp_mask = torch.zeros([L])
                    act_space_i = act_space[i_traj]
                    disp_mask[act_space_i] = 1
                    mask[i_traj-start_i_traj] = _mask[i_traj-start_i_traj]*disp_mask.unsqueeze(0)
                    h_mask[i_traj-start_i_traj] = _h_mask[i_traj-start_i_traj]*disp_mask.unsqueeze(0).unsqueeze(-1)
                    v_mask[i_traj-start_i_traj] = _v_mask[i_traj-start_i_traj]*disp_mask.unsqueeze(0).unsqueeze(-1)
            else:
                mask = _mask
                v_mask = _v_mask
                h_mask = _h_mask

            if self.sim_condition:
                return {
                    "idx": idx_source,
                    "name": "CrCoNi",
                    "species": torch.stack([data.z for data in dataset]),
                    "species_next": torch.stack([data.z for data in dataset_next]),
                    "x": torch.stack([data.pos for data in dataset]),
                    'x_next': torch.stack([data.pos for data in dataset_next]),
                    "cell": torch.stack([data.cell for data in dataset]),
                    "num_atoms": torch.stack([data.num_atoms for data in dataset]),
                    "mask": mask,
                    "v_mask": v_mask,
                    "h_mask": h_mask,
                    "e_now": torch.stack([data.E_now for data in dataset]),
                }
            else:
                return {
                    "idx": idx_source,
                    "name": "CrCoNi",
                    "species": torch.stack([data.z for data in dataset]),
                    # "species_next": torch.stack([data.z for data in dataset_next]),
                    "x": torch.stack([data.pos for data in dataset]),
                    "cell": torch.stack([data.cell for data in dataset]),
                    "num_atoms": torch.stack([data.num_atoms for data in dataset]),
                    "mask": mask,
                    "v_mask": v_mask,
                    "h_mask": h_mask,
                    "e_now": torch.stack([data.E_now for data in dataset]),
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
        


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, traj_dirname, cutoff, num_frames=None, random_starting_point=True, localmask=False, stage="train"):
        self.cutoff = cutoff
        self.h_traj_filenames = []
        self.v_traj_filenames = []
        self.traj_filenames = []
        for u1 in range(5):
            for k in range(100):
                idx = u1*100+k
                if stage == "train":
                    criterion = (k%3 <= 1)
                elif stage == "val":
                    criterion = (k%3 > 1)
                else:
                    raise Exception(f"Wrong stage str {stage}")

                if criterion:
                    self.h_traj_filenames.append(os.path.join(traj_dirname, f"encoded_h-{u1*100+k}.pt"))
                    self.v_traj_filenames.append(os.path.join(traj_dirname, f"encoded_v-{u1*100+k}.pt"))
                    self.traj_filenames.append(os.path.join(traj_dirname, f"dataset-{u1*100+k}.pt"))

        self.num_frames = num_frames
        self.stage = stage
        self.localmask = False
        self.random_starting_point = random_starting_point
        # self.LSS_reward_pool = torch.concat(LSS_reward_pool, dim=0)
        # self.partition = torch.logsumexp(-self.LSS_reward_pool, dim=0)
    
    def __len__(self):
        return len(self.traj_filenames)
    
    def __getitem__(self, idx):
        idx = idx % len(self.traj_filenames)

        _dataset = torch.load(self.traj_filenames[idx], weights_only=False)
        _dataset_h = torch.load(self.h_traj_filenames[idx], weights_only=False).squeeze(0)
        _dataset_v = torch.load(self.v_traj_filenames[idx], weights_only=False).squeeze(0)
        assert _dataset_h.shape[1] == _dataset[0].pos.shape[0], f"dataset_h shape {_dataset_h.shape} should be same as dataset shape {torch.stack([data.pos for data in _dataset]).shape}"

        if self.random_starting_point:
            start_i_traj = np.random.randint(0, len(_dataset_h)-self.num_frames, 1)[0]
        else:
            start_i_traj = 0
        if self.num_frames is None:
            self.num_frames = len(dataset_h)
        end_i_traj = start_i_traj+self.num_frames
        dataset_h = _dataset_h[start_i_traj:end_i_traj]
        dataset_v = _dataset_v[start_i_traj:end_i_traj]
        dataset = _dataset[start_i_traj:end_i_traj]
        # cell_tensor = torch.stack([data.cell for data in dataset]).reshape(self.num_frames, 9) # T,3,3
        cell = torch.stack([torch.tensor([torch.linalg.norm(data.cell[0]), 
                             torch.linalg.norm(data.cell[1]),
                             torch.linalg.norm(data.cell[2]),
                             torch.acos(torch.dot(data.cell[0], data.cell[1])/(torch.linalg.norm(data.cell[0])*torch.linalg.norm(data.cell[1])))/torch.pi*180,
                             torch.acos(torch.dot(data.cell[1], data.cell[2])/(torch.linalg.norm(data.cell[1])*torch.linalg.norm(data.cell[2])))/torch.pi*180,
                             torch.acos(torch.dot(data.cell[0], data.cell[2])/(torch.linalg.norm(data.cell[0])*torch.linalg.norm(data.cell[2])))/torch.pi*180,
                            ])
                             for data in dataset])
        
        return {
            "name": "CrCoNi_latent",
            "v": dataset_v,
            "h": dataset_h,
            "species": torch.stack([data.z for data in dataset]),
            "x": torch.stack([data.pos for data in dataset]),
            'frac_x': torch.stack([data.frac_pos for data in dataset]),
            "cell": cell,
        }