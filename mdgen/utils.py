import numpy as np
import scipy
import torch
from . import protein
from .geometry import atom14_to_atom37

def get_offsets(ref_frame, rigids):
    B, T, L = rigids.shape
    if T > 500000:
        offsets1 = ref_frame.invert().compose(rigids[:, : 500000]).to_tensor_7()
        offsets2 = ref_frame.invert().compose(rigids[:, 500000:]).to_tensor_7()
        return torch.cat([offsets1, offsets2], 1)
    else:
        return ref_frame.invert().compose(rigids).to_tensor_7()

def simplex_proj(seq):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)

class DirichletConditionalFlow:
    def __init__(self, K=5, alpha_min=1, alpha_max=100, alpha_spacing=0.01):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.alpha_spacing = alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        # if the bs is close to the edge of the simplex in one of its entries, then we want the c factor to be 0 for high alphas.
        # That is the rationale for why we return 0s in the case of an overflow.

        beta = scipy.special.beta(alpha, self.K - 1) # betafunction(alpha, K-1)
        beta_div = np.where(bs < 1, beta / ((1 - bs) ** (self.K - 1)), 0)
        beta_div_full = np.where((bs ** (alpha - 1)) > 0, beta_div / (bs ** (alpha - 1)), 0)

        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)

        final = interp * beta_div_full
        return final

def atom14_to_pdb(atom14, aatype, path):
    prots = []
    for i, pos in enumerate(atom14):
        pos = atom14_to_atom37(pos, aatype)
        prots.append(create_full_prot(pos, aatype=aatype))
    with open(path, 'w') as f:
        f.write(prots_to_pdb(prots))


def create_full_prot(
        atom37: np.ndarray,
        aatype=None,
        b_factors=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    atom37_mask = np.sum(np.abs(atom37), axis=-1) > 1e-7
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    chain_index = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index
    )


def prots_to_pdb(prots):
    ss = ''
    for i, prot in enumerate(prots):
        ss += f'MODEL {i}\n'
        prot = protein.to_pdb(prot)
        ss += '\n'.join(prot.split('\n')[2:-3])
        ss += '\nENDMDL\n'
    return ss


def _low_symm_couplings(material_type, min_coupling):
    if material_type == "Metals":
        weak_coupling = min([80, min_coupling])
    elif material_type == "Ceramics":
        weak_coupling = min([200, min_coupling])
    elif material_type == "Polymers":
        weak_coupling = min([1, min_coupling])
    else:
        raise ValueError(f"Wrong material_type {material_type}, supposed be \'Metals\' or \'Ceramics\' or \'Polymers\'")
    C14 = np.random.normal(loc=0, scale=0.1*weak_coupling)
    return C14


def Stiffness_from_modulus(SGn, bulk_dict, shear_dict, material_type="Metals"):
    match material_type:
        case "Metals":
            min_coupling = 80
            if bulk_dict is None:
                C11 = np.random.normal(200, 0.1*200)
                C12 = np.random.normal(100, 0.1*100)
                C44 = np.random.normal(60, 0.1*80)

        case "Ceramics":
            min_coupling = 200
            if bulk_dict is None:
                C11 = np.random.normal(400, 0.1*400)
                C12 = np.random.normal(200, 0.1*200)
                C44 = np.random.normal(175, 0.1*150)

        case "Polymers":
            min_coupling = 1
            if bulk_dict is None:
                C11 = np.random.normal(5.5, 0.1*9)
                C12 = np.random.normal(5.1/2, 0.1*4.9)
                C44 = np.random.normal(3.1/2, 0.1*2.9)
        case _:
            raise ValueError(f"Wrong material_type {material_type}, supposed be \'Metals\' or \'Ceramics\' or \'Polymers\'")
    if bulk_dict is not None:
        bulk = bulk_dict["voigt"]
        shear = shear_dict["voigt"]
        mu_ = shear
        lambda_ = bulk-2./3.*shear
        C11 = lambda_ + 2*mu_
        C12 = lambda_
        C44 = mu_

    stiffness_matrix = np.zeros([6,6])
    stiffness_matrix[0,0] = C11
    stiffness_matrix[1,1] = C11
    stiffness_matrix[2,2] = C11
    stiffness_matrix[0,1] = C12
    stiffness_matrix[0,2] = C12
    stiffness_matrix[1,2] = C12
    stiffness_matrix[3,3] = C44
    stiffness_matrix[4,4] = C44
    stiffness_matrix[5,5] = C44
    if SGn < 3:
        # raise ValueError(f"Triclinic SGn = {SGn}")
        C55 = np.random.rand()*(C11-C12) + C12
        C66 = np.random.rand()*(C11-C12) + C12
        C33 = np.random.normal(loc=C11)
        C23 = np.random.normal(loc=C12)
        C13 = np.random.normal(loc=C12)
        # min_coupling = min([C11, C12, C33, C23, C13, C55, C66, C44])
        C14 = _low_symm_couplings(material_type, min_coupling)
        C15 = _low_symm_couplings(material_type, min_coupling)
        C24 = _low_symm_couplings(material_type, min_coupling)
        C25 = _low_symm_couplings(material_type, min_coupling)
        C34 = _low_symm_couplings(material_type, min_coupling)
        C35 = _low_symm_couplings(material_type, min_coupling)
        C45 = _low_symm_couplings(material_type, min_coupling)
        C46 = _low_symm_couplings(material_type, min_coupling)
        
        # C12
        stiffness_matrix[0,2] = C13
        stiffness_matrix[1,2] = C23
        stiffness_matrix[0,3] = C14
        stiffness_matrix[0,4] = C15
        stiffness_matrix[1,3] = C24
        stiffness_matrix[1,4] = C25
        stiffness_matrix[2,3] = C34
        stiffness_matrix[2,4] = C35
        stiffness_matrix[3,4] = C45
        stiffness_matrix[3,5] = C46
        # C44
        stiffness_matrix[4,4] = C55
        stiffness_matrix[5,5] = C66
        pass
    elif 3 <= SGn < 16:
        # "Monoclinic"
        C55 = np.random.rand()*(C11-C12) + C12
        C66 = np.random.rand()*(C11-C12) + C12
        C33 = np.random.normal(loc=C11)
        C23 = np.random.normal(loc=C12)
        C13 = np.random.normal(loc=C12)
        # min_coupling = min([C11, C12, C33, C23, C13, C55, C66, C44])
        C15 = _low_symm_couplings(material_type, min_coupling)
        C25 = _low_symm_couplings(material_type, min_coupling)
        C35 = _low_symm_couplings(material_type, min_coupling)
        C46 = _low_symm_couplings(material_type, min_coupling)
        stiffness_matrix[2,2] = C33
        # C12
        stiffness_matrix[0,2] = C13
        stiffness_matrix[1,2] = C23
        stiffness_matrix[0,4] = C15
        stiffness_matrix[1,4] = C25
        stiffness_matrix[2,4] = C35
        stiffness_matrix[3,5] = C46
        # C44
        stiffness_matrix[4,4] = C55
        stiffness_matrix[5,5] = C66
        pass
    elif 16 <= SGn < 75:
        # "Orthorhombic"
        C55 = np.random.rand()*(C11-C12) + C12
        C66 = np.random.rand()*(C11-C12) + C12
        C33 = np.random.normal(loc=C11)
        C23 = np.random.normal(loc=C12)
        C13 = np.random.normal(loc=C12)
        stiffness_matrix[2,2] = C33
        # C12
        stiffness_matrix[0,2] = C13
        stiffness_matrix[1,2] = C23
        # C44
        stiffness_matrix[4,4] = C55
        stiffness_matrix[5,5] = C66
        pass
    elif 89 <= SGn < 105 or 107 <= SGn < 111:
        # "Tetragonal"
        C66 = (C11-C12)/2
        C33 = np.random.normal(loc=C11)
        C13 = np.random.normal(loc=C12)
        # min_coupling = min([C11, C12, C33, C13, C66, C44])
        C16 = _low_symm_couplings(material_type, min_coupling)
        stiffness_matrix[2,2] = C33
        stiffness_matrix[0,2] = C13
        stiffness_matrix[1,2] = C13
        stiffness_matrix[5,5] = C66
        stiffness_matrix[0,5] = C16
    elif 75 <= SGn < 89 or 105 <= SGn < 107 or 111 <= SGn < 143:
        # "Tetragonal"
        C66 = (C11-C12)/2
        C33 = np.random.normal(loc=C11)
        C13 = np.random.normal(loc=C12)
        stiffness_matrix[2,2] = C33
        stiffness_matrix[0,2] = C13
        stiffness_matrix[1,2] = C13
        stiffness_matrix[5,5] = C66
        pass
    elif 143 <= SGn < 168:
        # "Trigonal"
        C66 = (C11-C12)/2
        C33 = np.random.normal(loc=C11)
        C13 = np.random.normal(loc=C12)
        # min_coupling = min([C11, C12, C33, C13, C66, C44])
        C14 = _low_symm_couplings(material_type, min_coupling)
        stiffness_matrix[2,2] = C33
        stiffness_matrix[0,2] = C13
        stiffness_matrix[1,2] = C13
        stiffness_matrix[5,5] = C66
        stiffness_matrix[0,3] = C14
        stiffness_matrix[1,3] = -C14
        stiffness_matrix[4,5] = C14
        pass
    elif 168 <= SGn < 195:
        # "Hexagonal"
        C66 = (C11-C12)/2
        C33 = np.random.normal(loc=C11)
        C13 = np.random.normal(loc=C12)
        stiffness_matrix[2,2] = C33
        stiffness_matrix[0,2] = C13
        stiffness_matrix[1,2] = C13
        stiffness_matrix[5,5] = C66
        pass
    elif 195 <= SGn < 231:
        # "Cubic"
        pass
    else:
        raise ValueError(f"Invalid space group number: {SGn} (must be 1-230)")
    
    for i in range(6):
        for j in range(i+1,6):
            stiffness_matrix[j,i] = stiffness_matrix[i,j]

    return stiffness_matrix