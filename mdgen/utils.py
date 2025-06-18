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


# Obtain the stiffness matrix based on the bulk and shear moduli
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

# Obtain transition matrix of the lattice matrix based on coordinates change
def interp_then_rescale(A, B, x):
    Mx = (1 - x) * A + x * B
    det_Mx = torch.det(Mx)
    V = torch.det(A)
    scale = (V / det_Mx).abs().pow(1/3)
    return Mx * scale


def polar_decompose_batch(F):
    """
    Batched polar decomposition: F = R @ U,
    where R∈SO(3) (proper rotation, det=+1) and U is SPD.
    Input:  F of shape (N, 3, 3), assumed invertible.
    Returns:
      R (N,3,3), U (N,3,3)
    """
    U_svd, S, Vh = torch.linalg.svd(F)  # U_svd:(N,3,3), S:(N,3), Vh:(N,3,3)
    R = U_svd @ Vh
    # Ensure proper rotation (det=+1)
    detR = torch.det(R)                 # (N,)
    mask_reflect = detR < 0             # where reflection occurred
    if mask_reflect.any():
        # Flip last column of Vh where det(R)<0
        Vh[mask_reflect, -1, :] *= -1
        R = U_svd @ Vh

    # Build U = Vhᵀ @ diag(S) @ Vh  → SPD
    S_mat = torch.zeros_like(F)
    S_mat[:, 0, 0] = S[:, 0]
    S_mat[:, 1, 1] = S[:, 1]
    S_mat[:, 2, 2] = S[:, 2]
    U_mat = Vh.transpose(-2, -1) @ S_mat @ Vh
    return R, U_mat

def matrix_log_symmetric_batch(U):
    """
    Batched log for SPD matrices U (N,3,3) via eigh.
    Returns logU, real-symmetric.
    """
    eigvals, eigvecs = torch.linalg.eigh(U)    # eigvals:(N,3), eigvecs:(N,3,3)
    # clamp eigenvalues away from zero
    log_eig = torch.log(eigvals.clamp(min=1e-8))  # (N,3)
    logU = eigvecs @ torch.diag_embed(log_eig) @ eigvecs.transpose(-2, -1)
    return logU

def matrix_exp_symmetric_batch(logU):
    """
    Batched exp for real‐symmetric matrices logU (N,3,3) via eigh.
    Returns U = exp(logU), SPD.
    """
    eigvals, eigvecs = torch.linalg.eigh(logU)
    exp_eig = torch.exp(eigvals)  # (N,3)
    U = eigvecs @ torch.diag_embed(exp_eig) @ eigvecs.transpose(-2, -1)
    return U

def matrix_logm_SO3_batch(R):
    """
    Batched matrix log for R∈SO(3), returns skew‐symmetric (N,3,3).
    Uses: log(R) = (θ / (2 sin θ)) (R - Rᵀ),   θ = arccos((tr(R)-1)/2).
    """
    traces = R.diagonal(dim1=-2, dim2=-1).sum(-1)     # (N,)
    cos_t = ((traces - 1) / 2).clamp(-1.0, 1.0)
    theta = torch.acos(cos_t)                         # (N,)
    sin_t = torch.sin(theta)
    skew = 0.5 * (R - R.transpose(-2, -1))             # (N,3,3)

    factor = torch.zeros_like(theta)                   # (N,)
    mask = theta > 1e-5
    factor[mask] = theta[mask] / (2 * sin_t[mask])
    factor = factor.view(-1, 1, 1)                     # (N,1,1)

    logR = factor * skew                               # (N,3,3), skew‐symmetric
    return logR

def matrix_exp_SO3_batch(skew):
    """
    Batched Rodrigues’ formula for exp(skew) where skew∈so(3) (N,3,3).
    Returns R∈SO(3), det=+1.
    """
    a = skew[:, 2, 1]
    b = skew[:, 0, 2]
    c = skew[:, 1, 0]
    theta = torch.sqrt(a*a + b*b + c*c).clamp(min=1e-8)  # (N,)
    K = skew / theta.view(-1, 1, 1)                      # (N,3,3)
    I = torch.eye(3, device=skew.device).unsqueeze(0).expand_as(skew)
    sin_t = torch.sin(theta).view(-1, 1, 1)
    cos_t = torch.cos(theta).view(-1, 1, 1)
    R = I + sin_t * K + (1 - cos_t) * (K @ K)
    return R

def interpolate_polar_batch(cell0, cell1, t):
    """
    Fully batched interpolation between cell0, cell1 (N,3,3), preserving det>0.
    t can be shape (N,) or (N,1,1).  Returns (N,3,3) with det>0.
    """
    # Decompose both endpoints
    R0, U0 = polar_decompose_batch(cell0)   # (N,3,3), (N,3,3)
    R1, U1 = polar_decompose_batch(cell1)   # (N,3,3), (N,3,3)

    # Rotation interpolation on SO(3)
    R_rel = R0.transpose(-2, -1) @ R1       # (N,3,3)
    logR_rel = matrix_logm_SO3_batch(R_rel) # (N,3,3)
    if t.ndim == 1:
        t = t.view(-1, 1, 1)
    R_interp = R0 @ matrix_exp_SO3_batch(t * logR_rel)  # (N,3,3)

    # Stretch interpolation in log‐space
    logU0 = matrix_log_symmetric_batch(U0)  # (N,3,3)
    logU1 = matrix_log_symmetric_batch(U1)  # (N,3,3)
    logU_interp = (1 - t) * logU0 + t * logU1  # (N,3,3)
    U_interp = matrix_exp_symmetric_batch(logU_interp) # (N,3,3)

    # Recombine: h(t) = R(t) U(t)
    cell_t = R_interp @ U_interp  # (N,3,3), each det>0
    return cell_t


'''
def _TransM_by_dx(A,B,rcond=None):
    # M = torch.linalg.lstsq(A, B, rcond=None)[0].T
    ## Normalize
    M0 = torch.linalg.lstsq(A, B, rcond=rcond).solution
    det_M0 = torch.det(M0)
    M = M0 / det_M0.pow(1/3)  # M has det = 1
    F_fit = torch.transpose(M, 2,3)

    return F_fit
'''

def _TransM_by_dx(A: torch.Tensor,
                        B: torch.Tensor,
                        lam: float = 1e-6,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    Solve (AᵀA + λI) M = AᵀB in batch form, so M stays full‐rank.
    Then normalize det(M)=1 in a NaN‐safe way.
    
    Args:
      A    Tensor of shape (..., 3, 3)
      B    Tensor of shape (..., 3, 3)
      lam  small regularization constant (λ > 0)
    
    Returns:
      F_fit  Tensor of shape (..., 3, 3), which is Mᵀ after det‐normalization
    """
    # 1) Build AᵀA + λI, and AᵀB in batch form.
    #    If A is (..., 3, 3), then:
    AT = A.transpose(-2, -1)                 # shape (..., 3, 3)
    ATA = AT @ A                             # shape (..., 3, 3)
    
    # Add λI:
    # We want to do ATA + λ * I for each batch. The easiest is to create an identity of size 3×3,
    # then broadcast-add λ·I to every “slice” of ATA.
    I3 = torch.eye(3, device=A.device).expand(ATA.shape)  # shape (..., 3, 3)
    ATA_reg = ATA + lam * I3
    
    ATB = AT @ B                             # shape (..., 3, 3)
    
    # 2) Solve (ATA + λI) M = ATB for M in batched form:
    #    Use torch.linalg.solve, which expects square (..., 3,3) @ (..., 3,3) = (..., 3,3).
    M0 = torch.linalg.solve(ATA_reg, ATB) 
    assert not torch.isnan(M0).any(), f"{ATA_reg} {ATB} {A} {B}"
    det_M0 = torch.det(M0)
    assert not torch.isnan(M0).any(), f"{M0} {ATA_reg} {ATB} {A} {B}"
    # M = M0 / det_M0.pow(1/3)  # M has det = 1
    # 4) Build a “safe” real cube‐root of det(M₀): sign = ±1 (never 0)
    sign = torch.sign(det_M0)
    sign = torch.where(sign == 0, torch.tensor(1.0, device=sign.device), sign)

    #    - abs_det = |det(M₀)| clamped away from 0
    abs_det = det_M0.abs().clamp(min=eps)  # shape (...,)

    #    - real_cuberoot = sign * (abs_det)^(1/3)
    real_cuberoot = sign * abs_det.pow(1.0 / 3.0)  # shape (...,)

    # 5) Divide out that cube‐root so det(M) ≈ 1:
    scale = real_cuberoot.unsqueeze(-1).unsqueeze(-1)  # (..., 1, 1)
    M = M0 / scale   # now det(M) ≈ 1, and is real (no NaNs from negative det)
    assert not torch.isnan(M).any(), f"{M0} {ATA_reg} {ATB} {A} {B}"
    F_fit = torch.transpose(M, 2,3)

    return F_fit


def logm(A, eps: float = 1e-8):
    """
    Compute the principal matrix logarithm of a (batched) square matrix A via eigendecomposition,
    avoiding NaNs by clamping each eigenvalue’s magnitude away from zero.

    Args:
        A: Tensor of shape (..., N, N), real or complex.
        eps: Small floor for |eig| to avoid log(0).

    Returns:
        logA: Tensor of shape (..., N, N), complex if A had any non‐positive/complex eigenvalues.
    """
    # 1) Compute eigenvalues/eigenvectors (possibly complex)
    eigvals, eigvecs = torch.linalg.eig(A)
    # eigvals: (..., N), complex dtype
    # eigvecs: (..., N, N), complex dtype

    # 2) Clamp the magnitude |eigvals| to be >= eps, but keep the original phase
    abs_eig = torch.abs(eigvals)                       # (..., N) real ≥ 0
    phase = torch.atan2(eigvals.imag, eigvals.real)     # (..., N) real (angle in radians)
    abs_clamped = abs_eig.clamp(min=eps)                # (..., N) real
    eigvals_safe = abs_clamped * torch.exp(1j * phase)  # (..., N) complex

    # 3) Take the complex logarithm of each (clamped) eigenvalue
    log_eig = torch.log(eigvals_safe)                   # (..., N) complex

    # 4) Reconstruct:  logA = V @ diag(log_eig) @ V⁻¹
    V = eigvecs                                         # (..., N, N) complex
    V_inv = torch.linalg.inv(V)                         # (..., N, N) complex
    logD = torch.diag_embed(log_eig)                     # (..., N, N) complex
    logA = V @ (logD @ V_inv)                            # (..., N, N) complex

    return logA

def expm(A, eps: float = 1e-8):
    """
    Compute the matrix exponential of a (batched) square matrix A via eigendecomposition.
    If A is real but has complex eigenvalues, this returns a complex‐dtype tensor.

    Args:
        A: Tensor of shape (..., N, N), real or complex.
        eps: (Unused here, but kept for API symmetry with logm_torch.)

    Returns:
        expA: Tensor of shape (..., N, N), same dtype as A (possibly complex).
    """
    # 1) Compute eigenvalues/eigenvectors (possibly complex)
    eigvals, eigvecs = torch.linalg.eig(A)
    # eigvals: (..., N), complex dtype
    # eigvecs: (..., N, N), complex dtype

    # 2) Exponentiate each eigenvalue
    exp_eig = torch.exp(eigvals)                        # (..., N) complex

    # 3) Reconstruct:  exp(A) = V @ diag(exp_eig) @ V⁻¹
    V = eigvecs                                         # (..., N, N) complex
    V_inv = torch.linalg.inv(V)                         # (..., N, N) complex
    expD = torch.diag_embed(exp_eig)                     # (..., N, N) complex
    expA = V @ (expD @ V_inv)                            # (..., N, N) complex

    return expA

def interp_log_then_rescale(cell0, cell1, t, eps=1e-8):
    # 1) logs
    log0 = logm(cell0)   # (N,3,3), complex
    log1 = logm(cell1)   # (N,3,3), complex

    # 2) interp in log-space
    if t.ndim == 1:
        t = t.view(-1,1,1)
    log_t = (1 - t)*log0 + t*log1           # (N,3,3)

    # 3) exponentiate
    cand = expm(log_t)                       # (N,3,3), complex
    cand = cand.real                         # assume result should be real

    # 4) desired volume
    V0 = torch.det(cell0)                    # (N,)
    det_cand = torch.det(cand)               # (N,)

    # 5) real cube-root scale
    sign = torch.sign(det_cand)
    sign = torch.where(sign == 0,
                       torch.tensor(1.0, device=sign.device),
                       sign)
    abs_det = det_cand.abs().clamp(min=eps)
    real_cuberoot = sign * abs_det.pow(1.0/3.0)  # (N,)

    # 6) rescale so det(cell_t)=V0
    scale = (V0/ det_cand).abs().pow(1.0/3.0)  # (N,)
    cell_t = cand * scale.view(-1,1,1)        # (N,3,3)

    return cell_t.real


# from torch_linear_assignment import batch_linear_assignment


def latt_plan(x0, x1, cell_0, cell_1, xt, t):
    TransM_1 = _TransM_by_dx(x1, xt)
    cell_t_1 = TransM_1@cell_1
    vol_t_1 = (cell_t_1[:,:,0,:]*torch.cross(cell_t_1[:,:,1,:], cell_t_1[:,:,2,:])).sum(-1)
    TransM_0 = _TransM_by_dx(x0, xt)
    cell_t_0 = TransM_0@cell_0
    vol_t_0 = (cell_t_0[:,:,0,:]*torch.cross(cell_t_0[:,:,1,:], cell_t_0[:,:,2,:])).sum(-1)
    assert not torch.isnan(vol_t_0)
    assert not torch.isnan(vol_t_1)
    assert torch.allclose(vol_t_0, vol_t_1, rtol=5e-2), f"{vol_t_0} != {vol_t_1} \n from cell_t_0, cell_t_1 = {cell_t_0}, {cell_t_1} \n from cell_0, cell_1 = {cell_0}, {cell_1} \n Check:: x0.max(), x0.min()={x0.max(), x0.min()} x1.max(), x1.min()={x1.max(), x1.min()}  xt.max(), xt.min()={xt.max(), xt.min()} "
    # cell_t = expm((1-t[:,None,None,None])*logm(cell_t_0) + t[:,None,None,None]*logm(cell_t_1)).real
    B,T,N,_ = x0.shape
    cell_t = interpolate_polar_batch(cell_t_0.reshape(B*T,3,3), cell_t_1.reshape(B*T,3,3), t.unsqueeze(1).expand(-1,T).reshape(-1)).reshape(B,T,3,3)
    vol_t = (cell_t[:,:,0,:]*torch.cross(cell_t[:,:,1,:], cell_t[:,:,2,:])).sum(-1)
    assert not torch.isnan(vol_t)
    assert torch.allclose(vol_t, vol_t_1, rtol=5e-2), f"{vol_t} != {vol_t_1} \n from cell_t, cell_t_0, cell_t_1 = {cell_t}, {cell_t_0}, {cell_t_1} \n from cell_0, cell_1 = {cell_0}, {cell_1}"
    return cell_t