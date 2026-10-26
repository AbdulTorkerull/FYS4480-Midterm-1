"""Task 8 (iii): MPO of exercise 5 as (D, D, d, d) arrays, 16x16 contraction,
and <Psi|H|Psi> of the chi-truncated MPS by contracting the MPO with the MPS."""
import numpy as np
from itertools import combinations
from dmrg import MPS, pairing_terms, build_mpo, mpo_bond_dimensions   # teacher's code

L, D, d, xi = 4, 4, 2, 1.0
pair_states = list(combinations(range(L), 2))      # |1100>, |1010>, ... (same order as the npz)

# ---- local operators, basis {|0> empty, |1> pair};  matrix[out, in] -------------
I2 = np.eye(2)
Pp = np.array([[0., 0.], [1., 0.]])                # P+ |0> = |1>
Pm = Pp.T                                          # P- |1> = |0>
n  = np.diag([0., 2.])                             # two particles on a paired level

def build_hamiltonian(g, xi, nlev=4, npair=2):
    eps = [xi * p for p in range(nlev)]
    states = list(combinations(range(nlev), npair))   # |1100>, |1010>, |1001>, |0110>, |0101>, |0011>
    H = np.zeros((len(states), len(states)))
    for i, s in enumerate(states):
        H[i, i] = sum(2 * eps[p] for p in s) - g * npair / 2     # diagonal
        for j, t in enumerate(states):
            if len(set(s) & set(t)) == npair - 1:                # differ by one moved pair
                H[i, j] = -g / 2                                 # off-diagonal
    return H, states

def mpo_site(p, g):
    """W[p] of shape (D, D, d, d); row = left bond alpha, column = right bond beta."""
    eps = xi * p                                   # p = 0..3  <->  eps_p = xi (p-1), p = 1..4
    W = np.zeros((D, D, d, d))
    W[0, 0] = I2
    W[0, 1] = Pp
    W[0, 2] = Pm
    W[0, 3] = eps * n - (g / 2) * Pp @ Pm          # term finished on this site alone: eps n and p=q term
    W[1, 1] = I2
    W[1, 3] = -(g / 2) * Pm
    W[2, 2] = I2
    W[2, 3] = -(g / 2) * Pp
    W[3, 3] = I2
    return W

def mpo(g):
    return [mpo_site(p, g) for p in range(L)]

def dense(W):
    """Contract W[1]W[2]W[3]W[4] (bonds as matrices, physical indices by Kronecker) to 16x16."""
    l = np.array([1., 0, 0, 0]); r = np.array([0., 0, 0, 1.])
    H = np.einsum("a,abst->bst", l, W[0])
    for w in W[1:]:
        m = H.shape[1]
        H = np.einsum("bst,bcuv->csutv", H, w).reshape(D, m * d, m * d)
    return np.einsum("b,bst->st", r, H)

def bit_index(s):                                   # |n1n2n3n4> -> row of the 16x16 matrix (site 1 most significant)
    return sum(2 ** (L - 1 - p) for p in s)
idx = [bit_index(s) for s in pair_states]

# ---- MPS from a state vector ---------------------------------------------------
def tensor(psi):
    T = np.zeros((2,) * L)
    for a, s in zip(psi, pair_states):
        T[tuple(1 if p in s else 0 for p in range(L))] = a
    return T

def truncate_middle(psi, chi):
    """Keep chi Schmidt states across the middle cut, renormalise. Returns 16-vector, discarded weight."""
    U, S, Vt = np.linalg.svd(tensor(psi).reshape(4, 4))
    disc = np.sum(S[chi:] ** 2)
    S_t = S.copy(); S_t[chi:] = 0
    v = (U @ np.diag(S_t) @ Vt).reshape(-1)
    return v / np.linalg.norm(v), disc

def to_mps(v, tol=1e-12):
    """Sequential SVD of a 16-vector -> MPS tensors (chi_l, d, chi_r), left-canonical."""
    A, M, chi = [], v.reshape(1, -1), 1
    for i in range(L - 1):
        M = M.reshape(chi * d, -1)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        r = int(np.sum(S > tol))
        A.append(U[:, :r].reshape(chi, d, r))
        M = S[:r, None] * Vt[:r]; chi = r
    A.append(M.reshape(chi, d, 1))
    return MPS(A)

def energy_mpo(mps, W):
    """<psi|H|psi> with the MPO contracted into the MPS (teacher's MPS.expectation).
    Boundary vectors are absorbed so that the end bonds have dimension 1."""
    l = np.array([1., 0, 0, 0]); r = np.array([0., 0, 0, 1.])
    Wb = list(W)
    Wb[0] = np.einsum("a,abst->bst", l, W[0])[None]           # (1, D, d, d)
    Wb[-1] = np.einsum("abst,b->ast", W[-1], r)[:, None]      # (D, 1, d, d)
    return mps.expectation(Wb)

if __name__ == "__main__":
    data = np.load("exact_states.npz", allow_pickle=True)
    g_grid = data["g_values"]
    psi_grid = data["gs_exact"]  

    print("=== (a) 16x16 MPO matrix against the 6x6 matrix of exercise 2 ===")
    for g in (-1.0, -0.5, 0.5, 1.0):
        H16 = dense(mpo(g))
        H6, states = build_hamiltonian(g, xi)
        assert states == pair_states
        block = H16[np.ix_(idx, idx)]
        outside = np.delete(np.arange(16), idx)
        leak = np.abs(H16[np.ix_(outside, idx)]).max()          
        print(f"g={g:5.2f}  max|H_MPO - H_6x6| = {np.abs(block - H6).max():.1e}   "
              f"leak out of 2-pair sector = {leak:.1e}   Hermitian: {np.abs(H16 - H16.T).max():.1e}")

    print("\n=== (b) energy of the truncated MPS: MPO contraction vs 6x6 matrix ===")
    print(f"{'g':>6} {'chi':>4} {'discarded':>10} {'E_MPO':>12} {'<psi|H6|psi>':>13} {'diff':>8} {'E_exact':>10}  bonds")
    for g in (-1.0, -0.5, 0.5, 1.0):
        i = np.argmin(abs(g_grid - g))
        psi = psi_grid[i]
        H6, _ = build_hamiltonian(g_grid[i], xi)
        E0 = np.linalg.eigvalsh(H6)[0]
        for chi in (1, 2, 4):
            v, disc = truncate_middle(psi, chi)
            mps = to_mps(v)
            E_mpo = energy_mpo(mps, mpo(g_grid[i]))
            v6 = np.array([v[j] for j in idx])
            E_6 = v6 @ H6 @ v6
            print(f"{g_grid[i]:6.2f} {chi:4d} {disc:10.2e} {E_mpo:12.8f} {E_6:13.8f} {abs(E_mpo - E_6):8.0e} {E0:10.6f}  {mps.bond_dimensions()}")