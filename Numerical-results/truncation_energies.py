import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpo_cheks import build_hamiltonian, tensor, truncate_middle, pair_states, idx
from dmrg import entanglement_entropy

xi = 1.0
ex = np.load("exact_states.npz", allow_pickle=True)
g_grid = ex["g_values"]
psi_grid = ex["gs_exact"]  
hf = np.load("hartree_fock_states.npz", allow_pickle=True)

# ---- 1. Is the HF state a product state? Rank of the coefficient matrix at each cut --------
ranks = set()
for psi in hf["psi_HF"]:
    T = tensor(psi)
    r = []
    for cut in (1, 2, 3):
        sv = np.linalg.svd(T.reshape(2**cut, 2**(4 - cut)), compute_uv=False)
        r.append(int(np.sum(sv > 1e-10)))
    ranks.add(tuple(r))
print("HF bond dimensions at the three cuts (all g):", ranks)
print("HF state in the six-state basis:", hf["psi_HF"][0], "  labels:", list(hf["labels"]))

# ---- 2. energies ----------------------------------------------------------------------------
E_ex, E_hf, E_tci = [], [], []
E_chi = {1: [], 2: []}
disc = {1: [], 2: []}
overlap1 = []
for g, psi in zip(g_grid, psi_grid):
    H6, _ = build_hamiltonian(g, xi)
    E_ex.append(np.linalg.eigvalsh(H6)[0])
    E_hf.append(2 * xi - g)                                   # Task 4: E_HF = 2 - g for xi = 1
    E_tci.append(np.linalg.eigvalsh(H6[:5, :5])[0])           # Task 3: reference + 4 single-pair excitations, |34> dropped
    for chi in (1, 2):
        v, d = truncate_middle(psi, chi)
        v6 = v[idx]
        E_chi[chi].append(v6 @ H6 @ v6)
        disc[chi].append(d)
    v1, _ = truncate_middle(psi, 1)
    overlap1.append(abs(v1[idx][0]))                           # |<1100|chi=1 state>|
E_ex, E_hf, E_tci = map(np.array, (E_ex, E_hf, E_tci))
E_chi = {k: np.array(v) for k, v in E_chi.items()}
disc = {k: np.array(v) for k, v in disc.items()}
print("chi=1 state equals |1100> for all g:", np.allclose(overlap1, 1.0))
print("chi=1 energy == E_HF for all g:", np.allclose(E_chi[1], E_hf))

np.savez("truncation_results.npz", g=g_grid, E_exact=E_ex, E_HF=E_hf, E_trunc_CI=E_tci,
         E_chi1=E_chi[1], E_chi2=E_chi[2], disc1=disc[1], disc2=disc[2])

print(f"\n{'g':>6} {'E_exact':>9} {'E_HF=chi1':>10} {'E_chi2':>9} {'E_truncCI':>10} {'disc chi1':>10} {'disc chi2':>10}")
for gt in (-1.0, -0.5, 0.5, 1.0):
    i = np.argmin(abs(g_grid - gt))
    print(f"{g_grid[i]:6.2f} {E_ex[i]:9.5f} {E_hf[i]:10.5f} {E_chi[2][i]:9.5f} {E_tci[i]:10.5f} {disc[1][i]:10.2e} {disc[2][i]:10.2e}")

# ---- 3. plots ---------------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
ax1.plot(g_grid, E_ex, "k-", lw=2, label="exact")
ax1.plot(g_grid, E_chi[2], "--", color="#c4501b", lw=1.8, label=r"MPS $\chi=2$")
ax1.plot(g_grid, E_tci, "-.", color="#2a7f62", lw=1.8, label="truncated CI (5x5)")
ax1.plot(g_grid, E_hf, ":", color="#1f4e9c", lw=2.4, label=r"HF $=\chi=1$")
ax1.set_ylabel(r"$\langle H\rangle$"); ax1.set_title("Energies, $\\xi=1$")

m = np.abs(g_grid) > 1e-9
for lab, E, c, ls in ((r"HF $=\chi=1$", E_hf, "#1f4e9c", ":"),
                      (r"$\chi=2$", E_chi[2], "#c4501b", "--"),
                      ("trunc. CI", E_tci, "#2a7f62", "-.")):
    ax2.semilogy(g_grid[m], np.maximum(E[m] - E_ex[m], 1e-16), color=c, ls=ls, lw=1.8, label=lab)
ax2.set_ylabel(r"$E-E_{\rm exact}$"); ax2.set_title("Error")

for chi, c in ((1, "#1f4e9c"), (2, "#c4501b")):
    ax3.semilogy(g_grid[m], disc[chi][m], color=c, lw=1.8, label=rf"$\chi={chi}$")
ax3.set_ylabel(r"$\sum_{k>\chi}\sigma_k^2$"); ax3.set_title("Discarded weight (middle cut)")
for ax in (ax1, ax2, ax3):
    ax.set_xlabel("g"); ax.legend(frameon=False)
    ax.grid(True, color="#d9d9d9", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("truncation_comparison.png", dpi=200)