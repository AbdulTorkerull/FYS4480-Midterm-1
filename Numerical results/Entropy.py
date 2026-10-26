import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from dmrg import entanglement_entropy     

data = np.load("exact_states.npz", allow_pickle=True)
g_values = data["g_values"]
psi_all = data["gs_exact"]         # psi_all: (n_g, 6)
pair_states = list(combinations(range(4), 2))       # same order as the npz labels

def to_tensor(psi):
    T = np.zeros((2, 2, 2, 2))
    for amp, s in zip(psi, pair_states):
        T[tuple(1 if p in s else 0 for p in range(4))] = amp
    return T

def cut_data(psi, cut):
    """Singular values, Schmidt weights and entropy for the cut after site `cut`."""
    M = to_tensor(psi).reshape(2**cut, 2**(4 - cut))
    sv = np.linalg.svd(M, compute_uv=False)
    return sv, sv**2, entanglement_entropy(sv)

n_g = len(g_values)
S = np.zeros((n_g, 3))
weights = [np.zeros((n_g, 4)) for _ in range(3)]    # at most 4 nonzero values (2x8 -> 2, 4x4 -> 4)
for i, psi in enumerate(psi_all):
    for c in (1, 2, 3):
        sv, w, S[i, c - 1] = cut_data(psi, c)
        weights[c - 1][i, :len(w)] = w

np.savez("entropies_xi1.npz", g=g_values, S=S, w1=weights[0], w2=weights[1], w3=weights[2])

# xi = 0 value from exercise 7: weights (2/3, 1/6, 1/6)
S_xi0 = -(2/3*np.log(2/3) + 2*(1/6)*np.log(1/6))
g_positive = np.linspace(0, max(g_values), 200)

plt.figure(figsize=(6, 4))
plt.plot(g_values, S[:, 1], label=r"$\xi=1$, middle cut")
plt.plot(g_positive, np.full_like(g_positive, S_xi0), color="k", ls="--",label=rf"$\xi=0$, $g>0$: {S_xi0:.3f}")
plt.xlabel("g"); plt.ylabel(r"$S_2$"); plt.legend(); plt.tight_layout()
plt.savefig("entropy_middle_cut.png", dpi=200)

linestyles = ["-.", "--", ":"]
plt.figure(figsize=(6, 4))
for c in range(3):
    plt.plot(g_values, S[:, c], label=f"cut {c+1}", linestyle=linestyles[c])
plt.xlabel("g"); plt.ylabel(r"$S_\ell$"); plt.legend(); plt.tight_layout()
plt.savefig("entropy_all_cuts.png", dpi=200)

for gt in (-1.0, -0.5, 0.0, 0.5, 1.0):
    i = np.argmin(abs(g_values - gt))
    print(f"g={g_values[i]:5.2f}  S =", np.round(S[i], 4),
          " middle weights =", np.round(weights[1][i], 4))
print("xi=0 value:", round(S_xi0, 4))
print("max over g of S per cut:", np.round(S.max(axis=0), 4))
print("middle largest everywhere:", bool(np.all(S[:, 1] >= S.max(axis=1) - 1e-12)))