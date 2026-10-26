import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# One-body Hamiltonian: four spatial orbitals
h0 = np.diag([0.0, 1.0, 2.0, 3.0])

n_orbitals = 4
n_occupied = 2

# Same state order as the exact ground-state vectors: 1100, 1010, 1001, 0110, 0101, 0011
pair_states = list(combinations(range(n_orbitals), n_occupied))
labels = np.array([''.join('1' if p in s else '0' for p in range(n_orbitals)) for s in pair_states])


def solve_hf(g, tol=1e-10, max_iter=100):
    # Initial guess: occupy orbitals 1 and 2
    C = np.eye(n_orbitals)
    P = C[:, :n_occupied] @ C[:, :n_occupied].conj().T

    for iteration in range(1, max_iter + 1):
        # Construct the Hartree-Fock matrix from the current density matrix
        hHF = h0 - (g / 2.0) * P

        # Diagonalize: eigenvalues and eigenvectors
        eps, C = np.linalg.eigh(hHF)

        # New density matrix from the occupied orbitals
        C_occ = C[:, :n_occupied]
        P_new = C_occ @ C_occ.conj().T

        # Check convergence of the density matrix
        converged = np.linalg.norm(P_new - P) < tol
        P = P_new
        if converged:
            break

    # Total energy, not the sum of orbital energies
    E_HF = 2 * np.trace(h0 @ P).real \
           - (g / 2.0) * np.sum(np.abs(P)**2)

    return eps, C, P, E_HF, iteration


def hf_state_vector(C):
    """Hartree-Fock Slater determinant written in the six-state basis.

    Both spin species occupy the same spatial orbitals, so the amplitude of the state
    with levels (p, q) doubly occupied is det(C_occ[[p, q], :])**2.
    """
    C_occ = C[:, :n_occupied]
    return np.array([np.linalg.det(C_occ[list(s), :])**2 for s in pair_states])


g_values = np.linspace(-1, 1, 100)

E_HF_values, iterations_used = [], []
eps_all, C_all, P_all, psi_HF_all = [], [], [], []
for g in g_values:
    eps, C, P, E_HF, iterations = solve_hf(g)
    E_HF_values.append(E_HF)
    iterations_used.append(iterations)
    eps_all.append(eps)
    C_all.append(C)
    P_all.append(P)
    psi_HF_all.append(hf_state_vector(C))

print(f"Maximum number of iterations over all g: {max(iterations_used)}")

# Save everything needed for the comparison with the exact ground state
np.savez("hartree_fock_states.npz",
         g=g_values,
         E_HF=np.array(E_HF_values),
         eps=np.array(eps_all),          # orbital energies, shape (n_g, 4)
         C=np.array(C_all),              # orbital coefficients, shape (n_g, 4, 4); occupied = first 2 columns
         P=np.array(P_all),              # density matrices, shape (n_g, 4, 4)
         psi_HF=np.array(psi_HF_all),    # Slater determinant in the six-state basis, shape (n_g, 6)
         labels=labels)

# Analytical solution
E_analytic = 2 - g_values

# Plot
plt.plot(g_values, E_HF_values, label=r"Numerical $E_{\mathrm{HF}}$")
plt.plot(g_values, E_analytic, "--", label=r"Analytical $E=2-g$")

plt.xlabel("Coupling strength $g$")
plt.ylabel("Energy")
plt.title("Hartree-Fock Energy vs. Coupling Strength")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()