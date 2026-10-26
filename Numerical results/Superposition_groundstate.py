import numpy as np

# Set g = 1 for numerical verification
g = -1

# Hamiltonian matrix in the two-pair basis
B = np.array([
    [2, 1, 1, 1, 1, 0],
    [1, 2, 1, 1, 0, 1],
    [1, 1, 2, 0, 1, 1],
    [1, 1, 0, 2, 1, 1],
    [1, 0, 1, 1, 2, 1],
    [0, 1, 1, 1, 1, 2]
], dtype=float)

H = -(g / 2) * B

# Proposed normalized ground state
psi = np.ones(6) / np.sqrt(6)

# Check normalization
print("Normalization <psi|psi> =", np.vdot(psi, psi))

# Check eigenvalue equation H|psi> = E|psi>
Hpsi = H @ psi
E_psi = np.vdot(psi, Hpsi)

print("Energy of proposed state =", E_psi)
print("Eigenvalue equation satisfied:",
      np.allclose(Hpsi, E_psi * psi))

# Diagonalize H to find all eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(H)

print("All eigenvalues =", eigenvalues)
print("Ground-state energy =", eigenvalues[0])
print("Expected ground-state energy =", -3 * g)

# Verify the proposed energy is the lowest eigenvalue
print("Proposed state is a ground state:",
      np.isclose(E_psi, eigenvalues[0]))