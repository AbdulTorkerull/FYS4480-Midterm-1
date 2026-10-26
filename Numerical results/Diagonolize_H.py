import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Hamiltonian matrices
# --------------------------------------------------

def H_exact(g, xi=1.0):
    """Full 6x6 Hamiltonian in the basis
    |12>, |13>, |14>, |23>, |24>, |34>.
    """
    return np.array([
        [2*xi-g,   -g/2,     -g/2,     -g/2,     -g/2,      0],
        [-g/2,     4*xi-g,   -g/2,     -g/2,      0,       -g/2],
        [-g/2,     -g/2,     6*xi-g,    0,       -g/2,     -g/2],
        [-g/2,     -g/2,      0,       6*xi-g,   -g/2,     -g/2],
        [-g/2,      0,       -g/2,     -g/2,     8*xi-g,   -g/2],
        [0,        -g/2,     -g/2,     -g/2,     -g/2,    10*xi-g]
    ], dtype=float)


def H_approx(g, xi=1.0):
    """Truncated 5x5 Hamiltonian, excluding |34>."""
    return np.array([
        [2*xi-g,   -g/2,     -g/2,     -g/2,     -g/2],
        [-g/2,     4*xi-g,   -g/2,     -g/2,      0],
        [-g/2,     -g/2,     6*xi-g,    0,       -g/2],
        [-g/2,     -g/2,      0,       6*xi-g,   -g/2],
        [-g/2,      0,       -g/2,     -g/2,     8*xi-g]
    ], dtype=float)


# --------------------------------------------------
# 2. Diagonalization
# --------------------------------------------------

def diagonalize(H):
    """Eigenvalues in ascending order; eigenvectors are columns."""
    return np.linalg.eigh(H)


# --------------------------------------------------
# 3. Calculate energies and ground-state vectors
# --------------------------------------------------

def calculate_results(g_values, xi=1.0):
    E_exact = []
    E_approx = []

    all_exact = []
    all_approx = []

    gs_exact = []
    gs_approx = []

    for g in g_values:
        energies, vectors = diagonalize(H_exact(g, xi))
        E_exact.append(energies[0])
        all_exact.append(energies)
        gs_exact.append(vectors[:, 0])

        energies_a, vectors_a = diagonalize(H_approx(g, xi))
        E_approx.append(energies_a[0])
        all_approx.append(energies_a)
        gs_approx.append(vectors_a[:, 0])

    return {
        "g_values": np.array(g_values),
        "E_exact": np.array(E_exact),
        "E_approx": np.array(E_approx),
        "all_exact": np.array(all_exact),
        "all_approx": np.array(all_approx),
        "gs_exact": np.array(gs_exact),
        "gs_approx": np.array(gs_approx),
    }


# --------------------------------------------------
# 4. Plot ground-state energy versus g
# --------------------------------------------------

def plot_ground_energy(results, xi=1.0):
    g = results["g_values"]

    plt.figure(figsize=(8, 5))
    plt.plot(g, results["E_exact"], label="Exact (6x6)", linewidth=2)
    plt.plot(g, results["E_approx"], "--",
             label="Truncated (5x5)", linewidth=2)

    plt.xlabel(r"Pairing strength $g$")
    plt.ylabel(r"Ground-state energy $E_0$")
    plt.title(rf"Ground-state energy versus $g$ ($\xi={xi:g}$)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ground_state_energy.png", dpi=200)
    plt.show()


# --------------------------------------------------
# 5. Run calculations
# --------------------------------------------------

if __name__ == "__main__":
    xi = 1.0
    g_values = np.linspace(-1, 1, 201)

    results = calculate_results(g_values, xi)

    # Print matrices and results at a chosen g
    g_test = 1.0
    print(f"\nExact Hamiltonian at g={g_test}, xi={xi}:")
    print(H_exact(g_test, xi))

    E, V = diagonalize(H_exact(g_test, xi))
    print("\nExact eigenvalues:")
    print(E)
    print("\nExact ground-state energy:")
    print(E[0])
    print("\nExact ground-state eigenvector:")
    print(V[:, 0])

    Ea, Va = diagonalize(H_approx(g_test, xi))
    print("\nTruncated eigenvalues:")
    print(Ea)
    print("\nTruncated ground-state eigenvector:")
    print(Va[:, 0])

    # Plot results
    plot_ground_energy(results, xi)

    # Save numerical data for Part III
    np.savez(
        "exact_states.npz",
        **results,
        xi=xi
    )

    print("\nSaved results to exact_states.npz")
