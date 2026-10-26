
import numpy as np


def left_canonical_mps(C, N, d=2, tol=1e-12):
    """
    Full left-to-right SVD.

    Returns:
        Cs      : all matrices C_i before each SVD
        Rs      : all remainders R_i = Sigma_i @ Vt_i
        Us      : all left-canonical factors U_i
        tensors : MPS tensors with shape (left bond, d, right bond)
    """
    C = np.asarray(C, dtype=float)

    if C.size != d**N:
        raise ValueError(f"Expected {d**N} entries, got {C.size}.")

    Cs = []
    Rs = []
    Us = []
    tensors = []

    # Initial coefficient tensor, reshaped into N physical indices
    current = C.reshape([d] * N)
    left_bond = 1

    for i in range(N - 1):
        # Save C_i as a matrix
        C_i = current.reshape(left_bond * d, -1)
        Cs.append(C_i.copy())

        # SVD
        U, s, Vt = np.linalg.svd(C_i, full_matrices=False)

        # Remove numerically zero singular values
        rank = np.sum(s > tol * s[0]) if len(s) and s[0] > 0 else 0

        U = U[:, :rank]
        s = s[:rank]
        Vt = Vt[:rank, :]

        # R_i = Sigma_i @ Vt_i
        R_i = np.diag(s) @ Vt

        Us.append(U.copy())
        Rs.append(R_i.copy())

        # Store left-canonical tensor A_i
        A_i = U.reshape(left_bond, d, rank)
        tensors.append(A_i.copy())

        # R_i becomes the input for the next SVD
        current = R_i
        left_bond = rank

        # Reshape R_i into the next C matrix on the next iteration
        current = current.reshape(left_bond * d, -1) if i < N - 2 else current

    # The final remainder is the last MPS tensor
    final_tensor = current.reshape(left_bond, d, 1)
    tensors.append(final_tensor.copy())

    return Cs, Rs, Us, tensors



import numpy as np
import sympy as sp


def print_mps(Cs, Rs, Us, tensors, tol=1e-6):

    def pretty(x):
        x = float(np.real_if_close(x))

        # Recognize common fractions and square roots
        candidates = [
            sp.Integer(0), sp.Integer(1), sp.Integer(-1),
            sp.Rational(1, 2), sp.Rational(-1, 2),
            sp.Rational(1, 3), sp.Rational(-1, 3),
            sp.Rational(1, 4), sp.Rational(-1, 4),
            sp.sqrt(2)/2, -sp.sqrt(2)/2,
            sp.sqrt(3)/2, -sp.sqrt(3)/2,
            sp.sqrt(6)/6, -sp.sqrt(6)/6,
            1/sp.sqrt(2), -1/sp.sqrt(2),
            1/sp.sqrt(3), -1/sp.sqrt(3),
            1/sp.sqrt(6), -1/sp.sqrt(6),
            sp.sqrt(2), -sp.sqrt(2),
            sp.sqrt(3), -sp.sqrt(3),
        ]

        for value in candidates:
            if abs(float(value) - x) < tol:
                return str(value)

        return f"{x:.4f}"

    def show(name, M):
        M = np.asarray(M)
        entries = [[pretty(x) for x in row] for row in M.reshape(
            M.shape[0], -1
        )]

        widths = [
            max(len(row[j]) for row in entries)
            for j in range(len(entries[0]))
        ]

        print(f"\n{name} (shape = {M.shape})")
        for row in entries:
            print("[ " + "   ".join(
                value.rjust(widths[j])
                for j, value in enumerate(row)
            ) + " ]")

    print("=" * 60)
    print("         LEFT-CANONICAL MPS")
    print("=" * 60)

    for i, (C, U, R) in enumerate(zip(Cs, Us, Rs), start=1):
        show(f"C_{i}", C)
        show(f"U_{i}", U)
        show(f"R_{i}", R)

    for i, A in enumerate(tensors, start=1):
        print(f"\nMPS tensor A_{i}, shape = {A.shape}")

        for s in range(A.shape[1]):
            show(f"A_{i}[:,:,{s}]", A[:, s, :])

    print("\n" + "=" * 60)



# Example: four qubits
x = 1 / np.sqrt(6)

C1 = [
    [0, 0, 0, x, 0, x, x, 0],
    [0, x, x, 0, x, 0, 0, 0],
]

Cs, Rs, Us, tensors = left_canonical_mps(C1, N=4, d=2)

print_mps(Cs, Rs, Us, tensors)
