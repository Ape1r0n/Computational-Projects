import numpy as np



def pseudo_inverse_direct(A):
    return np.linalg.inv(A.T @ A) @ A.T


def pseudo_inverse_CGS(A):
    Q, R = QR_factorization_CGS(A)
    return np.linalg.inv(R) @ Q.T


def pseudo_inverse_MGS(A):
    Q, R = Gramm_Schmidt_Modified(A)
    return np.linalg.inv(R) @ Q.T


def sum_in_Gramm_Schmidt(x_i, y, i):
    sum = np.zeros((len(x_i)))
    for j in range(i):
        sum += np.dot(y[:, j], x_i) * y[:, j]
    return sum


def Gramm_Schmidt_Classical(x):
    m, n = x.shape[0], x.shape[1]
    y, y_tilde = np.zeros((m, n)), np.zeros((m, n))
    for i in range(n):
        y_tilde[:, i] = x[:, i] - sum_in_Gramm_Schmidt(x[:, i], y, i)
        if np.dot(y_tilde[:, i], y_tilde[:, i]) == 0:
            return y
        y[:, i] = y_tilde[:, i] / np.linalg.norm(y_tilde[:, i], ord=2)

    return y, y_tilde


def Gramm_Schmidt_Modified(x):
    Q = np.copy(x).astype(np.float64)
    n, m = Q.shape[0], Q.shape[1]
    R = np.zeros((m, m))

    for j in range(m):
        R[j, j] = np.linalg.norm(Q[:, j], ord=2)
        Q[:, j] = Q[:, j] / R[j, j]
        for i in range(j + 1, m):
            R[j, i] = np.dot(Q[:, i], Q[:, j])
            Q[:, i] = Q[:, i] - Q[:, j] * R[j, i]

    return Q, R


def QR_factorization_CGS(A):
    Q, Q_tilde = Gramm_Schmidt_Classical(A)
    m = A.shape[1]
    R = np.zeros((m, m))
    for j in range(m):
        for i in range(j):
            R[i][j] = np.dot(A[:, j], Q[:, i])
        R[j][j] = np.linalg.norm(Q_tilde[:, j], ord=2)
    return Q, R


def generate_A(arr, n):
    A = np.zeros((len(arr), n))
    for i in range(len(arr)):
        for j in range(n):
            A[i][j] = pow(arr[i] / 100.0, j)
    return A

