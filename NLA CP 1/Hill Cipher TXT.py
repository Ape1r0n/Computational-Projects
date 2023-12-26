import numpy as np


printables = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
              'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F',
              'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!',
              '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
              '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', '\t', '\n', '\r', '\x0b', '\x0c']


def mapping(m):
    if m in printables:
        return printables.index(m) + 1
    else:
        return -1


def inverse_mapping(m):
    if 0 < int(m) < 101:
        return printables[m-1]
    else:
        return None


def to_int_matrix(A):
    if A[0, 0] < 1.:
        return np.round(100 * A).astype(np.int16)
    return np.round(A).astype(np.int16)


def decrypted_to_string(A):
    result = ''.join(map(lambda row: ''.join(map(lambda elem: inverse_mapping(elem), row)), A))
    return result


def sum_of_non_diagonal_entries(A):
    n = len(A)
    sums = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                sums[i] += abs(A[i, j])
    return sums


def jacobi_matrix_form(A, b, k):
    n = len(A)
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    U = np.zeros((n, n))
    x = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i < j:
                U[i, j] = A[i, j]
            elif i == j:
                D[i, i] = A[i, i]
            else:
                L[i, j] = A[i, j]

    D_inv = np.linalg.inv(D)
    B_J = (-D_inv) @ (L + U)
    f_J = D_inv @ b

    while k > 0:
        x = (B_J @ x) + f_J
        k = k - 1

    return x


def textfile_to_matrix(file):
    with open(file, 'r') as f:
        content = f.read()

    n = int((len(content) - 1) ** 0.5) + 1  # Getting dimension n for matrix D

    content = content[:n ** 2]  # Making sure there are enough elements to make n by n matrix
    content += ' ' * (n ** 2 - len(content))  # Filling remaining entries with ' ', so that text is readable

    raw_D = np.array(list(content)).reshape(n, n)  # Making n by n array
    D_int = np.vectorize(lambda x: mapping(x))(raw_D)  # np.vectorize(f, ...) applies function element-wise to np.array
    D_float = np.vectorize(lambda x: mapping(x) / 100)(raw_D)

    return D_int, D_float  # D contains two digit arithmetic representation of each character


def generate_K_int(n):
    while True:
        K = np.random.randint(1, 100, size=(n, n))
        sonde = sum_of_non_diagonal_entries(K)  # "sonde" is acronym for function it is calling
        for i in range(n):
            K[i, i] += sonde[i]
        if all(np.linalg.eigvals(K) != 0):
            return K


def generate_K_floats(n):
    while True:
        K = np.round(np.random.uniform(0.01, 0.99, size=(n, n)), 2) # Making K entries 2 digit floats
        sonde = sum_of_non_diagonal_entries(K)  # "sonde" is acronym for function it is calling
        for i in range(n):
            K[i, i] += sonde[i]
        if all(np.linalg.eigvals(K) != 0):
            return K


def decrypt_direct(K, E):
    n = len(K)
    I = np.identity(n)
    augmented_matrix = np.concatenate((K, I), axis=1)  # We were taught this method in LA course

    for i in range(n):
        augmented_matrix[i, :] /= augmented_matrix[i, i]
        for j in range(n):
            if i != j:
                augmented_matrix[j, :] -= augmented_matrix[j, i] * augmented_matrix[i, :]

    inverse_K = augmented_matrix[:, n:]

    return inverse_K @ E  # @ is cooler way to multiply matrices


def decrypt_Jacobi(K, E, k):  # K @ D_tilda = E(Btw in the problem statement in RHS D should be replaced by E) (A,b,k)
    D_tilda = np.empty((len(E), len(E)))
    for i in range(len(E)):
        D_tilda[:, i] = jacobi_matrix_form(K, E[:, i], k)
    return D_tilda


#           --- TESTS FOR TEXTFILES ---

D_int, D_float = textfile_to_matrix("smth.txt")
#print("D_int: ", D_int, sep='\n')
#print("D_float: ", D_float, sep='\n')

K_int = generate_K_int(D_int.shape[0])
K_float = generate_K_floats(D_float.shape[0])
#print("K_int: ", K_int, sep='\n')
#print("K_float: ", K_float, sep='\n')

E_int = K_int @ D_int
E_float = K_float @ D_float
#print("E_int: ", E_int, sep='\n')
#print("E_float: ", E_float, sep='\n')

D_tilda_int = decrypt_direct(K_int, E_int)
D_tilda_float = decrypt_direct(K_float, E_float)



D_tilda_1 = to_int_matrix(D_tilda_int)
D_tilda_2 = to_int_matrix(D_tilda_float)

print("D_tilda_1: ", D_tilda_1, sep='\n')
print("D_tilda_2: ", D_tilda_2, sep='\n')

string_1 = decrypted_to_string(D_tilda_1)
string_2 = decrypted_to_string(D_tilda_2)
print("String 1: ", string_1, sep='\n')
print("String 2: ", string_2, sep='\n')

iterative_int = decrypt_Jacobi(K_int, E_int, 1000)
iterative_float = decrypt_Jacobi(K_float, E_float, 1000)
print("Iterative int: ", iterative_int, sep='\n')
print("Iterative float: ", iterative_int, sep='\n')

D_tilda_3 = to_int_matrix(iterative_int)
D_tilda_4 = to_int_matrix(iterative_int)
print("Iterative Tilda 1: ", D_tilda_3, sep='\n')
print("Iterative Tilda 2: ", D_tilda_4, sep='\n')

print(type(D_tilda_3))
string_3 = decrypted_to_string(D_tilda_3)
string_4 = decrypted_to_string(D_tilda_4)
print("String 3: ", string_3, sep='\n')
print("String 4: ", string_4, sep='\n')


# FUNCTIONS I THOUGHT I WOULD NEED AT SOME POINT IN TIME
#
# def jacobi_vector_form(A, b, k):
#     n = len(b)
#     x = np.zeros(n)
#
#     for _ in range(k):
#         temp = np.zeros(n)
#         for i in range(n):
#             sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
#             temp[i] = (b[i] - sigma) / A[i, i]
#         x = temp
#
#     return x
#
# def converges_in_Gauss_Seidl(A):
#     n = len(A)
#     L = np.zeros((n, n))
#     D = np.zeros((n, n))
#     U = np.zeros((n, n))
#
#     for i in range(n):
#         for j in range(n):
#             if i < j:
#                 U[i, j] = A[i, j]
#             elif i == j:
#                 D[i, i] = A[i, i]
#             else:
#                 L[i, j] = A[i, j]
#
#     return np.linalg.norm((np.linalg.inv(D + L) @ U), ord=2) < 1