import numpy as np
from scipy.io import wavfile


N = 10
printables = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
              'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F',
              'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!',
              '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
              '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', '\t', '\n', '\r', '\x0b', '\x0c']


def mapping(m):
    n = np.zeros(4, dtype=np.int16)
    for i in range(4):
        n[i] = m & 15
        m >>= 4
    return n


def inverse_mapping(m):
    n = 0
    for i in range(4):
        n <<= 4
        n += m[3-i]
    return n


def inverse_list(l):
    result = []
    for i in range(0, len(l), 4):
        m = np.array([l[i], l[i + 1], l[i + 2], l[i + 3]])
        result.append(inverse_mapping(m))

    return np.array(result).astype(np.int16)


def to_int_matrix(A):
    return np.round(A).astype(np.int16)


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


def map_list(l):  # Mapping multi-dimensional matrices into a single one
    flattened_array = np.array(l).flatten()
    result = np.array(list(map(mapping, flattened_array)))
    return result.flatten()


# def map_list(l):  # Old mapping, works for mono audio
#     return np.array(list(map(mapping, l))).flatten()


def array_to_matrix(inp):  # This time, n by n array would have been too much, I needed to come up with other algorithm
    content = map_list(inp)
    n = int((len(content) - 1) ** 0.5) + 1
    n = (n + N - 1) // N * N
    content = np.concatenate((content, np.zeros((n ** 2 - len(content)))), axis=None)
    raw_D = np.array(list(content)).reshape(n, n).astype(np.int16)
    print(raw_D)
    D_int = np.vectorize(lambda x: x)(raw_D)
    D_float = np.vectorize(lambda x: x / 100)(raw_D)

    res_int = np.zeros((n // N, n // N, N, N))
    res_float = np.zeros((n // N, n // N, N, N))

    for i in range(n // N):
        for j in range(n // N):
            res_int[i][j] = D_int[i * N:(i + 1) * N, j * N:(j + 1) * N]
            res_float[i][j] = D_float[i * N:(i + 1) * N, j * N:(j + 1) * N]

    return res_int, res_float


def generate_K_int(n):
    while True:
        K = np.random.randint(1, 100, size=(n, n))
        sonde = sum_of_non_diagonal_entries(K)
        for i in range(n):
            K[i, i] += sonde[i]
        if all(np.linalg.eigvals(K) != 0):
            return K


def generate_K_floats(n):
    while True:
        K = np.round(np.random.uniform(0.01, 0.99, size=(n, n)), 2)
        sonde = sum_of_non_diagonal_entries(K)
        for i in range(n):
            K[i, i] += sonde[i]
        if all(np.linalg.eigvals(K) != 0):
            return K


def inverse(K):
    n = len(K)
    I = np.identity(n)
    augmented_matrix = np.concatenate((K, I), axis=1)  # We were taught this method in LA course

    for i in range(n):
        augmented_matrix[i, :] /= augmented_matrix[i, i]
        for j in range(n):
            if i != j:
                augmented_matrix[j, :] -= augmented_matrix[j, i] * augmented_matrix[i, :]

    inverse_K = augmented_matrix[:, n:]

    return inverse_K


def decrypt_Jacobi(K, E, k):  # K @ D_tilda = E(Btw in the problem statement in RHS D should be replaced by E) (A,b,k)
    D_tilda = np.empty((len(E), len(E)))
    for i in range(len(E)):
        D_tilda[:, i] = jacobi_matrix_form(K, E[:, i], k)
    return D_tilda


#           --- TESTS FOR AUDIOFILES ---

bitrate, data = wavfile.read("./a1s.wav")

D_int, D_float = array_to_matrix(data)
print("D_int: ", D_int, sep='\n')
print("D_float: ", D_float, sep='\n')

K_int = generate_K_int(N)
K_float = generate_K_floats(N)
print("K_int: ", K_int, sep='\n')
print("K_float: ", K_float, sep='\n')

E_int = np.zeros(D_int.shape)
E_float = np.zeros(D_float.shape)
D_tilda_int = np.zeros(D_int.shape)
D_tilda_float = np.zeros(D_float.shape)

K_int_inverse = inverse(K_int)
K_float_inverse = inverse(K_float)


for i in range(D_int.shape[0]):
    for j in range(D_int.shape[1]):
        E_int[i, j] = K_int @ D_int[i, j]
        E_float[i, j] = K_float @ D_float[i, j]
        D_tilda_int[i, j] = K_int_inverse @ E_int[i, j]
        D_tilda_float[i, j] = K_float_inverse @ E_float[i, j]


print("E_int: ", E_int, sep='\n')
print("E_float: ", E_float, sep='\n')


print("D_tilda_int: ", D_tilda_int, sep='\n')
print("D_tilda_float: ", D_tilda_float, sep='\n')

D_tilda_1 = np.zeros(D_int.shape).astype(np.int16)
D_tilda_2 = np.zeros(D_int.shape).astype(np.int16)

for i in range(D_tilda_int.shape[0]):
    for j in range(D_tilda_int.shape[1]):
        D_tilda_1[i, j] = to_int_matrix(D_tilda_int[i, j])
        D_tilda_2[i, j] = np.round(D_tilda_float[i, j], 2) * 100

print("D_tilda_1: ", D_tilda_1, sep='\n')
# print("D_tilda_2: ", D_tilda_2, sep='\n')


D_tilda_3 = np.zeros(E_int.shape)
D_tilda_4 = np.zeros(E_float.shape)

for i in range(E_int.shape[0]):
    for j in range(E_float.shape[1]):
        D_tilda_3[i, j] = to_int_matrix(decrypt_Jacobi(K_int, E_int[i, j], 100))
        D_tilda_4[i, j] = np.round((decrypt_Jacobi(K_float, E_float[i, j], 100)), 2) * 100



data_1 = np.zeros(D_tilda_1.shape[0] * D_tilda_1.shape[1] * D_tilda_1.shape[2] * D_tilda_1.shape[3]).astype(np.int16)
data_2 = np.zeros(D_tilda_2.shape[0] * D_tilda_2.shape[1] * D_tilda_2.shape[2] * D_tilda_2.shape[3]).astype(np.int16)
data_3 = np.zeros(D_tilda_3.shape[0] * D_tilda_3.shape[1] * D_tilda_3.shape[2] * D_tilda_3.shape[3]).astype(np.int16)
data_4 = np.zeros(D_tilda_4.shape[0] * D_tilda_4.shape[1] * D_tilda_4.shape[2] * D_tilda_4.shape[3]).astype(np.int16)
ptr = 0
for i in range(D_tilda_1.shape[0] * D_tilda_1.shape[2]):
    for j in range(D_tilda_1.shape[1] * D_tilda_1.shape[3]):
        block_i = i // D_tilda_1.shape[2]
        block_j = j // D_tilda_1.shape[3]
        char_i = i % D_tilda_1.shape[2]
        char_j = j % D_tilda_1.shape[3]
        data_1[ptr] = (D_tilda_1[block_i, block_j, char_i, char_j].astype(np.int16))
        data_2[ptr] = (D_tilda_2[block_i, block_j, char_i, char_j].astype(np.int16))
        data_3[ptr] = (D_tilda_3[block_i, block_j, char_i, char_j].astype(np.int16))
        data_4[ptr] = (D_tilda_4[block_i, block_j, char_i, char_j].astype(np.int16))
        ptr += 1


data_1 = inverse_list(data_1)
data_2 = inverse_list(data_2)
data_3 = inverse_list(data_3)
data_4 = inverse_list(data_4)

result = data_2.reshape(-1, len(data.shape))

wavfile.write(data=result, filename="./out.wav", rate=bitrate)
print("Finished!")
print("Norm of difference of result and original data:", np.linalg.norm((result[0:len(data)].T if len(data.shape) == 1 else result[0:len(data)]) - data, 1), sep=' ')