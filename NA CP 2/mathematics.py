import numpy as np



# Problem 1

# Since floats have bigger range, they might leave bounds of 0 <= number <= 4294967295, where number is an integer. This is required since data of wavfile must be of type integer
def normalize(number) -> int:
    return int(max(min(number, 100000), -100000))


# Bézier function
def bezier(p0, p1, p2, p3, t):
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


# Segment will consist of 882 samples. Using bezier for every 3 samples and return the curve
def bezier_for_segment(segment):
    curve = []
    for i in range(0, 879, 3):  # 879 because we need 4 points to form a Bézier curve and 882 will be out of range
        p0 = segment[i]
        p1 = segment[i + 1]
        p2 = segment[i + 2]
        p3 = segment[i + 3]
        for t in np.arange(0, 1, 1 / 6):
            curve.append(normalize(bezier(p0, p1, p2, p3, t)))
    return curve


# Catmull-Rom function
def catmull_rom(p0, p1, p2, p3, t):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


# Same logic as bezier_for_segment
def catmull_rom_for_segment(segment):
    curve = []
    for i in range(0, 879, 3):  # 879 because we need 4 points to form a Catmull-Rom spline and 882 will be out of range
        p0 = segment[i]
        p1 = segment[i + 1]
        p2 = segment[i + 2]
        p3 = segment[i + 3]
        for t in np.arange(0, 1, 1 / 6):
            curve.append(normalize(catmull_rom(p0, p1, p2, p3, t)))
    return curve


# I will use Chebyshev nodes to interpolate using Lagrange Interpolation
def chebyshev_numbers(n=44):
    a = 0
    b = 881
    k = np.arange(n + 1, 1, -1)
    return np.round(0.5 * (a + b) + 0.5 * (b - a) * np.cos((k - 0.5) * np.pi / n)).astype(int)


x_nodes = chebyshev_numbers(n=44)


# Lagrange Interpolation code that uses Chebyshev nodes as x values and segment elements as y values
def lagrange_interpolation(segment, n=44):
    y = [segment[i] for i in x_nodes]
    L_n = []

    for x in range(0, len(segment)):
        L_x = 0
        for j in range(n):
            L_j = 1
            for i in range(n):
                if x_nodes[i] == x_nodes[j]:
                    continue
                L_j *= (x - x_nodes[i]) / (x_nodes[j] - x_nodes[i])
            L_x += y[j] * L_j
        L_n.append(L_x)

    return L_n


# Problem 2

def gaussian_radial_base(r, shape_parameter):
    return np.exp(-(shape_parameter * r) ** 2)


def inverse_quadratic(r, shape_parameter):
    return 1 / (1 + (shape_parameter * r) ** 2)


# A function that takes function phi and list of points and returns the matrix of phi values for each pair of points
def RBF_interpolation_matrix(phi, points, shape_parameter=None):
    n = len(points)
    phi_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            phi_matrix[i, j] = phi(np.linalg.norm(np.array(points[i]) - np.array(points[j]), ord=2), shape_parameter)
    return phi_matrix
