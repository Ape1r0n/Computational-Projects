import numpy as np
from KDTree import KDTree


# Vector norms
def norm1(a: np.array, b: np.array) -> float:
    return np.sum(np.abs(a - b))


def norm2(a: np.array, b: np.array) -> float:
    return np.sqrt(np.sum((a - b) ** 2))


def normInf(a: np.array, b: np.array) -> float:
    return np.max(np.abs(a - b))


def norm_p(a: np.array, b: np.array, p: int) -> float:
    return np.sum(np.abs(a - b) ** p) ** (1 / p)


# Matrix norms
def norm_1(A: np.array, B: np.array) -> float:
    return np.max(np.sum(np.abs(A - B), axis=0))


def norm_2(A: np.array, B: np.array) -> float:
    return np.max(np.linalg.eigvals((A - B) @ (A - B).T))


def norm_inf(A: np.array, B: np.array) -> float:
    return np.max(np.sum(np.abs(A - B), axis=1))


def norm_frobenius(A: np.array, B: np.array) -> float:
    return np.sqrt(np.sum((A - B) ** 2))


# DBSCAN with K-d tree
class DBSCAN:
    def __init__(self, eps, min_pts, norm=norm2):
        self.eps = eps
        self.min_pts = min_pts
        self.norm = norm
        self.labels_ = None
        self.cluster_centers_ = None

    def dbscan(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)  # -1 represents unvisited points
        tree = KDTree(X, dim=X.shape[1], norm=self.norm)
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1:
                continue

            neighbors = tree.get_knn(X[i], n_samples, return_dist_sq=True)
            neighbors = [(dist, point) for dist, point in neighbors if dist <= self.eps ** 2]

            if len(neighbors) < self.min_pts:  # Mark as noise
                self.labels_[i] = -2  # -2 represents noise points
                continue

            self.labels_[i] = cluster_id
            # Seed Set gets resized, while neighbors list does not
            seed_set = [(dist, point) for dist, point in neighbors]
            while seed_set:
                _, curr_point = seed_set.pop(0)
                curr_idx = np.where(np.all(X == curr_point, axis=1))[0][0]

                if self.labels_[curr_idx] == -1:  # Unvisited point
                    self.labels_[curr_idx] = cluster_id
                    curr_neighbors = tree.get_knn(curr_point, n_samples, return_dist_sq=True)
                    curr_neighbors = [(dist, point) for dist, point in \
                                      curr_neighbors if dist <= self.eps ** 2]
                    if len(curr_neighbors) >= self.min_pts:
                        seed_set.extend(curr_neighbors)

                elif self.labels_[curr_idx] == -2:  # Noise point
                    self.labels_[curr_idx] = cluster_id

            cluster_id += 1

        self.cluster_centers_ = []
        for cluster_id in set(self.labels_):
            if cluster_id >= 0:  # Ignore noise points
                cluster_points = X[self.labels_ == cluster_id]
                center = np.mean(cluster_points, axis=0)
                self.cluster_centers_.append(center)

        return self


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(42)
    cluster_1 = np.random.randn(50, 2) * 5 + [5, 5]
    cluster_2 = np.random.randn(50, 2) * 5 + [20, 20]
    cluster_3 = np.random.randn(50, 2) * 5 + [35, 5]
    outliers = np.random.uniform(low=0, high=40, size=(10, 2))
    X = np.vstack((cluster_1, cluster_2, cluster_3, outliers))

    dbscan = DBSCAN(eps=2, min_pts=5, norm=norm2)
    dbscan.dbscan(X)

    plt.figure(figsize=(10, 10))
    for cluster_id in set(dbscan.labels_):
        if cluster_id == -2:  # Noise points (outliers)
            color, marker = 'k', 'x'
        else:
            color = plt.colormaps['tab10'](cluster_id % 10)
            marker = 'o'

        cluster_points = X[dbscan.labels_ == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[color], marker=marker, label=f"Cluster {cluster_id}")

    if dbscan.cluster_centers_:
        centers = np.array(dbscan.cluster_centers_)
        plt.scatter(centers[:, 0], centers[:, 1], c='purple', s=100, marker='*', label='Centers')

    plt.title("DBSCAN Clustering with K-d Tree")
    plt.xlabel("X"), plt.ylabel("Y")
    plt.legend(), plt.grid(True), plt.show()
