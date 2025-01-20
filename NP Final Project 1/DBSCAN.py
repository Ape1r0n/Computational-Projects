import numpy as np
from collections import defaultdict


def norm2(a: np.array, b: np.array) -> float:
    return np.sqrt(np.sum((a - b) ** 2))


class DSU:
    def __init__(self, size):
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=int)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


class Grid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def get_cell_coords(self, point):
        return tuple((point // self.cell_size).astype(int))

    def insert(self, idx, point):
        cell_coords = self.get_cell_coords(point)
        self.grid[cell_coords].append(idx)

    def get_neighbors(self, point):
        cell_coords = self.get_cell_coords(point)
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_coords = (cell_coords[0] + dx, cell_coords[1] + dy)
                neighbors.extend(self.grid[neighbor_coords])
        return neighbors


class DBSCAN:
    def __init__(self, eps, min_pts, norm=norm2):
        self.eps = eps
        self.min_pts = min_pts
        self.norm = norm
        self.labels_ = None
        self.cluster_centers_ = None
        self.cluster_radii_ = None

    def dbscan(self, X):
        n_samples = X.shape[0]
        dsu = DSU(n_samples)
        grid = Grid(cell_size=np.ceil(self.eps))

        for idx, point in enumerate(X):
            grid.insert(idx, point)

        for idx, point in enumerate(X):
            neighbors = grid.get_neighbors(point)
            for neighbor_idx in neighbors:
                if neighbor_idx != idx and self.norm(X[idx], X[neighbor_idx]) <= self.eps:
                    dsu.union(idx, neighbor_idx)

        root_to_cluster = {}
        cluster_id = 0
        self.labels_ = np.full(n_samples, -2, dtype=int)  # -2 represents noise points

        for i in range(n_samples):
            root = dsu.find(i)
            if root not in root_to_cluster:
                root_to_cluster[root] = cluster_id
                cluster_id += 1
            self.labels_[i] = root_to_cluster[root]

        centers = []
        radii = []
        for c_id in set(self.labels_):
            if c_id >= 0:
                cluster_points = X[self.labels_ == c_id]
                if len(cluster_points) >= self.min_pts:
                    center = np.mean(cluster_points, axis=0)
                    radius = np.max([self.norm(point, center) for point in cluster_points])
                    centers.append(center)
                    radii.append(radius)

        # self.cluster_centers_ = np.array(centers)
        # self.cluster_radii_ = np.array(radii)

        filtered_centers = []
        filtered_radii = []
        for i, (center, radius) in enumerate(zip(centers, radii)):
            keep = True
            for j, (other_center, other_radius) in enumerate(zip(centers, radii)):
                if i != j and self.norm(center, other_center) < radius + other_radius:
                    if radius <= other_radius:
                        keep = False
                        break
            if keep:
                filtered_centers.append(center)
                filtered_radii.append(radius)

        self.cluster_centers_ = np.array(filtered_centers)
        self.cluster_radii_ = np.array(filtered_radii)


        return self