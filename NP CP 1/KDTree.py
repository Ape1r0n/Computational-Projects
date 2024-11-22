import heapq


class KDTree:
    def __init__(self, points, dim, norm=None):
        self.dim = dim
        self._root = self._make(points, 0)
        self._distance = norm if norm is not None else lambda a, b: sum((a[i] - b[i]) ** 2 for i in range(dim))

    def _make(self, points, i=0):
        if len(points) > 1:
            points = points[points[:, i].argsort()]  # NumPy sort along dimension i
            i = (i + 1) % self.dim
            m = len(points) // 2
            return [self._make(points[:m], i), self._make(points[m + 1:], i), points[m]]
        if len(points) == 1:
            return [None, None, points[0]]
        return None

    def _add_point(self, node, point, i=0):
        if node is not None:
            dx = node[2][i] - point[i]
            for j, c in ((0, dx >= 0), (1, dx < 0)):
                if c and node[j] is None:
                    node[j] = [None, None, point]
                elif c:
                    self._add_point(node[j], point, (i + 1) % self.dim)

    def add_point(self, point):
        if self._root is None:
            self._root = [None, None, point]
        else:
            self._add_point(self._root, point)

    def get_nearest(self, point, return_dist_sq=True):
        l = self._get_knn(self._root, point, 1, return_dist_sq, [])
        return l[0] if len(l) else None

    def get_knn(self, point, k, return_dist_sq=True):
        return self._get_knn(self._root, point, k, return_dist_sq, [])

    def _get_knn(self, node, point, k, res_sq_dist, heap, i=0, t_brk=1):
        if node is not None:
            dist_sq = self._distance(point, node[2])
            dx = node[2][i] - point[i]
            if len(heap) < k:
                heapq.heappush(heap, (-dist_sq, t_brk, node[2]))
            elif dist_sq < -heap[0][0]:
                heapq.heappushpop(heap, (-dist_sq, t_brk, node[2]))
            i = (i + 1) % self.dim

            for b in [dx < 0, dx >= 0][:1 + (dx * dx < -heap[0][0])]:
                self._get_knn(node[int(b)], point, k, res_sq_dist, heap, i, (t_brk << 1) | int(b))

        if t_brk == 1:
            return [(-h[0], h[2]) if res_sq_dist else h[2] for h in sorted(heap)][::-1]

    def _walk(self, node):
        if node is not None:
            for j in 0, 1:
                for x in self._walk(node[j]):
                    yield x
            yield node[2]

    def __iter__(self):
        return self._walk(self._root)

    def __repr__(self):
        return f"KDTree({list(self)})"



def plot_kdtree(tree, dim, depth=0, bounds=None):
    if tree is None:
        return

    node = tree[2]
    if node is None:
        return

    axis = depth % dim

    if bounds is None:
        bounds = [[-10, 10], [-10, 10]]  # Example bounds for 2D case

    plt.scatter(*node[:2], c='red')

    if axis == 0:
        plt.plot([node[0], node[0]], bounds[1], 'k--')
        plot_kdtree(tree[0], dim, depth + 1, [[bounds[0][0], node[0]], bounds[1]])
        plot_kdtree(tree[1], dim, depth + 1, [[node[0], bounds[0][1]], bounds[1]])
    else:
        plt.plot(bounds[0], [node[1], node[1]], 'k--')
        plot_kdtree(tree[0], dim, depth + 1, [bounds[0], [bounds[1][0], node[1]]])
        plot_kdtree(tree[1], dim, depth + 1, [bounds[0], [node[1], bounds[1][1]]])



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    points = np.array([
        [2, 3],
        [5, 4],
        [9, 6],
        [4, 7],
        [8, 1],
        [7, 2]
    ])

    # Create the KDTree
    kdtree = KDTree(points, dim=2)

    # Plot the KDTree with dimension splits
    plt.figure(figsize=(8, 8))
    plot_kdtree(kdtree._root, kdtree.dim)
    plt.xlim(-1, 10)
    plt.ylim(-1, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('K-d Tree Dimension Splits')
    plt.show()
