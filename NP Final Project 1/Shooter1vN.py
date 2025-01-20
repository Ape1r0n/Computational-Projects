import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from DBSCAN import DBSCAN
from PIL import Image
from Solvers import MainGuy



class Shooter1vN:
    def __init__(self, img_path, seed=95):
        self.img = np.array(Image.open(img_path))
        if self.img is None:
            raise ValueError(f"Could not load image from path: {img_path}")
        (self.shooter_position, self.shooter_radius), (self.centers, self.radii) = self.setup(self.img, seed)

        # For output
        self.shooter = MainGuy(self.img.shape[1], self.img.shape[0])
        self.setup_simulation()

    @staticmethod
    def get_edge_points(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Sobel kernels for x and y matrices
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = np.zeros_like(img, dtype=float)
        grad_y = np.zeros_like(img, dtype=float)

        grad_x[1:-1, 1:-1] = (
                sobel_x[0, 0] * img[:-2, :-2] + sobel_x[0, 1] * img[:-2, 1:-1] + sobel_x[0, 2] * img[:-2, 2:] +
                sobel_x[1, 0] * img[1:-1, :-2] + sobel_x[1, 1] * img[1:-1, 1:-1] + sobel_x[1, 2] * img[1:-1, 2:] +
                sobel_x[2, 0] * img[2:, :-2] + sobel_x[2, 1] * img[2:, 1:-1] + sobel_x[2, 2] * img[2:, 2:]
        )

        grad_y[1:-1, 1:-1] = (
                sobel_y[0, 0] * img[:-2, :-2] + sobel_y[0, 1] * img[:-2, 1:-1] + sobel_y[0, 2] * img[:-2, 2:] +
                sobel_y[1, 0] * img[1:-1, :-2] + sobel_y[1, 1] * img[1:-1, 1:-1] + sobel_y[1, 2] * img[1:-1, 2:] +
                sobel_y[2, 0] * img[2:, :-2] + sobel_y[2, 1] * img[2:, 1:-1] + sobel_y[2, 2] * img[2:, 2:]
        )

        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        strong_edges = grad_magnitude > 100.0  # 100 is simply a threshold for strong edges
        edge_points = np.column_stack(np.where(strong_edges))

        return edge_points


    def setup(self, img, seed=95, shooter_size=5):
        random.seed(seed)
        np.random.seed(seed)

        edge_points = self.get_edge_points(img)
        dbscan = DBSCAN(eps=5, min_pts=5)
        dbscan.dbscan(edge_points)
        centers = dbscan.cluster_centers_
        radii = dbscan.cluster_radii_

        def is_valid_position(candidate, centers, r, min_distance=5):
            for center, radius in zip(centers, r):
                distance = np.linalg.norm(candidate - center)
                if distance < radius + shooter_size + 2 or distance < min_distance:
                    return False
            return True

        img_shape = np.array(img).shape[:2]
        while True:
            candidate = np.random.uniform(low=[0, 0], high=img_shape)
            if is_valid_position(candidate, centers, radii):
                break

        return (candidate, shooter_size), (centers, radii)


    def current_state(self):
        img_copy = cv2.cvtColor(np.array(self.img.copy()), cv2.COLOR_BGR2RGB)

        for center, radius in zip(self.centers, self.radii):
            cv2.circle(img_copy,(int(center[1]), int(center[0])),  # Swap x and y for OpenCV's coordinate system
                int(radius), (255, 0, 0), 2
            )

        cv2.circle(img_copy, (int(self.shooter_position[1]), int(self.shooter_position[0])),
                   int(self.shooter_radius), (0, 255, 0), -1  # Filled circle
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(img_copy)
        plt.axis('off')
        plt.title("Current State")
        plt.show()


    def setup_simulation(self):
        self.shooter.set_shooter((self.shooter_position[1], self.shooter_position[0]), self.shooter_radius)
        for center, radius in zip(self.centers, self.radii):
            self.shooter.add_target(center[1], center[0], radius)


    def run_simulation(self):
        self.shooter.simulate()



if __name__ == "__main__":
    fail_image = "report/Image to Fail.png"
    paths = ["report/Image 1.png", "report/Image 2.jpeg", "report/Image 3.jpg", "report/Image 4.png",]
    for path in paths:
        print(f"Running simulation for {path}")
        shooter = Shooter1vN(path)
        shooter.run_simulation()
        print(f"Simulation for {path} complete")
