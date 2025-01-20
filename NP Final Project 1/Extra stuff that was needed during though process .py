import matplotlib.pyplot as plt
import numpy as np
from DBSCAN import DBSCAN
from PIL import Image
from time import time
from Shooter1vN import Shooter1vN


def visualize_results(img, edge_points, centers, radii, labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Edge-detected image with DBSCAN clusters
    axes[0].imshow(img, cmap='gray')
    cmap = plt.get_cmap('tab10') if len(np.unique(labels)) <= 10 else plt.get_cmap('hsv')
    colors = cmap(np.linspace(0, 1, len(np.unique(labels))))
    for label, color in zip(np.unique(labels), colors):
        if label != -1:
            axes[0].scatter(edge_points[labels == label, 1], edge_points[labels == label, 0], s=1, color=color)
    axes[0].set_title("Edge Detection with DBSCAN")

    # Right: Circles with centers and radii
    axes[1].imshow(img, cmap='gray')
    for center, radius in zip(centers, radii):
        circle = plt.Circle((center[1], center[0]), radius, color='blue', fill=False)
        axes[1].add_artist(circle)
    axes[1].set_title("Detected Circles")

    plt.show()


def display_edge_points(img_path):
    img = np.array(Image.open(img_path).convert('L'))
    t1 = time()
    edge_points = Shooter1vN.get_edge_points(img)
    t2 = time()
    print(f"Edge detection took {t2 - t1:.2f} seconds")
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.imshow(img, cmap='gray')
    # ax1.set_title('Original Image')
    # ax2.imshow(img, cmap='gray')
    # ax2.scatter(edge_points[:, 1], edge_points[:, 0], s=1, c='r')
    # ax2.set_title('Edge Points')
    plt.imshow(img, cmap='gray')
    plt.scatter(edge_points[:, 1], edge_points[:, 0], s=1, c='r')
    plt.show()


def test1(img_path):
    img = np.array(Image.open(img_path).convert('L'))

    # Step 1: Get edge points
    t1 = time()
    edge_points = Shooter1vN.get_edge_points(img)
    t2 = time()
    print(f"Edge detection took {t2 - t1:.2f} seconds")

    # Step 2: Cluster edge points and calculate centers and radii
    t3 = time()
    dbscan = DBSCAN(eps=5, min_pts=25)
    dbscan.dbscan(edge_points)
    centers = dbscan.cluster_centers_
    radii = dbscan.cluster_radii_
    labels = dbscan.labels_
    t4 = time()
    print(f"DBSCAN clustering took {t4 - t3:.2f} seconds")

    # Step 3: Visualize results
    t5 = time()
    visualize_results(img, edge_points, centers, radii, labels)
    t6 = time()
    print(f"Visualization took {t6 - t5:.2f} seconds\n")