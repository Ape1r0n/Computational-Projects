from scipy import ndimage
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class CannyEdgeDetector:
    def __init__(self, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.img_smoothed = None
        self.gradient = None
        self.theta = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold


    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        return np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal


    def sobel_filters(self, image):
        K_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
        K_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

        I_x = ndimage.convolve(image, K_x)
        I_y = ndimage.convolve(image, K_y)

        magnitude = np.hypot(I_x, I_y)
        magnitude = magnitude / magnitude.max() * 255 if magnitude.max() > 0 else magnitude
        direction = np.arctan2(I_y, I_x)

        return (magnitude, direction)

    def non_max_suppression(self, image, degrees):
        height, width = image.shape
        output = np.zeros_like(image)
        direction = degrees * 180. / np.pi
        direction[direction < 0] += 180

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                q = 255
                r = 255

                if (0 <= direction[i, j] < 22.5) or (157.5 <= direction[i, j] <= 180):  # 0
                    q = image[i, j + 1]
                    r = image[i, j - 1]
                elif 22.5 <= direction[i, j] < 67.5:  # 45
                    q = image[i + 1, j - 1]
                    r = image[i - 1, j + 1]
                elif 67.5 <= direction[i, j] < 112.5:  # 90
                    q = image[i + 1, j]
                    r = image[i - 1, j]
                elif 112.5 <= direction[i, j] < 157.5:  # 135
                    q = image[i - 1, j - 1]
                    r = image[i + 1, j + 1]

                if (image[i, j] >= q) and (image[i, j] >= r):
                    output[i, j] = image[i, j]
                else:
                    output[i, j] = 0


        return output

    def double_thresholding(self, img):
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        height, width = img.shape
        res = np.zeros((height, width), dtype=np.float32)

        strong_i, strong_j = np.where(img >= highThreshold)
        weak_i, weak_j = np.where((img >= lowThreshold) & (img < highThreshold))

        res[strong_i, strong_j] = self.strong_pixel
        res[weak_i, weak_j] = self.weak_pixel

        return res

    def hysteresis(self, img):
        height, width = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        result = np.copy(img).astype(np.float32)

        dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if result[i, j] == weak:
                    connected_to_strong = False
                    for k in range(8):
                        if result[i + dx[k], j + dy[k]] == strong:
                            connected_to_strong = True
                            break

                    result[i, j] = strong if connected_to_strong else 0

        return result


    def edge_detect(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        self.img_smoothed = ndimage.convolve(image, self.gaussian_kernel(self.kernel_size, self.sigma))
        self.gradient, self.theta = self.sobel_filters(self.img_smoothed)
        self.nonMaxImg = self.non_max_suppression(self.gradient, self.theta)
        self.thresholdImg = self.double_thresholding(self.nonMaxImg)
        img_final = self.hysteresis(self.thresholdImg)
        return img_final

    def display_step(self, image):
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    def edge_detect_visual(self, image):
        img_final = self.edge_detect(image)
        # Visualize all steps
        self.display_step(self.img_smoothed)  # Smoothed image
        self.display_step(self.gradient)    # Gradient magnitude
        self.display_step(self.nonMaxImg)      # Non-maximum suppressed image
        self.display_step(self.thresholdImg)    # Thresholded image
        self.display_step(img_final)            # Final edge-detected image

        return img_final


if __name__ == "__main__":
    image_path = "Documentation/Son_Goku.png"
    image = np.array(Image.open(image_path))

    detector = CannyEdgeDetector(
        sigma=1.4,
        kernel_size=5,
        weak_pixel=75,
        strong_pixel=255,
        lowthreshold=0.05,
        highthreshold=0.15
    )

    edges = detector.edge_detect_visual(image)