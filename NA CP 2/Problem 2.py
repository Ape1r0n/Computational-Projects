import os

from PIL import Image
from mathematics import *


def randomly_remove_pixels(img, p):
    img_copy = img.copy()
    total_pixels = img_copy.shape[0] * img_copy.shape[1]
    num_pixels_to_remove = int(total_pixels * p)
    rows, cols = np.indices(img_copy.shape[:2])
    pixel_indices = np.column_stack((rows.ravel(), cols.ravel()))

    selected_indices = np.random.choice(pixel_indices.shape[0], num_pixels_to_remove, replace=False)
    for idx in selected_indices:
        img_copy[pixel_indices[idx, 0], pixel_indices[idx, 1]] = 0
    removed_indices = [(pixel_indices[idx][0], pixel_indices[idx][1]) for idx in selected_indices]

    return img_copy, removed_indices


# 3 different radial basis functions
def thin_plane_spline(r, shape_parameter=None):
    if r == 0:
        return 0
    else:
        return r ** 2 * np.log(r)


def restore_image(img, removed_indices, phi, shape_parameter=None):
    height, width = img.shape[:2]
    all_coordinates = [(i, j) for i in range(height) for j in range(width)]
    leftout_coordinates =[c for c in all_coordinates if c not in removed_indices]
    phi_matrix = RBF_interpolation_matrix(phi, leftout_coordinates, shape_parameter)
    rhs = np.array([img[i, j] for i, j in leftout_coordinates])
    weights = np.linalg.solve(phi_matrix, rhs)
    restored_img = np.zeros_like(img)

    for i, j in all_coordinates:
        rbf_sum = 0
        for k, (x, y) in enumerate(leftout_coordinates):
            r = np.linalg.norm([x - i, y - j], ord=2)
            rbf_sum += weights[k] * phi(r, shape_parameter)
        restored_img[i, j] = rbf_sum

    return restored_img


if __name__ == "__main__":
    folder_path = "materialForReport"
    file_name = "Gojo_32.png"
    file_path = os.path.join(folder_path, file_name)
    img = np.array(Image.open(file_path))
    percentages = [0.1, 0.25, 0.5, 0.75]
    removed_indices = []

    for p in percentages:
        img_array, r_is = randomly_remove_pixels(img, p)
        removed_indices.append(r_is)
        lessened_image = Image.fromarray(img_array)
        output_file_path = os.path.join(folder_path, f"{file_name[:-4]}_{int(p * 100)}percent_removed.png")
        lessened_image.save(output_file_path)

    print("4 modified images, each of them lacking 10, 25, 50 and 75 percent of pixels have been generated.")
    
    # Restoring the images  and saving them
    for p, removed_indices_p in zip(percentages, removed_indices):
        # Restore the image using the thin-plate spline radial basis function
        print(f"Restoring the image with {int(p * 100)} percent removed pixels using the thin plane spline radial basis function.")
        restored_img = restore_image(img, removed_indices_p, thin_plane_spline)
        output_file_path = os.path.join(folder_path, f"{file_name[:-4]}_{int(p * 100)}percent_restored_thin_plane_spline.png")
        Image.fromarray(restored_img.astype(np.uint8)).save(output_file_path)

        # Restore the image using the Gaussian radial basis function
        print(f"Restoring the image with {int(p * 100)} percent removed pixels using the Gaussian radial basis function.")
        restored_img = restore_image(img, removed_indices_p, gaussian_radial_base, shape_parameter=0.5)
        output_file_path = os.path.join(folder_path, f"{file_name[:-4]}_{int(p * 100)}percent_restored_gaussian.png")
        Image.fromarray(restored_img.astype(np.uint8)).save(output_file_path)

        # Restore the image using the Inverse Quadratic radial basis function
        print(f"Restoring the image with {int(p * 100)} percent removed pixels using the Inverse Quadratic radial basis function.")
        restored_img = restore_image(img, removed_indices_p, inverse_quadratic, shape_parameter=0.1)
        output_file_path = os.path.join(folder_path, f"{file_name[:-4]}_{int(p * 100)}percent_restored_inverse_quadratic.png")
        Image.fromarray(restored_img.astype(np.uint8)).save(output_file_path)
    
    print("Restored images have been saved.")
