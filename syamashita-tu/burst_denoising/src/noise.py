from PIL import Image
import numpy as np


def gaussian_noise(image, mean=0, std=1):
    """
    Add Gaussian noise to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: Noisy image.
    """
    np.random.seed(0)

    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are valid
    return noisy_image.astype(np.uint8)

img = Image.open("syamashita-tu/burst_denoising/data/inputs/u10_Ship_2K.png")

img_array = np.array(img)
img


noisy_img_array = np.zeros_like(img_array)

for i in range(15):
    noisy_img_array[i] = gaussian_noise(noisy_img_array, mean=0, std=0.1)
noisy_img = Image.fromarray(noisy_img_array)
