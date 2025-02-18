from typing import List
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def load_image_as_array(image_path: str) -> np.ndarray:
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
    except Exception as e:
        error_msg = f""" ↓\n
{'*' * 50 + "\n"}
Error loading image: {e}
{"\n"+'*' * 50}
""".strip()
        raise Exception(error_msg)
    return image_array


def downsample_image_by_half(image: np.ndarray, iterations: int = 1) -> np.ndarray:

    final_size = image.shape[0] // (2 ** iterations)
    if final_size < 2:
        raise ValueError(f"Image size {final_size} is too small for step {iterations}")

    for _ in range(iterations):
        keep_indices = np.array([i for i in range(image.shape[0]) if i % 2 == 0])
        image = image[keep_indices][:, keep_indices]
    return image


def upsample_image(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    for _ in range(iterations):
        image = np.repeat(image, 2, axis=0)
        image = np.repeat(image, 2, axis=1)
    return image


def plot_images(
        original_image,
        downsampled_images,
        figsize=None,
        save_path=None,
) -> None:

    n_images = 1 + len(downsampled_images)  # original + downsampled
    if figsize is None:
        figsize = (2.3 * n_images, 2.3)

    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]

    # Plot original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original ({original_image.shape[0]}x{original_image.shape[1]})')
    axes[0].axis('off')

    # Plot each image
    for idx, img in enumerate(downsampled_images, start=1):
        axes[idx].imshow(img, cmap='gray')
        try:
            h, w = img.shape[:2]
            axes[idx].set_title(f'({h}x{w})')
        except Exception:
            axes[idx].set_title("Image")
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_images_by_size(
        original_image,
        downsampled_images,
        save_path=None
) -> None:
    images = [original_image] + downsampled_images
    n_images = len(images)

    width_ratios = [img.shape[1] for img in images]

    scale = 0.05
    total_width = sum(width_ratios) * scale
    # Use the maximum height among the images to set the figure height.
    fig_height = max(img.shape[0] for img in images) * scale

    fig = plt.figure(figsize=(total_width, fig_height))
    gs = gridspec.GridSpec(1, n_images, width_ratios=width_ratios)

    # Plot each image in its own subplot.
    for idx, img in enumerate(images):
        ax = fig.add_subplot(gs[idx])
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def calculate_rmse(original_image: np.ndarray, upsampled_images: List[np.ndarray]) -> List[np.ndarray]:
    rmse_values = []
    for upsampled_image in upsampled_images:
        squared_diff = (original_image - upsampled_image) ** 2
        mean_squared_diff = np.mean(squared_diff)
        rmse = np.sqrt(mean_squared_diff)
        rmse_values.append(rmse)
    return rmse_values


def main():
    image_path = r"C:\Users\Adib\Desktop\Image Processing\1\man.bmp"

    ### r"C:\Users\Adib\Desktop\Image Processing\1\man.bmp"
    ### r"C:\Users\Adib\Desktop\Image Processing\1\IMG_5203.JPEG"
    original_image = load_image_as_array(image_path)
    print(50 * "*")
    print("Image Array:\n",original_image)
    print(50 * "*")
    print("Original Image Shape:\n",original_image.shape)
    print(50 * "*")

    downsampled_images = [downsample_image_by_half(original_image, i) for i in range(1, 6)]
    print("Down-sampled Image(1 Iteration):\n",downsampled_images[0])

    plot_images(
        original_image=original_image,
        downsampled_images=downsampled_images,
        save_path="downsampled_image.png",
    )

    plot_images_by_size(
        original_image=original_image,
        downsampled_images=downsampled_images,
        save_path="downsampled_image_different_sizes.png",
    )

    print(50 * "*")
    upsampled_images = [upsample_image(downsampled_images[i-1], i) for i in range(1, 6)]
    print("Up-sampled Image(1 Iteration):\n", upsampled_images[0])

    plot_images(
        original_image=original_image,
        downsampled_images=upsampled_images,
        save_path="upsampled_image.png",
    )

    rmse_values = calculate_rmse(original_image, upsampled_images)
    print(50 * "*" + "\n")
    for i, rmse in enumerate(rmse_values, start=1):
        print(f'RMSE for downsampled image {i}->{downsampled_images[i-1].shape}: {rmse}')


if __name__ == '__main__':
    main()

