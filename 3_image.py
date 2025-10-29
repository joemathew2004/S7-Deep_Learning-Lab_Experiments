import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def load_image(path):
    image_rgb = Image.open(path).convert('RGB')
    image_gray = image_rgb.convert('L')  # Convert to grayscale

    transform = transforms.ToTensor()
    tensor_gray = transform(image_gray)

    return image_rgb, tensor_gray, image_gray


def histogram_equalization(img_tensor):
    img = img_tensor.squeeze().numpy()
    img_eq = cv2.equalizeHist((img * 255).astype(np.uint8))
    img_eq_tensor = torch.from_numpy(img_eq).float() / 255.0
    return img_eq_tensor.unsqueeze(0)


def binarize(img_tensor):
    img = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return torch.from_numpy(binary).float().unsqueeze(0) / 255.0


def morphological_operation(img_tensor, operation='opening', kernel_size=3):
    img = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == 'opening':
        morphed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Unsupported operation. Use 'opening' or 'closing'.")

    return torch.from_numpy(morphed).float().unsqueeze(0) / 255.0


def show_images(images, titles, rows=3, cols=2):
    assert len(images) == len(titles), "Images and titles count must match"
    plt.figure(figsize=(12, 9))

    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        
        if isinstance(images[i], torch.Tensor):
            plt.imshow(images[i].squeeze(), cmap="gray")
        else:
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "3_sample_image.png"

    rgb_image, gray_tensor, gray_pil = load_image(image_path)
    equalized_tensor = histogram_equalization(gray_tensor)
    binary_tensor = binarize(equalized_tensor)
    opened_tensor = morphological_operation(binary_tensor, operation='opening')
    closed_tensor = morphological_operation(binary_tensor, operation='closing')

    show_images(
        [rgb_image, gray_tensor, equalized_tensor, binary_tensor, opened_tensor, closed_tensor],
        ['RGB Image', 'Grayscale', 'Histogram Equalization', 'Binarized', 'Opening', 'Closing'],
        rows=3,
        cols=2
    )

