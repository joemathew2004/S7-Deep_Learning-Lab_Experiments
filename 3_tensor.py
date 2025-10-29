import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def load_image(path):
    image_rgb = Image.open(path).convert('RGB')
    image_gray = image_rgb.convert('L')

    # Convert to tensor
    tensor_gray = tf.convert_to_tensor(np.array(image_gray) / 255.0, dtype=tf.float32)
    tensor_gray = tf.expand_dims(tensor_gray, axis=-1)  # shape (H, W, 1)

    return image_rgb, tensor_gray, image_gray

def histogram_equalization(img_tensor):
    img = tf.squeeze(img_tensor).numpy()
    img_eq = cv2.equalizeHist((img * 255).astype(np.uint8))
    img_eq_tensor = tf.convert_to_tensor(img_eq / 255.0, dtype=tf.float32)
    img_eq_tensor = tf.expand_dims(img_eq_tensor, axis=-1)
    return img_eq_tensor

def binarize(img_tensor):
    img = (tf.squeeze(img_tensor).numpy() * 255).astype(np.uint8)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_tensor = tf.convert_to_tensor(binary / 255.0, dtype=tf.float32)
    binary_tensor = tf.expand_dims(binary_tensor, axis=-1)
    return binary_tensor

def morphological_operation(img_tensor, operation='opening', kernel_size=3):
    img = (tf.squeeze(img_tensor).numpy() * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == 'opening':
        morphed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Unsupported operation. Use 'opening' or 'closing'.")

    morphed_tensor = tf.convert_to_tensor(morphed / 255.0, dtype=tf.float32)
    morphed_tensor = tf.expand_dims(morphed_tensor, axis=-1)
    return morphed_tensor

def show_images(images, titles, rows=3, cols=2):
    assert len(images) == len(titles), "Images and titles count must match"
    plt.figure(figsize=(12, 9))

    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        img = images[i]
        if isinstance(img, tf.Tensor):
            plt.imshow(tf.squeeze(img), cmap='gray')
        else:
            plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "3_sample_image.png"  # Replace with your image path

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

