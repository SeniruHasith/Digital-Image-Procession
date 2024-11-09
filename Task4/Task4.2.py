import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarize_and_label():
    # Load the image labeled as Q4.jpg
    image_path = '../images/Q4.jpg'
    image = cv2.imread(image_path)

    # Check if the image is loaded
    if image is None:
        print(f"Could not open or find the image {image_path}")
        return

    # Convert to grayscale if necessary
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, otsu_thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Label the regions in the binarized image
    num_labels, labels_im = cv2.connectedComponents(otsu_thresholded)

    # Create an empty colored image for visualization
    labeled_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Create a color map for different labels
    colors = plt.get_cmap('jet', num_labels)

    # Assign colors to each label
    for label in range(1, num_labels):  # Start from 1 to skip the background
        mask = (labels_im == label)
        # Get color from the colormap and convert to integer
        rgba_color = colors(label)  # Get color as a tuple (R, G, B, A)
        color = (int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255))
        labeled_image[mask] = color  # Assign the color to the masked area

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Binarized Image")
    plt.imshow(otsu_thresholded, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Labeled Regions")
    plt.imshow(labeled_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    binarize_and_label()
