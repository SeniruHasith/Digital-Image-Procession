import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
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

    # Create an overlay image
    overlay_image = cv2.addWeighted(image, 0.5, cv2.cvtColor(otsu_thresholded, cv2.COLOR_GRAY2BGR), 0.5, 0)

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Otsu's Thresholded Image")
    plt.imshow(otsu_thresholded, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Overlay Image")
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Plot histogram with Otsu threshold
    plt.subplot(2, 2, 4)
    plt.title("Histogram with Otsu's Threshold")
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.plot(hist)
    threshold_value = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    plt.axvline(threshold_value, color='r', linestyle='dashed', linewidth=2)
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
