import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the image labeled as Q3.jpg
    image_path = '../images/Q3.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded
    if image is None:
        print(f"Could not open or find the image {image_path}")
        return

    # Define Prewitt filter kernels
    prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    # Apply Prewitt filter and convert to float32
    prewitt_x = cv2.filter2D(image, cv2.CV_32F, prewitt_kernel_x)
    prewitt_y = cv2.filter2D(image, cv2.CV_32F, prewitt_kernel_y)
    prewitt = cv2.magnitude(prewitt_x, prewitt_y)

    # Apply Sobel filter (OpenCV has built-in Sobel)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)

    # Define Roberts filter kernels
    roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Apply Roberts filter and convert to float32
    roberts_x = cv2.filter2D(image, cv2.CV_32F, roberts_kernel_x)
    roberts_y = cv2.filter2D(image, cv2.CV_32F, roberts_kernel_y)
    roberts = cv2.magnitude(roberts_x, roberts_y)

    # Display the results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Prewitt Horizontal")
    plt.imshow(prewitt_x, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Prewitt Vertical")
    plt.imshow(prewitt_y, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Sobel Filter")
    plt.imshow(sobel, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Roberts Diagonal")
    plt.imshow(roberts_x, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Roberts Anti-Diagonal")
    plt.imshow(roberts_y, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
