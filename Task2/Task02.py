import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def check_image_exists(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}. Please check the file path.")
    return True


def load_and_check_image(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}. Please check if it's a valid image file.")
    return img


def process_and_display_all(image_path='images/Q2.jpg'):
    try:
        # Check and load image
        check_image_exists(image_path)
        img = load_and_check_image(image_path)
        print("Image loaded successfully!")

        # Perform histogram equalization
        equ = cv2.equalizeHist(img)

        # Create CLAHE objects and apply
        clahe24 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))
        clahe48 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(48, 48))
        clahe_img24 = clahe24.apply(img)
        clahe_img48 = clahe48.apply(img)

        # Create a large figure with all visualizations
        plt.figure(figsize=(20, 12))

        # First row: Original and Equalized Images
        plt.subplot(3, 4, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(3, 4, 2)
        plt.imshow(equ, cmap='gray')
        plt.title('Global Histogram Equalization')
        plt.axis('off')

        # Histograms for original and equalized images
        plt.subplot(3, 4, 3)
        plt.hist(img.ravel(), 256, [0, 256], color='gray', alpha=0.7)
        plt.title('Histogram of Original Image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(3, 4, 4)
        plt.hist(equ.ravel(), 256, [0, 256], color='gray', alpha=0.7)
        plt.title('Histogram of Equalized Image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        # Second row: CLAHE Results
        plt.subplot(3, 4, 5)
        plt.imshow(clahe_img24, cmap='gray')
        plt.title('CLAHE (24x24 tiles)')
        plt.axis('off')

        plt.subplot(3, 4, 6)
        plt.imshow(clahe_img48, cmap='gray')
        plt.title('CLAHE (48x48 tiles)')
        plt.axis('off')

        # Histograms for CLAHE results
        plt.subplot(3, 4, 7)
        plt.hist(clahe_img24.ravel(), 256, [0, 256], color='gray', alpha=0.7)
        plt.title('Histogram of CLAHE (24x24)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(3, 4, 8)
        plt.hist(clahe_img48.ravel(), 256, [0, 256], color='gray', alpha=0.7)
        plt.title('Histogram of CLAHE (48x48)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        # Third row: Difference images
        plt.subplot(3, 4, 9)
        plt.imshow(cv2.absdiff(equ, img), cmap='gray')
        plt.title('Difference: Global - Original')
        plt.axis('off')

        plt.subplot(3, 4, 10)
        plt.imshow(cv2.absdiff(clahe_img24, img), cmap='gray')
        plt.title('Difference: CLAHE24 - Original')
        plt.axis('off')

        plt.subplot(3, 4, 11)
        plt.imshow(cv2.absdiff(clahe_img48, img), cmap='gray')
        plt.title('Difference: CLAHE48 - Original')
        plt.axis('off')

        # Add a text box with summary statistics
        plt.subplot(3, 4, 12)
        plt.axis('off')
        stats_text = (
            f"Image Statistics:\n\n"
            f"Original:\n"
            f"Mean: {img.mean():.1f}\n"
            f"Std: {img.std():.1f}\n\n"
            f"Global HE:\n"
            f"Mean: {equ.mean():.1f}\n"
            f"Std: {equ.std():.1f}\n\n"
            f"CLAHE 24x24:\n"
            f"Mean: {clahe_img24.mean():.1f}\n"
            f"Std: {clahe_img24.std():.1f}\n"
        )
        plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Main execution
if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    image_path = '../images/Q2.jpg'  # Modify this path if your image is in a different location
    process_and_display_all(image_path)