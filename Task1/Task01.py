import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_and_get_size(image_path):
    """
    Load image and return its dimensions
    """
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Error: Could not load image")
    height, width = img.shape[:2]
    return img, height, width


def invert_rectangle(img, y1, x1, y2, x2):
    """
    Invert colors within specified rectangle coordinates
    """
    # Create a copy of the image
    modified = img.copy()
    # Invert the colors within the rectangle
    modified[y1:y2, x1:x2] = 255 - modified[y1:y2, x1:x2]
    return modified


def create_middle_third_negative(img):
    """
    Create negative of middle third portion
    """
    height, width = img.shape[:2]
    start_x = width // 3
    end_x = (2 * width) // 3

    # Create a copy and invert middle third
    middle_negative = img.copy()
    middle_negative[:, start_x:end_x] = 255 - middle_negative[:, start_x:end_x]
    return middle_negative


def main():
    # Load image and get dimensions
    try:
        image_path = '../images/Q1.jpg'
        original_img, height, width = load_and_get_size(image_path)
        print(f"Image dimensions: {width}x{height} pixels")

        # Get rectangle coordinates from user
        print("\nEnter rectangle coordinates:")
        y1 = int(input("Enter top-left y1 coordinate: "))
        x1 = int(input("Enter top-left x1 coordinate: "))
        y2 = int(input("Enter bottom-right y2 coordinate: "))
        x2 = int(input("Enter bottom-right x2 coordinate: "))

        # Validate coordinates
        if not (0 <= y1 < y2 <= height and 0 <= x1 < x2 <= width):
            raise ValueError("Invalid coordinates! Please check the image dimensions.")

        # Process images
        inverted_rect_img = invert_rectangle(original_img, y1, x1, y2, x2)
        middle_negative_img = create_middle_third_negative(original_img)

        # Convert from BGR to RGB for display
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        inverted_rgb = cv2.cvtColor(inverted_rect_img, cv2.COLOR_BGR2RGB)
        middle_negative_rgb = cv2.cvtColor(middle_negative_img, cv2.COLOR_BGR2RGB)

        # Display results
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(original_rgb)
        plt.axis('off')

        plt.subplot(132)
        plt.title('Inverted Rectangle')
        plt.imshow(inverted_rgb)
        plt.axis('off')

        plt.subplot(133)
        plt.title('Middle Third Negative')
        plt.imshow(middle_negative_rgb)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()



