import os
import numpy as np
from PIL import Image
import albumentations as A
from tqdm import tqdm


def create_augmentation_pipeline():
    """Create an augmentation pipeline using Albumentations"""
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomGamma(p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.3),
        A.Resize(256, 256)
    ])
    return transform


def load_and_augment_images(input_dir, output_dir, num_augmented_per_image=5):
    """
    Load images from input directory and apply augmentation

    Args:
        input_dir (str): Directory containing original normal images
        output_dir (str): Directory to save augmented images
        num_augmented_per_image (int): Number of augmented versions to create per original image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the augmentation pipeline
    transform = create_augmentation_pipeline()

    # Process each image in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            image_path = os.path.join(input_dir, filename)
            image = np.array(Image.open(image_path))

            # Generate augmented versions
            for i in range(num_augmented_per_image):
                # Apply augmentation
                augmented = transform(image=image)
                aug_image = augmented['image']

                # Save augmented image
                base_name = os.path.splitext(filename)[0]
                aug_filename = f"{base_name}_aug_{i + 1}.png"
                aug_path = os.path.join(output_dir, aug_filename)

                Image.fromarray(aug_image).save(aug_path)


def main():
    # Configuration
    input_directory = "D:\\Projects\\Image-processing-project\\dataset\\malignant"  # Replace with your input directory
    output_directory = "D:\\Projects\\Image-processing-project\\dataset\\malignant"  # Replace with your output directory
    augmentations_per_image = 5  # Adjust this number based on your needs

    print("Starting image augmentation process...")
    load_and_augment_images(input_directory, output_directory, augmentations_per_image)
    print("Augmentation complete!")

    # Print summary
    original_count = len([f for f in os.listdir(input_directory)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    augmented_count = len([f for f in os.listdir(output_directory)
                           if f.lower().endswith('.png')])

    print(f"\nSummary:")
    print(f"Original images: {original_count}")
    print(f"Augmented images generated: {augmented_count}")
    print(f"Total images available: {original_count + augmented_count}")


if __name__ == "__main__":
    main()
