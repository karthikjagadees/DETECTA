import os
import cv2
import json
from collections import Counter


def get_dataset_paths():
    """
    Returns the paths of the Train and Test dataset folders.
    """
    dataset_root = "dataset"

    train_path = os.path.join(dataset_root, "Train")
    test_path = os.path.join(dataset_root, "Test")

    return train_path, test_path


def verify_dataset(train_path, test_path):
    """
    Verify that the Train and Test folders exist.
    """
    print("\nChecking dataset folders...")

    if os.path.exists(train_path):
        print("[OK] Train folder found.")
    else:
        print("[ERROR] Train folder NOT found.")

    if os.path.exists(test_path):
        print("[OK] Test folder found.")
    else:
        print("[ERROR] Test folder NOT found.")


def count_images(folder_path):
    """
    Count the number of image files in the Image folder.
    """
    image_folder = os.path.join(folder_path, "Image")

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    image_count = 0

    for file in os.listdir(image_folder):
        if file.lower().endswith(image_extensions):
            image_count += 1

    return image_count


def count_masks(folder_path):
    """
    Count the number of segmentation mask files.
    """
    mask_folder = os.path.join(folder_path, "GT_Object")

    mask_extensions = (".png", ".jpg", ".jpeg", ".bmp")

    mask_count = 0

    for file in os.listdir(mask_folder):
        if file.lower().endswith(mask_extensions):
            mask_count += 1

    return mask_count


def verify_image_mask_pairs(folder_path):
    """
    Verify that every image has a corresponding segmentation mask.
    """
    image_folder = os.path.join(folder_path, "Image")
    mask_folder = os.path.join(folder_path, "GT_Object")

    image_files = {
        os.path.splitext(file)[0]
        for file in os.listdir(image_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    }

    mask_files = {
        os.path.splitext(file)[0]
        for file in os.listdir(mask_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    }

    missing_masks = image_files - mask_files
    extra_masks = mask_files - image_files

    return missing_masks, extra_masks


def check_corrupted_files(folder_path, subfolder):
    """
    Check for corrupted image or mask files.
    """
    folder = os.path.join(folder_path, subfolder)

    corrupted_files = []

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    for file in os.listdir(folder):

        if file.lower().endswith(image_extensions):

            file_path = os.path.join(folder, file)

            image = cv2.imread(file_path)

            if image is None:
                corrupted_files.append(file)

    return corrupted_files

def analyze_image_dimensions(folder_path, subfolder):
    """
    Analyze image dimensions in the given folder.
    """

    folder = os.path.join(folder_path, subfolder)

    dimensions = Counter()

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    for file in os.listdir(folder):

        if file.lower().endswith(image_extensions):

            file_path = os.path.join(folder, file)

            image = cv2.imread(file_path)

            if image is not None:

                height, width = image.shape[:2]

                dimensions[(width, height)] += 1

    return dimensions

def verify_image_mask_dimensions(folder_path):
    """
    Verify that every image and its corresponding mask
    have the same dimensions.
    """

    image_folder = os.path.join(folder_path, "Image")
    mask_folder = os.path.join(folder_path, "GT_Object")

    mismatched_files = []

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    for file in os.listdir(image_folder):

        if file.lower().endswith(image_extensions):

            image_path = os.path.join(image_folder, file)

            mask_name = os.path.splitext(file)[0] + ".png"
            mask_path = os.path.join(mask_folder, mask_name)

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            if image is None or mask is None:
                continue

            if image.shape[:2] != mask.shape[:2]:
                mismatched_files.append(file)

    return mismatched_files

def main():
    print("=" * 60)
    print("      COD10K Dataset Analysis & Verification")
    print("=" * 60)

    train_path, test_path = get_dataset_paths()

    print(f"Training Folder : {train_path}")
    print(f"Testing Folder  : {test_path}")

    verify_dataset(train_path, test_path)

    train_images = count_images(train_path)
    test_images = count_images(test_path)

    print(f"\nTraining Images : {train_images}")
    print(f"Testing Images  : {test_images}")

    train_masks = count_masks(train_path)
    test_masks = count_masks(test_path)

    print(f"\nTraining Masks  : {train_masks}")
    print(f"Testing Masks   : {test_masks}")

    train_missing, train_extra = verify_image_mask_pairs(train_path)
    test_missing, test_extra = verify_image_mask_pairs(test_path)

    print("\nImage-Mask Pair Verification")
    print("-" * 35)

    print(f"Training Missing Masks : {len(train_missing)}")
    print(f"Training Extra Masks   : {len(train_extra)}")

    print(f"Testing Missing Masks  : {len(test_missing)}")
    print(f"Testing Extra Masks    : {len(test_extra)}")

    train_corrupted_images = check_corrupted_files(train_path, "Image")
    test_corrupted_images = check_corrupted_files(test_path, "Image")

    train_corrupted_masks = check_corrupted_files(train_path, "GT_Object")
    test_corrupted_masks = check_corrupted_files(test_path, "GT_Object")

    print("\nCorrupted File Verification")
    print("-" * 35)

    print(f"Training Corrupted Images : {len(train_corrupted_images)}")
    print(f"Testing Corrupted Images  : {len(test_corrupted_images)}")

    print(f"Training Corrupted Masks  : {len(train_corrupted_masks)}")
    print(f"Testing Corrupted Masks   : {len(test_corrupted_masks)}")

    train_dimensions = analyze_image_dimensions(train_path, "Image")
    test_dimensions = analyze_image_dimensions(test_path, "Image")

    print("\nImage Dimension Analysis")
    print("-" * 35)

    print(f"Unique Training Image Sizes : {len(train_dimensions)}")
    print(f"Unique Testing Image Sizes  : {len(test_dimensions)}")

    print("\nTop 5 Training Image Sizes:")

    for size, count in train_dimensions.most_common(5):
        print(f"{size} : {count}")

    print("\nTop 5 Testing Image Sizes:")

    for size, count in test_dimensions.most_common(5):
        print(f"{size} : {count}")

    train_mismatched = verify_image_mask_dimensions(train_path)
    test_mismatched = verify_image_mask_dimensions(test_path)

    print("\nImage-Mask Dimension Verification")
    print("-" * 35)

    print(f"Training Mismatched Dimensions : {len(train_mismatched)}")
    print(f"Testing Mismatched Dimensions  : {len(test_mismatched)}")


if __name__ == "__main__":
    main()