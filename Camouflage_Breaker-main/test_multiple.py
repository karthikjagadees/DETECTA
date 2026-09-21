# test_variety.py
from inference.pipeline import CamouflageBreakerPipeline
import glob
import os
import random

print("=" * 50)
print("Testing Variety of Animals")
print("=" * 50)

pipeline = CamouflageBreakerPipeline(
    'saved_models/resunet_best.pth',
    'saved_models/classifier_best.pth',
    'saved_models/class_mapping.json'
)

# Get all test images
all_images = glob.glob('dataset/Test/Image/*.jpg')

# Group by animal type (extract animal name from filename)
animal_groups = {}
for img in all_images:
    filename = os.path.basename(img)
    parts = filename.split('-')
    if len(parts) >= 6:
        animal = parts[5].replace('.jpg', '')
        # Remove numbers from animal name
        animal = ''.join([c for c in animal if not c.isdigit()])
        if animal not in animal_groups:
            animal_groups[animal] = []
        animal_groups[animal].append(img)

print(f"\nFound {len(animal_groups)} different animal types\n")

# Test one image from each animal type (up to 10)
test_images = []
for animal, images in list(animal_groups.items())[:10]:
    test_images.append(images[0])

print(f"Testing {len(test_images)} different animals:\n")

for img in test_images:
    filename = os.path.basename(img)
    result = pipeline.predict(img)
    print(f"{filename:<60} -> {result['class_name']} ({result['confidence']:.1f}%)")