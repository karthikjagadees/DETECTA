# fix_class_mapping.py
import os
import json
import re

def extract_animal_names():
    """Extract animal names from COD10K filenames"""
    animal_names = {}
    
    # Check both Train and Test folders
    for split in ['Train', 'Test']:
        image_dir = f'dataset/{split}/Image'
        if not os.path.exists(image_dir):
            continue
            
        for filename in os.listdir(image_dir):
            if not filename.endswith('.jpg'):
                continue
                
            # Parse filename: COD10K-CAM-1-Aquatic-1-BatFish-2.jpg
            try:
                parts = filename.split('-')
                if len(parts) >= 6:
                    # Get class ID (index 4)
                    class_id = int(parts[4])
                    # Get animal name (index 5, before the number)
                    animal_part = parts[5]
                    # Remove .jpg and trailing number
                    animal_name = re.sub(r'\d+\.jpg$', '', animal_part)
                    animal_name = animal_name.replace('.jpg', '')
                    
                    if class_id not in animal_names:
                        animal_names[class_id] = animal_name
            except:
                continue
    
    return animal_names

# Extract names
animal_names = extract_animal_names()
print(f"Found {len(animal_names)} unique animal names")

# Sort by class ID
class_mapping = {str(cid): name for cid, name in sorted(animal_names.items())}

# Save to file
with open('saved_models/class_mapping.json', 'w') as f:
    json.dump(class_mapping, f, indent=2)

print("\nClass mapping:")
for cid, name in sorted(class_mapping.items(), key=lambda x: int(x[0])):
    print(f"  {cid}: {name}")

print(f"\n✅ Saved {len(class_mapping)} classes to saved_models/class_mapping.json")