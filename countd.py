import json
from collections import Counter

# Corrected path to the annotations file
annotations_path = "/Users/benjaminrush/Downloads/EE/self_driving_car_dataset.coco/export/_annotations.coco.json"

# Load the annotations file
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

# Get the category IDs and names
categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# Count the occurrences of each category in the annotations
category_counts = Counter()
for annotation in coco_data["annotations"]:
    category_id = annotation["category_id"]
    category_counts[category_id] += 1

# Print the results
print("Obstacle Counts:")
for category_id, count in category_counts.items():
    category_name = categories.get(category_id, "Unknown")
    print(f"{category_name}: {count}")
