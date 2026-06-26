import shutil, os, random
from pathlib import Path
from collections import defaultdict

random.seed(42)

image_src = os.path.join(os.getcwd(), "data", 'flickr-image-dataset', 'versions', '1', 'flickr30k_images', 'flickr30k_images')
label_src = os.path.join(os.getcwd(), 'data', 'labels')

# Targets (per project requirements: 500 train + 100 val per class)
TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100

# Categorize each labeled image by what it contains
person_only = []      # has person but no vehicle
vehicle_only = []     # has vehicle but no person
both = []             # has both

for label_file in Path(label_src).glob("*.txt"):
    classes_in_file = set()
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            classes_in_file.add(cls_id)

    if not classes_in_file:
        continue  # empty label file
    if 0 in classes_in_file and 1 in classes_in_file:
        both.append(label_file.stem)
    elif 0 in classes_in_file:
        person_only.append(label_file.stem)
    elif 1 in classes_in_file:
        vehicle_only.append(label_file.stem)

print(f"Pool sizes — person-only: {len(person_only)}, vehicle-only: {len(vehicle_only)}, both: {len(both)}")

random.shuffle(person_only)
random.shuffle(vehicle_only)
random.shuffle(both)

# Strategy:
#   - Use ALL "both" images for free coverage of both classes
#   - Top up with person-only and vehicle-only to hit per-class targets
#
# Count current per-class coverage from "both":
person_count = len(both)    # each "both" image contributes 1 to person count
vehicle_count = len(both)   # and 1 to vehicle count

# How many more we need of each (after "both")
person_need = max(0, (TRAIN_PER_CLASS + VAL_PER_CLASS) - person_count)
vehicle_need = max(0, (TRAIN_PER_CLASS + VAL_PER_CLASS) - vehicle_count)

if len(vehicle_only) < vehicle_need:
    print(f"WARNING: only {len(vehicle_only)} vehicle-only images, need {vehicle_need}. Will use all available.")
    vehicle_need = len(vehicle_only)
if len(person_only) < person_need:
    print(f"WARNING: only {len(person_only)} person-only images, need {person_need}.")
    person_need = len(person_only)

# Build the combined pool
selected = both + vehicle_only[:vehicle_need] + person_only[:person_need]
random.shuffle(selected)

# Split: take TRAIN_PER_CLASS + VAL_PER_CLASS images roughly proportionally
# Simpler: split the selected pool 5:1 (matches 500:100 ratio)
n_val = (len(selected) * VAL_PER_CLASS) // (TRAIN_PER_CLASS + VAL_PER_CLASS)
val_stems = selected[:n_val]
train_stems = selected[n_val:]

# Prepare directories
for split in ["train", "val"]:
    for kind in ["images", "labels"]:
        d = f"data/{kind}/{split}"
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

def copy_split(stems, split):
    for stem in stems:
        shutil.copy2(os.path.join(image_src, f"{stem}.jpg"),
                     os.path.join("data", "images", split, f"{stem}.jpg"))
        shutil.copy2(os.path.join(label_src, f"{stem}.txt"),
                     os.path.join("data", "labels", split, f"{stem}.txt"))

copy_split(train_stems, "train")
copy_split(val_stems, "val")

# Report final class balance
def count_classes(split):
    p, v = 0, 0
    for f in Path(f"data/labels/{split}").glob("*.txt"):
        with open(f) as fh:
            classes = {int(l.split()[0]) for l in fh if l.strip()}
        if 0 in classes: p += 1
        if 1 in classes: v += 1
    return p, v

train_p, train_v = count_classes("train")
val_p, val_v = count_classes("val")
print(f"train: {len(train_stems)} images  |  person: {train_p}, vehicle: {train_v}")
print(f"val:   {len(val_stems)} images  |  person: {val_p}, vehicle: {val_v}")