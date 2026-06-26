import torch
import pandas as pd
import os
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ONE_HOT = {'person': 0, 'vehicle': 1}
TARGET_PER_CLASS = 800   

sam = build_sam3_image_model()
sam_processor = Sam3Processor(sam)

image_path = os.path.join(os.getcwd(), "data", 'flickr-image-dataset', 'versions', '1', 'flickr30k_images', 'flickr30k_images')
label_path = os.path.join(os.getcwd(), 'data', 'labels')
os.makedirs(label_path, exist_ok=True)

prompts = pd.read_csv('data/prompts/prompts.csv')

# Categorize rows
prompts['has_person']  = prompts['persons'].notna()
prompts['has_vehicle'] = prompts['vehicles'].notna()

both_df    = prompts[prompts['has_person'] & prompts['has_vehicle']]
vehicle_df = prompts[prompts['has_vehicle'] & ~prompts['has_person']]
person_df  = prompts[prompts['has_person']  & ~prompts['has_vehicle']]

print(f"Pool: both={len(both_df)}, vehicle-only={len(vehicle_df)}, person-only={len(person_df)}")

# Build a balanced work list
# - All "both" first (each contributes to BOTH class counts)
# - Then top up vehicle-only and person-only to reach TARGET_PER_CLASS each
both_rows    = both_df.sample(frac=1, random_state=42)
vehicle_rows = vehicle_df.sample(frac=1, random_state=42)
person_rows  = person_df.sample(frac=1, random_state=42)

# After taking all "both", how many more single-class images do we need?
needed_vehicle = max(0, TARGET_PER_CLASS - len(both_rows))
needed_person  = max(0, TARGET_PER_CLASS - len(both_rows))

selected = pd.concat([
    both_rows,
    vehicle_rows.head(needed_vehicle),
    person_rows.head(needed_person),
])
selected = selected.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Processing {len(selected)} images")


def run_sam3_for_prompt(img, prompt):
    """Returns list of (x1, y1, x2, y2) boxes for the given text prompt."""
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam_processor.set_image(img)
        result = sam_processor.set_text_prompt(state=state, prompt=prompt)
    return result['boxes'].cpu().numpy()


def yolo_line(class_id, x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    if w <= 0 or h <= 0:
        return None
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    return f'{ONE_HOT[class_id]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n'


for row in selected.itertuples():
    img_filepath = os.path.join(image_path, row.image_name)
    if not os.path.exists(img_filepath):
        print(f'image missing: {row.image_name}, skipping')
        continue

    img = Image.open(img_filepath)
    img_w, img_h = img.size

    all_lines = []

    # Run SAM3 once per class that's present in this image
    if row.has_person:
        print(f'SAM3 person inference on {row.image_name}')
        boxes = run_sam3_for_prompt(img, row.persons)
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            line = yolo_line('person', x1, y1, x2, y2, img_w, img_h)
            if line:
                all_lines.append(line)

    if row.has_vehicle:
        print(f'SAM3 vehicle inference on {row.image_name}')
        boxes = run_sam3_for_prompt(img, row.vehicles)
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            line = yolo_line('vehicle', x1, y1, x2, y2, img_w, img_h)
            if line:
                all_lines.append(line)

    if not all_lines:
        print(f'no boxes found for {row.image_name}, skipping')
        continue

    label_filename = os.path.join(label_path, row.image_name.replace('.jpg', '.txt'))
    with open(label_filename, 'w') as fout:
        fout.writelines(all_lines)
