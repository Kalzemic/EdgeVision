import pandas as pd
import os
import re

results_path = os.path.join(os.getcwd(),'data','flickr-image-dataset','versions','1','flickr30k_images','results.csv')

results_df = pd.read_csv(results_path, sep='|')
results_df[' comment'] = results_df[' comment'].fillna('')

person_keywords = ['mans?', 'woman', 'women', 'persons?', 'child', 'children',
                   'men', 'peoples?', 'people', 'boys?', 'girls?', 'kids?',
                   'guys?', 'ladys?', 'ladies', 'adults?', 'babies', 'baby']
vehicle_keywords = ['cars?', 'trucks?', 'bus', 'buses', 'motorcycles?',
                    'bikes?', 'bicycles?', 'vans?', 'scooters?',
                    'tractors?', 'vehicles?', 'taxis?', 'jeeps?']

person_regex  = r'\b(?:' + '|'.join(person_keywords)  + r')\b'
vehicle_regex = r'\b(?:' + '|'.join(vehicle_keywords) + r')\b'

results_df['has_person']  = results_df[' comment'].str.contains(person_regex,  flags=re.IGNORECASE, regex=True)
results_df['has_vehicle'] = results_df[' comment'].str.contains(vehicle_regex, flags=re.IGNORECASE, regex=True)

results_df = results_df.groupby('image_name').agg({
    'has_person':  'any',
    'has_vehicle': 'any',
}).reset_index()

results_df['persons']  = results_df['has_person'].map(lambda b: 'person' if b else '')
results_df['vehicles'] = results_df['has_vehicle'].map(lambda b: 'vehicle' if b else '')

filtered_df = results_df[['image_name', 'persons', 'vehicles']]
filtered_df = filtered_df[
    filtered_df['persons'].str.len().gt(0) |
    filtered_df['vehicles'].str.len().gt(0)
]
filtered_df.to_csv(os.path.join(os.getcwd(),'data','prompts','prompts.csv'), index=False)