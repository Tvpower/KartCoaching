import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class GoKartDataset(Dataset):
    def __init__(self, json_file, image_dir, processor):
        self.images_dir = image_dir
        self.processor = processor

        with open(json_file, 'r') as f:
            data = json.load(f)

        self.label_map = {
            0: 'Track_Segment',
            1: 'Turn_in',
            2: 'Apex',
            3: 'Exit',
            4: 'Runner'
        }

        self.segment_type_map = {'Curve': 0, 'Straight': 1, 'Race_Start': 2}
        self.direction_map = {'Left': 0, 'Right': 1, 'Unknown': 2}
        self.point_label_map = {'Turn_in': 0, 'Apex': 1, 'Exit': 2, 'None': 3}

        # parse the data - only keep frames with annotations
        self.data = []
        for item in data['items']:
            if len(item['annotations']) == 0:
                continue

            frame_num = item['attr']['frame']
            image_path = item['image']['path']

            #Extract the track segment attributes and points
            tracks_attrs = None
            points_data = []

            for ann in item['annotations']:
                if ann['type'] == 'label': #track segment
                    # For label annotations, attributes might be in the annotation itself
                    tracks_attrs = ann.get('attributes', {})
                elif ann['type'] == 'points':   #turn_in/apex/exit
                    label_name = self.label_map[ann['label_id']]
                    points_data.append({
                        'label': label_name,
                        'coords': ann['points']
                    })


            #This part is both delicate and tricky.
            #1. We need to make sure that the track segment is present in the frame
            #2. We need to make sure that the track segment has at least one point
            #bc we want the training to only happy on labeled frames
            if not tracks_attrs:
                continue

            if not points_data:
                points_data = [{
                    'label': 'None',
                    'coords': [0.0, 0.0]
                }]

            for point in points_data:
                self.data.append({
                    'frame': frame_num,
                    'img_path': image_path,
                    'segment_type': tracks_attrs.get('Type', 'Unknown'),
                    'curve_number': int(tracks_attrs.get('Number', 0)),
                    'direction': tracks_attrs.get('Direction', 'Unknown'),
                    'point_label' : point['label'],
                    'x': point['coords'][0],
                    'y': point['coords'][1],
                    'has_point': point['label'] != 'None'
                })

        print(f"loaded {len(self.data)} training samples from {len(set([d['frame'] for d in self.data]))} unique frames")
        print(f"samples yes points :) : {sum(1 for d in self.data if d['has_point'])}")
        print(f"samples no points >:( : {sum(1 for d in self.data if not d['has_point'])}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        img_path = os.path.join(self.images_dir, "default" , item['img_path'])
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        # Convert labels to tensors
        segment_type = torch.tensor(self.segment_type_map.get(item['segment_type'], 2), dtype=torch.long)
        curve_number = torch.tensor(max(0, min(13, item['curve_number'] - 1)), dtype=torch.long)  # 1-14 -> 0-13
        direction = torch.tensor(self.direction_map.get(item['direction'], 2), dtype=torch.long)
        point_label = torch.tensor(self.point_label_map.get(item['point_label'], 3), dtype=torch.long)
        coords = torch.tensor([item['x'], item['y']], dtype=torch.float32)
        
        return {
            'pixel_val': pixel_values,
            'segment_type': segment_type,
            'curve_number': curve_number,
            'direction': direction,
            'point_label': point_label,
            'coords': coords
        }