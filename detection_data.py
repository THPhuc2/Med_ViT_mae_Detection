import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class DetectionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transforms=None, img_size=224):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transforms = transforms
        self.img_size = img_size

        # Drop NaN bounding boxes
        self.data = self.data.dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])

        # Ensure class_id is int (avoid float bug)
        self.data["class_id"] = self.data["class_id"].astype(int)

        # Remove class_id = 14 (no finding)
        self.data = self.data[self.data['class_id'] != 14]

        # Re-index class_id to [0..N-1]
        unique_classes = sorted(self.data['class_id'].unique())
        self.class_map = {cls_id: i for i, cls_id in enumerate(unique_classes)}
        self.data['class_id'] = self.data['class_id'].map(self.class_map)

        print(f"🟢 Kept {len(self.class_map)} classes: {self.class_map}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Resize image to (img_size, img_size)
        image = image.resize((self.img_size, self.img_size))
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        # Scale bbox accordingly
        x_min = row["x_min"] * scale_x
        y_min = row["y_min"] * scale_y
        x_max = row["x_max"] * scale_x
        y_max = row["y_max"] * scale_y

        boxes = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        labels = torch.tensor([row["class_id"]], dtype=torch.int64)

        # Add optional COCO-style fields
        area = (x_max - x_min) * (y_max - y_min)
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([area], dtype=torch.float32),
            "iscrowd": iscrowd
        }

        if self.transforms:
            image = self.transforms(image)
        else:
            image = ToTensor()(image)

        # Debug assert
        assert image.shape[1:] == (self.img_size, self.img_size), f"❌ Image shape wrong: {image.shape}"

        return image, target
