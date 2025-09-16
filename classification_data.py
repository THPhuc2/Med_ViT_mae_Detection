import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import os

class XrayClassificationDataset(Dataset):
    def __init__(self, df, img_root, transform=None):
        self.img_root = img_root
        self.transform = transform

        # ✅ Tính nhãn nhị phân:
        #   - Nếu tất cả class_id = 14 → No Finding → label = 1
        #   - Nếu có bất kỳ class_id != 14 → Có bệnh → label = 0
        grouped = df.groupby("filename")["class_id"].apply(
            lambda x: 1 if (x == 14).all() else 0
        ).reset_index()

        grouped.columns = ["filename", "binary_label"]
        self.df = grouped

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        label = row["binary_label"]

        img_path = os.path.join(self.img_root, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

from torchvision import transforms
import pandas as pd

# Load CSV detection
df = pd.read_csv("/home/datnvt/project/Medical_CLARA/data_detection/xray_detection_combined.csv")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
dataset = XrayClassificationDataset(
    df=df,
    img_root="/home/datnvt/project/Medical_CLARA/all_images",
    transform=transform
)

# Kiểm tra thử
img, label = dataset[0]
print("Ảnh:", img.shape)
print("Label:", label)
