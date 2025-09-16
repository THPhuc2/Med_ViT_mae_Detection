import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViTMAEModel, ViTMAEConfig
import math
from detection_data import DetectionDataset, collate_fn
from classification_data import XrayClassificationDataset
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import os
from lightning.pytorch.loggers import WandbLogger
import logging
logger = logging.getLogger(__name__)
os.system("wandb login --relogin d8dbd91c9717ac3a104742d8f247ae4012526297")    # của Phúc d8dbd91c9717ac3a104742d8f247ae4012526297  138c38699b36fb0223ca0f94cde30c6d531895ca
# wandb.init(project="mae_training", sync_tensorboard=True)
# wandb.init(project="mae_training")
wandb_logger = WandbLogger(
    project="mae_training_classification",
    log_model="all",
)

# --- Positional Encoding ---
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)

# --- MLP ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
# --- DETR Head ---
class DETRHead(nn.Module):
    def __init__(self, d_model, num_classes, num_queries=100):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

    def forward(self, features):
        batch_size = features.size(0)
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        features = features.transpose(0, 1)
        queries = queries.transpose(0, 1)
        hs = self.transformer_decoder(queries, features)
        hs = hs.transpose(0, 1)
        class_logits = self.class_embed(hs)
        bbox_coords = self.bbox_embed(hs).sigmoid()
        return {'pred_logits': class_logits, 'pred_boxes': bbox_coords}
    

class ViTMAEClassifier(nn.Module):
    def __init__(self, vit_mae, num_classes=2):
        super().__init__()
        self.vit_mae = vit_mae  # 👈 bạn truyền encoder đã load từ ckpt vào

        hidden_size = self.vit_mae.config.hidden_size
        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vit_mae(pixel_values)
        x = outputs.last_hidden_state  # [B, N, C]
        x = x.mean(dim=1)              # GAP
        return self.cls_head(x)
    

# 2. Load checkpoint
def load_mae_weights(mae_model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("vit_mae."):
            new_key = k[len("vit_mae."):]
            new_state_dict[new_key] = v

    missing, unexpected = mae_model.load_state_dict(new_state_dict, strict=False)
    print("✅ MAE weights loaded.")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

class HungarianMatcher(nn.Module):
    def forward(self, outputs, targets):
        indices = []
        for pred_logits, pred_boxes, target in zip(outputs['pred_logits'], outputs['pred_boxes'], targets):
            tgt_boxes = target['boxes']
            if tgt_boxes.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            ious = box_iou(pred_boxes, tgt_boxes)  # (num_queries, num_targets)
            cost = 1 - ious

            # ⚠️ Check xem cost có hợp lệ không
            if not torch.isfinite(cost).all():
                # print("❌ Cost matrix has NaN or Inf!")
                # print("📦 pred_boxes:", pred_boxes)
                # print("🎯 tgt_boxes:", tgt_boxes)
                # Có thể bỏ qua batch này hoặc thay thế giá trị lỗi
                cost = torch.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)

            src_idx, tgt_idx = linear_sum_assignment(cost.cpu().detach().numpy())
            indices.append((
                torch.as_tensor(src_idx, dtype=torch.int64),
                torch.as_tensor(tgt_idx, dtype=torch.int64)
            ))

        return indices
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou_loss
from torchvision.ops import sigmoid_focal_loss

from torchvision.ops import generalized_box_iou
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher

        self.alpha = 0.25
        self.gamma = 2.0

        self.empty_weight = torch.ones(self.num_classes + 1)
        self.empty_weight[-1] = 0.1  # nhẹ cho "no object"

    def forward(self, outputs, targets):
        pred_logits = outputs['pred_logits']  # (B, num_queries, num_classes + 1)
        pred_boxes = outputs['pred_boxes']    # (B, num_queries, 4)

        indices = self.matcher(outputs, targets)

        idx = self._get_src_permutation_idx(indices)

        # ---------- ✅ FOCAL LOSS ----------
        target_classes_o = torch.cat([
            t['labels'][J] for t, (_, J) in zip(targets, indices)
        ], dim=0)

        pred_classes = pred_logits[idx]  # (N, num_classes + 1)

        # Chuyển label -> one hot
        target_onehot = F.one_hot(
            target_classes_o,
            num_classes=self.num_classes + 1
        ).float()

        # Clamp tránh NaN
        pred_classes = pred_classes.clamp(min=-10, max=10)

        prob = pred_classes.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(pred_classes, target_onehot, reduction='none')
        p_t = prob * target_onehot + (1 - prob) * (1 - target_onehot)
        alpha_factor = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
        focal_weight = alpha_factor * (1 - p_t) ** self.gamma
        loss_ce = (focal_weight * ce_loss).mean()

        # ---------- ✅ L1 BBOX LOSS ----------
        target_boxes = torch.cat([
            t["boxes"][J] for t, (_, J) in zip(targets, indices)
        ], dim=0)
        pred_boxes_matched = pred_boxes[idx]
        loss_bbox = F.l1_loss(pred_boxes_matched, target_boxes, reduction='mean')

        # ---------- ✅ GIoU ----------
        loss_giou = 1.0 - torch.diag(generalized_box_iou(pred_boxes_matched, target_boxes)).mean()

        # ---------- ✅ TOTAL ----------
        loss_total = loss_ce + 5.0 * loss_bbox + 2.0 * loss_giou

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_total": loss_total
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
from tqdm import tqdm

def train_step(model, criterion, data_loader, optimizer, device, epoch=None, rank=0):
    model.train()
    total_loss = 0

    pbar = tqdm(
        enumerate(data_loader), 
        total=len(data_loader), 
        disable=(rank != 0),
        desc=f"Epoch {epoch}"
    )

    for batch_idx, (images, labels) in pbar:
        if isinstance(images, tuple):
            images = images[0]

        images = images.to(device)
        labels = torch.as_tensor(labels, device=device)


        outputs = model(images)                   # [B, num_classes]
        loss = criterion(outputs, labels)         # CrossEntropyLoss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if rank == 0:
            pbar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
            })

    return total_loss / len(data_loader)
def eval_step(model, criterion, data_loader, device, rank=0):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in data_loader:
            if isinstance(images, tuple):
                images = images[0]

            images = images.to(device)
            labels = torch.as_tensor(labels, device=device)



            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
import pandas as pd
import wandb
from torchvision import transforms

# Giả sử các hàm và lớp sau đã import:
# - ViTMAEDETR
# - load_mae_weights
# - SetCriterion, HungarianMatcher
# - DetectionDataset, collate_fn
# - train_step, eval_step

epochs = 200
batch_sizes = 350

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        wandb.init(
            project="detection-vitmae",
            name=f"run-vitmae-{wandb.util.generate_id()}",
            config={"epochs": epochs, "batch_size": batch_sizes, "lr": 5e-5, "num_classes": 15}
        )

    # Transform
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor()
    ])
    # Load + split CSV
    csv_path = "/home/datnvt/project/Medical_CLARA/data_detection/xray_detection_combined.csv"
    img_root = "/home/datnvt/project/Medical_CLARA/all_images"

    df = pd.read_csv(csv_path)
    filenames = df["filename"].unique()
    train_fns, val_fns = train_test_split(filenames, test_size=0.2, random_state=42)

    train_df = df[df["filename"].isin(train_fns)].reset_index(drop=True)
    val_df = df[df["filename"].isin(val_fns)].reset_index(drop=True)

    # Datasets
    # train_dataset = DetectionDataset(train_df, img_root, transform=transform)
    # val_dataset = DetectionDataset(val_df, img_root, transform=transform)

    train_dataset = XrayClassificationDataset(train_df, img_root, transform=transform)
    val_dataset = XrayClassificationDataset(val_df, img_root, transform=transform)

    # Samplers
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_sizes, sampler=val_sampler, collate_fn=collate_fn)

    # Load model
    NUM_CLASSES = 15
    # model = ViTMAEDETR(num_classes=NUM_CLASSES, pretrained_mae=False)
    vit_mae = ViTMAEModel(ViTMAEConfig())
    ckpt_path = "/home/datnvt/project/Medical_CLARA/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=033-valid/loss=0.02.ckpt"
    load_mae_weights(vit_mae, ckpt_path)
    model=ViTMAEClassifier(num_classes=NUM_CLASSES, vit_mae=vit_mae)
    # ckpt_path = "/home/datnvt/project/Medical_CLARA/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=033-valid/loss=0.02.ckpt"
    # load_mae_weights(model, ckpt_path)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"❄️ Frozen: {name}")
        else:
            print(f"🔥 Trainable: {name}")

    # 🧊 Nếu bạn nghi encoder đang bị freeze, thì unfreeze toàn bộ tại đây:
    # 🎯 Sau khi load weight, unfreeze toàn bộ trước khi DDP

    # ✅ Force unfreeze toàn bộ encoder + head
    for name, param in model.named_parameters():
        param.requires_grad = True
        print(f"🔥 Unfrozen: {name}")





    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # Loss
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=NUM_CLASSES, matcher=matcher).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train_step(model, criterion, train_loader, optimizer, device, epoch=epoch, rank=local_rank)
        val_loss = eval_step(model, criterion, val_loader, device, rank=local_rank)

        if local_rank == 0:
            print(f"\n✅ Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            wandb.log({
                "train/loss_total": train_loss,
                "val/loss_total": val_loss,
                "epoch": epoch
            })

            # ✅ SAVE every 5 epochs
            if local_rank == 0:  # nếu dùng DDP
                print(f"🔁 Epoch {epoch}/{epochs}")
                if epoch % 5 == 0 or epoch == epochs - 1:
                    save_path = f"/home/datnvt/project/Medical_CLARA/output_detection/ckpt_epoch_{epoch:03}.pth"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.module.state_dict(), save_path)
                    print(f"📦 Saved checkpoint to {save_path}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = f"/home/datnvt/project/Medical_CLARA/output_detection/best_model.pth"
                    torch.save(model.module.state_dict(), save_path)
                    print(f"🌟 New best model saved at Epoch {epoch} with Val Loss: {val_loss:.4f}")



    # Save model
    if local_rank == 0:
        save_path = "/home/datnvt/project/Medical_CLARA/output_detection/detection_vitmae_final.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.module.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")
        wandb.finish()
    
if __name__ == "__main__":
    main()
