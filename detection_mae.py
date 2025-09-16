# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from transformers import ViTMAEModel, ViTMAEConfig
# import math
# from detection_data import DetectionDataset, collate_fn
# from classification_data import XrayClassificationDataset
# import argparse
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# import wandb
# import os
# from lightning.pytorch.loggers import WandbLogger
# import logging
# logger = logging.getLogger(__name__)
# os.system("wandb login --relogin d8dbd91c9717ac3a104742d8f247ae4012526297")    # của Phúc d8dbd91c9717ac3a104742d8f247ae4012526297  138c38699b36fb0223ca0f94cde30c6d531895ca
# # wandb.init(project="mae_training", sync_tensorboard=True)
# # wandb.init(project="mae_training")
# wandb_logger = WandbLogger(
#     project="mae_training",
#     log_model="all",
# )

# # --- Positional Encoding ---
# class PositionalEncoding2D(nn.Module):
#     def __init__(self, d_model, max_len=1000):
#         super().__init__()
#         self.d_model = d_model
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         seq_len = x.size(1)
#         return x + self.pe[:seq_len].unsqueeze(0)

# # --- MLP ---
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x

# # --- DETR Head ---
# class DETRHead(nn.Module):
#     def __init__(self, d_model, num_classes, num_queries=100):
#         super().__init__()
#         self.query_embed = nn.Embedding(num_queries, d_model)
#         decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
#         self.class_embed = nn.Linear(d_model, num_classes + 1)
#         self.bbox_embed = MLP(d_model, d_model, 4, 3)

#     def forward(self, features):
#         batch_size = features.size(0)
#         queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
#         features = features.transpose(0, 1)
#         queries = queries.transpose(0, 1)
#         hs = self.transformer_decoder(queries, features)
#         hs = hs.transpose(0, 1)
#         class_logits = self.class_embed(hs)
#         bbox_coords = self.bbox_embed(hs).sigmoid()
#         return {'pred_logits': class_logits, 'pred_boxes': bbox_coords}

# # --- ViTMAE + DETR --- DETECTION MODEL ---
# """
# MODEL THỰC HIỆN:
# - Sử dụng ViTMAE làm encoder để trích xuất đặc trưng từ ảnh
# - Sử dụng DETR head để dự đoán các bounding box và class labels
# """
# class ViTMAEDETR(nn.Module):
#     def __init__(self, num_classes, num_queries=100, pretrained_mae=False):
#         super().__init__()
#         if pretrained_mae:
#             # self.vit_mae = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
#             pass
#         else:
#             config = ViTMAEConfig()
#             self.vit_mae = ViTMAEModel(config)

#         d_model = self.vit_mae.config.hidden_size
#         self.pos_encoding = PositionalEncoding2D(d_model)
#         self.detr_head = DETRHead(d_model, num_classes, num_queries)

#     def forward(self, pixel_values):
#         outputs = self.vit_mae(pixel_values, output_hidden_states=True)
#         encoder_outputs = outputs.last_hidden_state
#         encoder_outputs = self.pos_encoding(encoder_outputs)
#         return self.detr_head(encoder_outputs)
# import torch.nn as nn
# from transformers import ViTMAEModel, ViTMAEConfig

# # class ViTMAEClassifier(nn.Module):
# #     def __init__(self, vit_mae, num_classes=2):
# #         super().__init__()
# #         self.vit_mae = vit_mae  # 👈 bạn truyền encoder đã load từ ckpt vào

# #         hidden_size = self.vit_mae.config.hidden_size
# #         self.cls_head = nn.Sequential(
# #             nn.LayerNorm(hidden_size),
# #             nn.Linear(hidden_size, num_classes)
# #         )

# #     def forward(self, pixel_values):
# #         outputs = self.vit_mae(pixel_values)
# #         x = outputs.last_hidden_state  # [B, N, C]
# #         x = x.mean(dim=1)              # GAP
# #         return self.cls_head(x)
# # --- Load pretrained MAE checkpoint ---
# def load_mae_weights(model, ckpt_path):
#     ckpt = torch.load(ckpt_path, map_location='cpu')
#     state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith("vit_mae."):
#             new_key = k[len("vit_mae."):]
#             new_state_dict[new_key] = v

#     missing, unexpected = model.vit_mae.load_state_dict(new_state_dict, strict=False)
#     print("✅ MAE weights loaded.")
#     print("Missing keys:", missing)
#     print("Unexpected keys:", unexpected)

# # --- Hungarian Matcher (dummy) ---
# # class HungarianMatcher(nn.Module):
# #     def forward(self, outputs, targets):
# #         return [(torch.arange(0, len(t["labels"])), torch.arange(0, len(t["labels"]))) for t in targets]
# from torchvision.ops import box_iou
# from scipy.optimize import linear_sum_assignment

# from scipy.optimize import linear_sum_assignment
# from torchvision.ops import box_iou

# class HungarianMatcher(nn.Module):
#     def forward(self, outputs, targets):
#         indices = []
#         for pred_logits, pred_boxes, target in zip(outputs['pred_logits'], outputs['pred_boxes'], targets):
#             tgt_boxes = target['boxes']
#             if tgt_boxes.numel() == 0:
#                 indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
#                 continue

#             ious = box_iou(pred_boxes, tgt_boxes)  # (num_queries, num_targets)
#             cost = 1 - ious

#             # ⚠️ Check xem cost có hợp lệ không
#             if not torch.isfinite(cost).all():
#                 # print("❌ Cost matrix has NaN or Inf!")
#                 # print("📦 pred_boxes:", pred_boxes)
#                 # print("🎯 tgt_boxes:", tgt_boxes)
#                 # Có thể bỏ qua batch này hoặc thay thế giá trị lỗi
#                 cost = torch.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)

#             src_idx, tgt_idx = linear_sum_assignment(cost.cpu().detach().numpy())
#             indices.append((
#                 torch.as_tensor(src_idx, dtype=torch.int64),
#                 torch.as_tensor(tgt_idx, dtype=torch.int64)
#             ))

#         return indices



# # --- Dummy Criterion (classification + bbox) ---
# # class SetCriterion(nn.Module):
# #     def __init__(self, num_classes):
# #         super().__init__()
# #         self.num_classes = num_classes
# #         self.empty_weight = torch.ones(num_classes + 1)
# #         self.empty_weight[-1] = 0.1

# #     def forward(self, outputs, targets):
# #         return {"loss_ce": torch.tensor(0.0), "loss_bbox": torch.tensor(0.0), "loss_giou": torch.tensor(0.0)}
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SetCriterion(nn.Module):
#     def __init__(self, num_classes, matcher):
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.class_loss_fn = nn.CrossEntropyLoss()
#         self.bbox_loss_fn = nn.L1Loss()

#         # ✅ Trọng số cho lớp "no object"
#         self.empty_weight = torch.ones(self.num_classes + 1)
#         self.empty_weight[-1] = 0.1  # trọng số nhỏ cho lớp "no object"

#     def forward(self, outputs, targets):
#         """
#         outputs:
#             - pred_logits: (B, num_queries, num_classes+1)
#             - pred_boxes: (B, num_queries, 4)

#         targets:
#             - list of dicts, each with 'labels' (num_objs,) and 'boxes' (num_objs, 4)
#         """
#         pred_logits = outputs['pred_logits']  # (B, num_queries, num_classes + 1)
#         pred_boxes = outputs['pred_boxes']    # (B, num_queries, 4)

#         indices = self.matcher(outputs, targets)  # list of (src_idx, tgt_idx)

#         # Classification loss
#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0)
#         pred_classes = pred_logits[idx]
#         loss_ce = F.cross_entropy(
#             pred_classes,
#             target_classes_o,
#             weight=self.empty_weight.to(pred_classes.device)
#         )

#         # BBox loss (L1)
#         target_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0)
#         pred_boxes_matched = pred_boxes[idx]
#         loss_bbox = F.l1_loss(pred_boxes_matched, target_boxes)

#         return {
#             "loss_ce": loss_ce,
#             "loss_bbox": loss_bbox,
#             "loss_total": loss_ce + loss_bbox
#         }

#     def _get_src_permutation_idx(self, indices):
#         # Lấy index cho các query được match
#         batch_idx = torch.cat([
#             torch.full_like(src, i) for i, (src, _) in enumerate(indices)
#         ])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         # Lấy index cho ground truth tương ứng
#         batch_idx = torch.cat([
#             torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
#         ])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import box_iou, generalized_box_iou_loss
# from torchvision.ops import sigmoid_focal_loss

# from torchvision.ops import generalized_box_iou
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import generalized_box_iou

# # class SetCriterion(nn.Module):
# #     def __init__(self, num_classes, matcher):
# #         super().__init__()
# #         self.num_classes = num_classes
# #         self.matcher = matcher

# #         self.alpha = 0.25
# #         self.gamma = 2.0

# #         self.empty_weight = torch.ones(self.num_classes + 1)
# #         self.empty_weight[-1] = 0.1  # nhẹ cho "no object"

# #     def forward(self, outputs, targets):
# #         pred_logits = outputs['pred_logits']  # (B, num_queries, num_classes + 1)
# #         pred_boxes = outputs['pred_boxes']    # (B, num_queries, 4)

# #         indices = self.matcher(outputs, targets)

# #         idx = self._get_src_permutation_idx(indices)

# #         # ---------- ✅ FOCAL LOSS ----------
# #         target_classes_o = torch.cat([
# #             t['labels'][J] for t, (_, J) in zip(targets, indices)
# #         ], dim=0)

# #         pred_classes = pred_logits[idx]  # (N, num_classes + 1)

# #         # Chuyển label -> one hot
# #         target_onehot = F.one_hot(
# #             target_classes_o,
# #             num_classes=self.num_classes + 1
# #         ).float()

# #         # Clamp tránh NaN
# #         pred_classes = pred_classes.clamp(min=-10, max=10)

# #         prob = pred_classes.sigmoid()
# #         ce_loss = F.binary_cross_entropy_with_logits(pred_classes, target_onehot, reduction='none')
# #         p_t = prob * target_onehot + (1 - prob) * (1 - target_onehot)
# #         alpha_factor = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
# #         focal_weight = alpha_factor * (1 - p_t) ** self.gamma
# #         loss_ce = (focal_weight * ce_loss).mean()

# #         # ---------- ✅ L1 BBOX LOSS ----------
# #         target_boxes = torch.cat([
# #             t["boxes"][J] for t, (_, J) in zip(targets, indices)
# #         ], dim=0)
# #         pred_boxes_matched = pred_boxes[idx]
# #         loss_bbox = F.l1_loss(pred_boxes_matched, target_boxes, reduction='mean')

# #         # ---------- ✅ GIoU ----------
# #         loss_giou = 1.0 - torch.diag(generalized_box_iou(pred_boxes_matched, target_boxes)).mean()

# #         # ---------- ✅ TOTAL ----------
# #         loss_total = loss_ce + 5.0 * loss_bbox + 2.0 * loss_giou

# #         return {
# #             "loss_ce": loss_ce,
# #             "loss_bbox": loss_bbox,
# #             "loss_giou": loss_giou,
# #             "loss_total": loss_total
# #         }

# #     def _get_src_permutation_idx(self, indices):
# #         batch_idx = torch.cat([
# #             torch.full_like(src, i) for i, (src, _) in enumerate(indices)
# #         ])
# #         src_idx = torch.cat([src for (src, _) in indices])
# #         return batch_idx, src_idx


# # # --- Train step ---
# # def train_step(model, criterion, data_loader, optimizer, device):
# #     model.train()
# #     total_loss = 0
# #     for batch_idx, (images, targets) in enumerate(data_loader):
# #         images = images.to(device)
# #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
# #         outputs = model(images)
# #         loss_dict = criterion(outputs, targets)
# #         losses = sum(loss_dict.values())
# #         optimizer.zero_grad()
# #         losses.backward()
# #         optimizer.step()
# #         total_loss += losses.item()
# #         print(f"Batch {batch_idx} - Loss: {losses.item():.4f}")
# #     return total_loss / len(data_loader)
# # --- Train step ---
# # def train_step(model, criterion, data_loader, optimizer, device):
# #     model.train()
# #     total_loss = 0

# #     for batch_idx, (images, targets) in enumerate(data_loader):
# #         # ✅ Nếu images là tuple (ví dụ (image_tensor, ...)), lấy phần đầu tiên
# #         if isinstance(images, tuple):
# #             images = images[0]

# #         images = images.to(device)
# #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# #         outputs = model(images)
# #         loss_dict = criterion(outputs, targets)
# #         # print("🧪 Loss breakdown:", {k: v.item() for k, v in loss_dict.items()})
# #         losses = sum(loss_dict.values())

# #         optimizer.zero_grad()
# #         losses.backward()
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ✅ Gradient clipping
# #         optimizer.step()


# #         total_loss += losses.item()
# #         # print(f"Batch {batch_idx} - Loss: {losses.item():.4f}")

# #     return total_loss / len(data_loader)
# from tqdm import tqdm

# def train_step(model, criterion, data_loader, optimizer, device, epoch=None, rank=0):
#     model.train()
#     total_loss = 0

#     pbar = tqdm(
#         enumerate(data_loader), 
#         total=len(data_loader), 
#         disable=(rank != 0),
#         desc=f"Epoch {epoch}"
#     )

#     for batch_idx, (images, targets) in pbar:
#         if isinstance(images, tuple):
#             images = images[0]

#         images = images.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         outputs = model(images)
#         loss_dict = criterion(outputs, targets)
#         losses = sum(loss_dict.values())

#         optimizer.zero_grad()
#         losses.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()

#         total_loss += losses.item()

#         if rank == 0:
#             pbar.set_postfix({
#                 "Batch Loss": f"{losses.item():.4f}",
#                 "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
#             })

#     return total_loss / len(data_loader)


# def eval_step(model, criterion, data_loader, device, rank=0):
#     model.eval()
#     total_loss = 0

#     with torch.no_grad():
#         for images, targets in data_loader:
#             if isinstance(images, tuple):
#                 images = images[0]

#             images = images.to(device)
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#             outputs = model(images)
#             loss_dict = criterion(outputs, targets)
#             losses = sum(loss_dict.values())
#             total_loss += losses.item()

#     return total_loss / len(data_loader)

# from torchvision import transforms

# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor()
# # ])
# from torchvision import transforms

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])



# import os
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader
# from detection_data import DetectionDataset, collate_fn
# from torchvision import transforms
# import wandb
# import os
# import torch.distributed as dist

# # epochs = 100
# # batch_sizes = 100

# # def main():
# #     local_rank = int(os.environ["LOCAL_RANK"])
# #     dist.init_process_group(backend="nccl")
# #     torch.cuda.set_device(local_rank)
# #     device = torch.device("cuda", local_rank)

# #     # 🟡 Chỉ rank 0 khởi tạo wandb
# #     if local_rank == 0:
# #         wandb.init(
# #             project="detection-vitmae",
# #             name=f"run-vitmae-{wandb.util.generate_id()}",
# #             config={
# #                 "epochs": epochs,
# #                 "batch_size": batch_sizes,
# #                 "lr": 1e-4,
# #                 "num_classes": 15,
# #             }
# #         )

# #     # Transform
# #     transform = transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         transforms.ToTensor()
# #     ])

# #     # Load model
# #     NUM_CLASSES = 15
# #     model = ViTMAEDETR(num_classes=NUM_CLASSES, pretrained_mae=False)
# #     ckpt_path = "/home/datnvt/project/Medical_CLARA/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=033-valid/loss=0.02.ckpt"
# #     load_mae_weights(model, ckpt_path)
# #     model.to(device)
# #     model = DDP(model, device_ids=[local_rank])

# #     # Criterion
# #     matcher = HungarianMatcher()
# #     criterion = SetCriterion(num_classes=NUM_CLASSES, matcher=matcher).to(device)

# #     # Optimizer
# #     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# #     # Dataset + Sampler
# #     dataset = DetectionDataset(
# #         csv_file="/home/datnvt/project/Medical_CLARA/data_detection/xray_detection_combined.csv",
# #         img_root="/home/datnvt/project/Medical_CLARA/all_images",
# #         transform=transform
# #     )
# #     sampler = torch.utils.data.distributed.DistributedSampler(dataset)
# #     dataloader = DataLoader(dataset, batch_size=batch_sizes, sampler=sampler, collate_fn=collate_fn)

# #     for epoch in range(epochs):
# #         sampler.set_epoch(epoch)
# #         avg_loss = train_step(model, criterion, dataloader, optimizer, device, epoch=epoch, rank=local_rank)

# #         if local_rank == 0:
# #             print(f"\n✅ [Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
# #             wandb.log({
# #                 "train/loss_total": avg_loss,
# #                 "epoch": epoch
# #             })


# #         # Save model only on rank 0
# #     if local_rank == 0:
# #         save_path = "./outputs/detection_vitmae_final.pth"
# #         os.makedirs(os.path.dirname(save_path), exist_ok=True)

# #         # Lấy lại model gốc (bên trong DDP)
# #         torch.save(model.module.state_dict(), save_path)
# #         print(f"✅ Model saved to {save_path}")

# #     if local_rank == 0:
# #         wandb.finish()


# # if __name__ == "__main__":
# #     main()

# import os
# import torch
# import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import wandb
# from torchvision import transforms

# # Giả sử các hàm và lớp sau đã import:
# # - ViTMAEDETR
# # - load_mae_weights
# # - SetCriterion, HungarianMatcher
# # - DetectionDataset, collate_fn
# # - train_step, eval_step

# epochs = 200
# batch_sizes = 350

# def main():
#     local_rank = int(os.environ["LOCAL_RANK"])
#     dist.init_process_group(backend="nccl")
#     torch.cuda.set_device(local_rank)
#     device = torch.device("cuda", local_rank)

#     if local_rank == 0:
#         wandb.init(
#             project="detection-vitmae",
#             name=f"run-vitmae-{wandb.util.generate_id()}",
#             config={"epochs": epochs, "batch_size": batch_sizes, "lr": 5e-5, "num_classes": 15}
#         )

#     # Transform
#     # transform = transforms.Compose([
#     #     transforms.Resize((224, 224)),
#     #     transforms.ToTensor()
#     # ])
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1),
#         transforms.ToTensor()
#     ])
#     # Load + split CSV
#     csv_path = "/home/datnvt/project/Medical_CLARA/data_detection/xray_detection_combined.csv"
#     img_root = "/home/datnvt/project/Medical_CLARA/all_images"

#     df = pd.read_csv(csv_path)
#     filenames = df["filename"].unique()
#     train_fns, val_fns = train_test_split(filenames, test_size=0.2, random_state=42)

#     train_df = df[df["filename"].isin(train_fns)].reset_index(drop=True)
#     val_df = df[df["filename"].isin(val_fns)].reset_index(drop=True)

#     # Datasets
#     train_dataset = DetectionDataset(train_df, img_root, transform=transform)
#     val_dataset = DetectionDataset(val_df, img_root, transform=transform)

#     # train_dataset = XrayClassificationDataset(train_df, img_root, transform=transform)
#     # val_dataset = XrayClassificationDataset(val_df, img_root, transform=transform)

#     # Samplers
#     train_sampler = DistributedSampler(train_dataset)
#     val_sampler = DistributedSampler(val_dataset, shuffle=False)

#     # Loaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_sizes, sampler=train_sampler, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=batch_sizes, sampler=val_sampler, collate_fn=collate_fn)

#     # Load model
#     NUM_CLASSES = 15
#     model = ViTMAEDETR(num_classes=NUM_CLASSES, pretrained_mae=False)

#     ckpt_path = "/home/datnvt/project/Medical_CLARA/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=033-valid/loss=0.02.ckpt"
#     load_mae_weights(model, ckpt_path)
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             print(f"❄️ Frozen: {name}")
#         else:
#             print(f"🔥 Trainable: {name}")

#     # 🧊 Nếu bạn nghi encoder đang bị freeze, thì unfreeze toàn bộ tại đây:
#     # 🎯 Sau khi load weight, unfreeze toàn bộ trước khi DDP

#     # ✅ Force unfreeze toàn bộ encoder + head
#     for name, param in model.named_parameters():
#         param.requires_grad = True
#         print(f"🔥 Unfrozen: {name}")





#     model.to(device)
#     model = DDP(model, device_ids=[local_rank])

#     # Loss
#     matcher = HungarianMatcher()
#     criterion = SetCriterion(num_classes=NUM_CLASSES, matcher=matcher).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#     best_val_loss = float("inf")

#     for epoch in range(epochs):
#         train_sampler.set_epoch(epoch)

#         train_loss = train_step(model, criterion, train_loader, optimizer, device, epoch=epoch, rank=local_rank)
#         val_loss = eval_step(model, criterion, val_loader, device, rank=local_rank)

#         if local_rank == 0:
#             print(f"\n✅ Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#             wandb.log({
#                 "train/loss_total": train_loss,
#                 "val/loss_total": val_loss,
#                 "epoch": epoch
#             })

#             # ✅ SAVE every 5 epochs
#             if local_rank == 0:  # nếu dùng DDP
#                 print(f"🔁 Epoch {epoch}/{epochs}")
#                 if epoch % 5 == 0 or epoch == epochs - 1:
#                     save_path = f"/home/datnvt/project/Medical_CLARA/output_detection_v2/ckpt_epoch_{epoch:03}.pth"
#                     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#                     torch.save(model.module.state_dict(), save_path)
#                     print(f"📦 Saved checkpoint to {save_path}")
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     save_path = f"/home/datnvt/project/Medical_CLARA/output_detection_v2/best_model.pth"
#                     torch.save(model.module.state_dict(), save_path)
#                     print(f"🌟 New best model saved at Epoch {epoch} with Val Loss: {val_loss:.4f}")



#     # Save model
#     if local_rank == 0:
#         save_path = "/home/datnvt/project/Medical_CLARA/output_detection_v2/detection_vitmae_final.pth"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         torch.save(model.module.state_dict(), save_path)
#         print(f"✅ Model saved to {save_path}")
#         wandb.finish()
    
# if __name__ == "__main__":
#     main()
# full_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMAEModel
from scipy.optimize import linear_sum_assignment


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


from transformers import ViTMAEConfig, ViTMAEModel
import torch

class ViTMAEDETR(nn.Module):
    def __init__(self, mae_ckpt_path, num_classes=13, num_queries=100, hidden_dim=768):
        super().__init__()

        # Tạo config tương ứng
        config = ViTMAEConfig()
        self.encoder = ViTMAEModel(config)

        # Load .ckpt vào encoder
        state_dict = torch.load(mae_ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = {k.replace("encoder.", ""): v for k, v in state_dict["state_dict"].items() if "encoder." in k}
        self.encoder.load_state_dict(state_dict, strict=False)

        # Phần còn lại giữ nguyên
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLPHead(hidden_dim, hidden_dim, 4, num_layers=3)


    def forward(self, x):
        B = x.size(0)
        encoder_outputs = self.encoder(pixel_values=x)
        feat = encoder_outputs.last_hidden_state.permute(1, 0, 2)  # (N, B, C)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, feat).transpose(0, 1)  # (B, num_queries, C)

        pred_logits = self.class_embed(hs)
        pred_boxes = self.bbox_embed(hs).sigmoid()
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)  # (bs, queries, classes)
        out_bbox = outputs["pred_boxes"]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(center_to_corners_format(out_bbox[b]), center_to_corners_format(tgt_bbox))

            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            indices.append(linear_sum_assignment(C.cpu()))
        return indices


def center_to_corners_format(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    from torchvision.ops import generalized_box_iou as g_iou
    return g_iou(boxes1, boxes2)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef=0.1, losses=["labels", "boxes"]):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / target_boxes.size(0)
        loss_giou = 1 - torch.diag(generalized_box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))).mean()

        return {"loss_cls": loss_cls * 0.5, "loss_bbox": loss_bbox * 5.0, "loss_giou": loss_giou * 2.0}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(torch.tensor(src), i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([torch.tensor(src) for (src, _) in indices])
        return batch_idx, src_idx
