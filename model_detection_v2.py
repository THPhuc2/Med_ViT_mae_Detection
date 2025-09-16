
import os
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
from datetime import datetime
from detection_data import DetectionDataset
from mae_backbone import build_mae_backbone
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hugging Face helpers
from huggingface_hub import HfFolder, HfApi, REMOVEDhub_download

def parse_args():
    parser = argparse.ArgumentParser(description="Train MAE Faster R-CNN")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run name for HF folder (e.g. train_101)")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with all images")
    parser.add_argument("--mae_ckpt", type=str, required=True, help="Path to pretrained MAE .ckpt")
    parser.add_argument("--base_save_dir", type=str,
                        default="/home/datnvt/project/mae/checkpoints_detection_2",
                        help="Base directory to save checkpoints")
    return parser.parse_args()

def collate_fn(batch):
    return tuple(zip(*batch))

def save_state_cpu(model, path):
    # save model.state_dict with tensors moved to cpu to avoid CUDA tensors in file
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(sd, path)

from torchvision.ops import box_iou

def evaluate(model, data_loader, device):
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                if len(output["boxes"]) == 0 or len(target["boxes"]) == 0:
                    continue
                iou = box_iou(output["boxes"].cpu(), target["boxes"].cpu())
                iou_scores.append(iou.max().item())
    return sum(iou_scores)/len(iou_scores) if iou_scores else 0.0


def main():
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay

    # ---------- WandB ----------
    wandb.login(key="d8dbd91c9717ac3a104742d8f247ae4012526297")
    wandb.init(
        project="mae-detection",
        name=f"run-e{num_epochs}_bs{batch_size}_lr{lr}",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": "FasterRCNN",
            "backbone": "MAE-Huge",
            "image_size": 224,
            "num_classes": 14
        }
    )

    # ---------- Setup run folder + HF ----------
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"train_{num_epochs}_{now}"
    base_save_dir = args.base_save_dir
    local_run_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(local_run_dir, exist_ok=True)

    # HF settings
    HfFolder.save_token("REMOVED") 
    repo_id = "THP2903/ViT_MAE_Huge_Detection"
    api = HfApi()  

    # ---------- Data + model ----------
    # csv_path = "/home/datnvt/project/data/data_detection/xray_detection_combined.csv"
    # image_dir = "/home/datnvt/project/data/all_images"
    # mae_ckpt = "/home/datnvt/project/mae/outputs_rand_.../sample-epoch=060-valid/loss=0.02.ckpt"
    csv_path = args.csv_path
    image_dir = args.image_dir
    mae_ckpt = args.mae_ckpt
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # resize_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomHorizontalFlip(0.5),   # augment
    #     ])

    full_dataset = DetectionDataset(csv_path, image_dir, transforms=resize_transform)
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    backbone = build_mae_backbone(mae_ckpt)
    anchor_generator = AnchorGenerator(sizes=((32,64,128,256,512),), aspect_ratios=((0.5,1.0,2.0),))
    transform = GeneralizedRCNNTransform(min_size=224, max_size=224,
                                        image_mean=[0.485,0.456,0.406], image_std=[0.229,0.224,0.225])
    model = FasterRCNN(backbone, num_classes=14, rpn_anchor_generator=anchor_generator, transform=transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(
    #                             model.parameters(),
    #                             lr=0.005, momentum=0.9, weight_decay=0.0005
    #                         )
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # from torch.optim.lr_scheduler import StepLR
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    best_val_loss = float("inf")
    best_epoch = -1

    # ---------- Training loop with upload-on-save ----------
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]")
        for images, targets in progress_bar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # Validation (lưu ý model.train() để model trả về loss với targets)
        model.train()
        # model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"🔍 Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})

        # 
        model.eval()
        val_iou = evaluate(model, val_loader, device)
        print(f"📊 Val IoU: {val_iou:.4f}")
        wandb.log({"epoch": epoch+1, "val_iou": val_iou})

        # scheduler.step(avg_val_loss)

        # Save best model locally (giữ lại)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(local_run_dir, "best_model.pth")
            save_state_cpu(model, best_model_path)
            print(f"💾 Saved BEST model at epoch {best_epoch} (val {best_val_loss:.4f})")

            # cũng upload best_model ngay (không xóa local)
            try:
                api.upload_file(
                    path_or_fileobj=best_model_path,
                    path_in_repo=f"{run_name}/best_model.pth",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print("📤 Uploaded best_model.pth to HF")
            except Exception as e:
                print("⚠️ Upload best_model failed:", e)

        # Save periodic checkpoint every 10 epochs -> upload -> delete local checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_name = f"model_epoch_{epoch+1}.pth"
            ckpt_path = os.path.join(local_run_dir, ckpt_name)
            save_state_cpu(model, ckpt_path)
            print(f"🕓 Saved checkpoint {ckpt_name} locally")

            # Upload to HF under folder run_name/
            try:
                api.upload_file(
                    path_or_fileobj=ckpt_path,
                    path_in_repo=f"{run_name}/{ckpt_name}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"📤 Uploaded {ckpt_name} to HF at {run_name}/")
                # xóa local sau khi upload thành công
                os.remove(ckpt_path)
                print(f"🧹 Removed local checkpoint {ckpt_name}")
            except Exception as e:
                print("⚠️ Upload (or delete) failed for", ckpt_name, ":", e)
                # Nếu upload fail, giữ lại local checkpoint để debug

    # Sau khi train xong -> save last model, upload last + cleanup nhưng giữ best + last local
    last_model_path = os.path.join(local_run_dir, "last_model.pth")
    save_state_cpu(model, last_model_path)
    print(f"🎉 Saved last_model.pth locally")

    try:
        api.upload_file(
            path_or_fileobj=last_model_path,
            path_in_repo=f"{run_name}/last_model.pth",
            repo_id=repo_id,
            repo_type="model"
        )
        print("📤 Uploaded last_model.pth to HF")
    except Exception as e:
        print("⚠️ Upload last_model failed:", e)

    # Clean up local folder: giữ lại chỉ best_model.pth và last_model.pth
    for f in os.listdir(local_run_dir):
        if f not in ["best_model.pth", "last_model.pth"]:
            try:
                os.remove(os.path.join(local_run_dir, f))
            except Exception:
                pass
    print(f"🧹 Local cleanup done. Kept best_model.pth (epoch {best_epoch}) and last_model.pth")

    wandb.finish()

if __name__ == "__main__":
    main()
