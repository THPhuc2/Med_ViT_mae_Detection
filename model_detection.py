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

def parse_args():
    parser = argparse.ArgumentParser(description="Train MAE Faster R-CNN")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    
    return parser.parse_args()

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    args = parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay

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

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_{now}"
    os.makedirs(save_dir, exist_ok=True)

    csv_path = "/home/datnvt/project/data/data_detection/xray_detection_combined.csv"
    image_dir = "/home/datnvt/project/data/all_images"
    # mae_ckpt = "/home/datnvt/project/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=033-valid/loss=0.02.ckpt"
    mae_ckpt = "/home/datnvt/project/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge_2/files/output_ptln/sample-epoch=060-valid/loss=0.02.ckpt"
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = DetectionDataset(csv_path, image_dir, transforms=resize_transform)
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    backbone = build_mae_backbone(mae_ckpt)

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    transform = GeneralizedRCNNTransform(
        min_size=224,
        max_size=224,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    model = FasterRCNN(
        backbone,
        num_classes=14,
        rpn_anchor_generator=anchor_generator,
        transform=transform
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float("inf")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    """
    code train MAE Faster R-CNN with validation and model saving
    train tốt nhưng loss chỉ giảm nhẹ, cần xem lại
    """
    # for epoch in range(num_epochs):
    #     # 🔵 TRAINING
    #     model.train()
    #     total_train_loss = 0.0
    #     progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]")

    #     for images, targets in progress_bar:
    #         images = list(img.to(device) for img in images)
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #         loss_dict = model(images, targets)
    #         loss = sum(loss for loss in loss_dict.values())

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_train_loss += loss.item()
    #         progress_bar.set_postfix({"Loss": loss.item()})

    #     avg_train_loss = total_train_loss / len(train_loader)
    #     print(f"✅ Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
    #     wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

    #     # 🔍 VALIDATION
    #     model.train()  # ⬅️ ADD THIS LINE to ensure model returns losses
    #     total_val_loss = 0.0
    #     with torch.no_grad():
    #         for images, targets in val_loader:
    #             images = list(img.to(device) for img in images)
    #             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #             loss_dict = model(images, targets)
    #             loss = sum(loss for loss in loss_dict.values())
    #             total_val_loss += loss.item()

    #     avg_val_loss = total_val_loss / len(val_loader)
    #     print(f"🔍 Validation Loss: {avg_val_loss:.4f}")
    #     wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})

    #     # 💾 Save best model
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    #         print(f"💾 Saved BEST model at epoch {epoch+1} with val loss {best_val_loss:.4f}")


    #     # 💾 Save every 10 epochs
    #     if (epoch + 1) % 10 == 0:
    #         torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))
    #         print(f"🕓 Saved checkpoint at epoch {epoch+1}")

    #         # mới thêm :))
    #         # 🚀 Push lên Hugging Face ngay sau khi save
    #         upload_folder(
    #             folder_path=save_dir,
    #             repo_id="THP2903/ViT_MAE_Huge_Detection",
    #             path_in_repo=f"checkpoint_epoch_{epoch+1}",
    #             repo_type="model"
    #         )
    #         print(f"📤 Uploaded checkpoint {ckpt_name} to Hugging Face")

    #         # ❌ Xóa checkpoint local sau khi push
    #         os.remove(ckpt_path)
    #         print(f"🧹 Removed local {ckpt_name}")
    #         #  mới thêm   :))

    # # 💾 Save last model
    # # torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

    # # mới thêm :))
    # # 💾 Save last model
    # last_model_path = os.path.join(save_dir, "last_model.pth")
    # torch.save(model.state_dict(), last_model_path)
    # print(f"🎉 Training finished. Saved {last_model_path}")

    # # 🚀 Push last_model lên HF
    # upload_folder(
    #     folder_path=save_dir,
    #     repo_id="THP2903/ViT_MAE_Huge_Detection",
    #     path_in_repo="last_model",
    #     repo_type="model"
    # )

    # # 🧹 Cleanup: giữ lại best + last, xóa file linh tinh khác
    # for f in os.listdir(save_dir):
    #     if f not in ["best_model.pth", "last_model.pth"]:
    #         os.remove(os.path.join(save_dir, f))



    # print(f"🎉 Finished training. Final model saved to {save_dir}/last_model.pth")
    # wandb.finish()



    """
    cod đúng là ra bouding box nhưng loss ko giảm
    
    """

    for epoch in range(num_epochs):
        # 🔵 TRAINING
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

        # 🔍 VALIDATION
        model.train()  # ⬅️ ADD THIS LINE to ensure model returns losses
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

        # 💾 Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"💾 Saved BEST model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

        # 💾 Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))
            print(f"🕓 Saved checkpoint at epoch {epoch+1}")

    # 💾 Save last model
    torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
    print(f"🎉 Finished training. Final model saved to {save_dir}/last_model.pth")
    wandb.finish()

    """
    cod dom 
    """


    # for epoch in range(num_epochs):
    #     model.train()
    #     total_train_loss = 0.0
    #     progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]")

    #     for images, targets in progress_bar:
    #         images = list(img.to(device) for img in images)
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #         loss_dict = model(images, targets)
    #         loss = sum(loss for loss in loss_dict.values())

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_train_loss += loss.item()
    #         progress_bar.set_postfix({"Loss": loss.item()})

    #     avg_train_loss = total_train_loss / len(train_loader)
    #     print(f"✅ Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
    #     wandb.log({
    #         "epoch": epoch + 1,
    #         "train_loss": avg_train_loss
    #     })

    #     model.train()
    #     total_val_loss = 0.0
    #     with torch.no_grad():
    #         for images, targets in val_loader:
    #             images = list(img.to(device) for img in images)
    #             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #             loss_dict = model(images, targets)
    #             loss = sum(loss for loss in loss_dict.values())
    #             total_val_loss += loss.item()

    #     avg_val_loss = total_val_loss / len(val_loader)
    #     print(f"🔍 Validation Loss: {avg_val_loss:.4f}")
    #     wandb.log({
    #         "epoch": epoch + 1,
    #         "val_loss": avg_val_loss
    #     })

    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    #         print(f"💾 Saved BEST model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    # torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
    # print(f"🎉 Finished training. Final model saved to {save_dir}/last_model.pth")
    # wandb.finish()

    """
    đã train thư nhưng loss giam nhưng độ chính xác ko cao
    """
    # for epoch in range(num_epochs):
    # # 🔵 TRAINING
    #     model.train()
    #     total_train_loss = 0.0
    #     progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]")

    #     for images, targets in progress_bar:
    #         images = [img.to(device) for img in images]
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #         loss_dict = model(images, targets)
    #         loss = sum(loss for loss in loss_dict.values())

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_train_loss += loss.item()
    #         progress_bar.set_postfix({"Loss": loss.item()})

    #     avg_train_loss = total_train_loss / len(train_loader)
    #     print(f"✅ Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
    #     wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

    #     # 🔍 VALIDATION
    #     model.train()  # For FasterRCNN to return losses even in eval mode
    #     total_val_loss = 0.0
    #     with torch.no_grad():
    #         for images, targets in val_loader:
    #             images = [img.to(device) for img in images]
    #             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #             loss_dict = model(images, targets)
    #             loss = sum(loss for loss in loss_dict.values())
    #             total_val_loss += loss.item()

    #     avg_val_loss = total_val_loss / len(val_loader)
    #     print(f"🔍 Validation Loss: {avg_val_loss:.4f}")
    #     wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})

    #     # 🔄 STEP SCHEDULER
    #     scheduler.step(avg_val_loss)
    #     for i, param_group in enumerate(optimizer.param_groups):
    #         current_lr = param_group['lr']
    #         print(f"📉 [LR] Epoch {epoch+1} - Group {i}: LR = {current_lr}")
    #         wandb.log({f"lr_group_{i}": current_lr, "epoch": epoch + 1})

    #     # 💾 SAVE BEST MODEL
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    #         print(f"💾 Saved BEST model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    #     # 💾 SAVE CHECKPOINT EACH 10 EPOCHS
    #     if (epoch + 1) % 10 == 0:
    #         torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))
    #         print(f"🕓 Saved checkpoint at epoch {epoch+1}")

    # # 💾 SAVE FINAL MODEL
    # torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
    # print(f"🎉 Finished training. Final model saved to {save_dir}/last_model.pth")
    # wandb.finish()


    # for epoch in range(num_epochs):
    #     # 🔵 TRAINING
    #     model.train()
    #     total_train_loss = 0.0
    #     progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]")

    #     for images, targets in progress_bar:
    #         images = [img.to(device) for img in images]
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #         loss_dict = model(images, targets)
    #         loss = sum(loss for loss in loss_dict.values())

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_train_loss += loss.item()
    #         progress_bar.set_postfix({"Loss": loss.item()})

    #         # 🔹 Log từng loss thành phần
    #         log_data = {f"train/{k}": v.item() for k, v in loss_dict.items()}
    #         log_data.update({"train/total_loss": loss.item(), "epoch": epoch + 1})
    #         wandb.log(log_data)

    #     avg_train_loss = total_train_loss / len(train_loader)
    #     print(f"✅ Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    #     # 🔍 VALIDATION
    #     model.train()  # FasterRCNN cần train mode để trả loss
    #     total_val_loss = 0.0
    #     with torch.no_grad():
    #         for images, targets in val_loader:
    #             images = [img.to(device) for img in images]
    #             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #             loss_dict = model(images, targets)
    #             loss = sum(loss for loss in loss_dict.values())
    #             total_val_loss += loss.item()

    #             # 🔹 Log val loss thành phần
    #             log_data = {f"val/{k}": v.item() for k, v in loss_dict.items()}
    #             log_data.update({"val/total_loss": loss.item(), "epoch": epoch + 1})
    #             wandb.log(log_data)

    #     avg_val_loss = total_val_loss / len(val_loader)
    #     print(f"🔍 Validation Loss: {avg_val_loss:.4f}")

    #     # 🔄 Scheduler
    #     scheduler.step(avg_val_loss)

    #     # 💾 Save best model
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    #         print(f"💾 Saved BEST model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
