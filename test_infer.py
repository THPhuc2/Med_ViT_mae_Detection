import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from mae_backbone import build_mae_backbone
import os



# ==== CONFIG ====
# model_ckpt = "/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_20250717_102349/best_model.pth"
# model_ckpt = "/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_20250718_082014/best_model.pth"
model_ckpt = "/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_20250718_082014/model_epoch_40.pth"
mae_ckpt = "/home/datnvt/project/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=033-valid/loss=0.02.ckpt"
# mae_ckpt = "/home/datnvt/project/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=032-valid/loss=0.02.ckpt"
image_path = "/home/datnvt/project/Medical_CLARA/infer/demo_clara/image_test/test4.png"
output_path = "/home/datnvt/project/mae/output/infer_result_test4.jpg"

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis"
]
# CLASS_NAMES = [
#     "Phình động mạch chủ",        # 0 - Aortic enlargement
#     "Xẹp phổi",                   # 1 - Atelectasis
#     "Vôi hóa",                    # 2 - Calcification
#     "Tim to",                     # 3 - Cardiomegaly
#     "Đông đặc phổi",              # 4 - Consolidation
#     "Bệnh phổi kẽ",               # 5 - ILD (Interstitial Lung Disease)
#     "Thâm nhiễm",                 # 6 - Infiltration
#     "Mờ phổi",                    # 7 - Lung Opacity
#     "Nốt/Khối u",                 # 8 - Nodule/Mass
#     "Tổn thương khác",            # 9 - Other lesion
#     "Tràn dịch màng phổi",       # 10 - Pleural effusion
#     "Dày màng phổi",             # 11 - Pleural thickening
#     "Tràn khí màng phổi",        # 12 - Pneumothorax
#     "Xơ phổi"                     # 13 - Pulmonary fibrosis
# ]


# ==== LOAD IMAGE ====
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size

# Resize for model input (keep original for visualization)
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image_tensor = resize_transform(image)

# ==== BUILD MODEL ====
backbone = build_mae_backbone(mae_ckpt)

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

transform_model = GeneralizedRCNNTransform(
    min_size=224,
    max_size=224,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)

model = FasterRCNN(
    backbone=backbone,
    num_classes=14,
    rpn_anchor_generator=anchor_generator,
    transform=transform_model
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model.to(device)
model.eval()

# ==== INFERENCE ====
with torch.no_grad():
    outputs = model([image_tensor.to(device)])
    boxes = outputs[0]["boxes"]
    scores = outputs[0]["scores"]
    labels = outputs[0]["labels"]

# ==== FILTER RESULTS ====
threshold = 0.4
keep = scores >= threshold
boxes = boxes[keep].cpu()
scores = scores[keep].cpu()
labels = labels[keep].cpu()

print("📦 Boxes after thresholding:", boxes)
print("🔖 Labels after thresholding:", labels)
import matplotlib.pyplot as plt

def save_feature_map_as_heatmap(feature_map, save_path, index=0):
    fmap = feature_map[0]  # shape [C, H, W]
    fmap_img = fmap[index].detach().cpu().numpy()
    plt.imshow(fmap_img, cmap="viridis")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"🧠 Saved feature map to {save_path}")

# ==== RESCALE BOXES TO ORIGINAL IMAGE ====
def rescale_boxes(boxes, from_size, to_size):
    from_w, from_h = from_size
    to_w, to_h = to_size
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    return [
        [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        for x1, y1, x2, y2 in boxes.tolist()
    ]

rescaled_boxes = rescale_boxes(boxes, (224, 224), (orig_w, orig_h))
print("📏 Rescaled boxes:", rescaled_boxes)

# ==== DRAW AND SAVE ====
# ==== DRAW AND SAVE ====
def save_bboxes(orig_image, boxes, labels, scores, save_path):
    draw = ImageDraw.Draw(orig_image)

    # Sử dụng font chữ lớn hơn
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=1000)
    except:
        font = ImageFont.load_default(size=100)

    for box, label, score in zip(boxes, labels, scores):
        draw.rectangle(box, outline="red", width=5)
        label_text = f"{CLASS_NAMES[label]}: {score:.2f}"
        draw.text((box[0], max(0, box[1] - 28)), label_text, fill="yellow", font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    orig_image.save(save_path)
    print(f"💾 Saved to {save_path}")


save_bboxes(image.copy(), rescaled_boxes, labels, scores, output_path)
