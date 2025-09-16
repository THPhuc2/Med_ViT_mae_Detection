# import torch
# from torchvision import transforms
# from PIL import Image, ImageDraw, ImageFont
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.transform import GeneralizedRCNNTransform
# from mae_backbone import build_mae_backbone
# import os
# import matplotlib.pyplot as plt

# # ==== CONFIG ====
# model_ckpt = "/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_20250718_082014/best_model.pth"
# mae_ckpt = "/home/datnvt/project/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=033-valid/loss=0.02.ckpt"
# image_path = "/home/datnvt/project/Medical_CLARA/infer/demo_clara/image_test/test4.png"
# output_path = "/home/datnvt/project/mae/output/test4_detected.jpg"
# feature_output_path = "/home/datnvt/project/mae/output/test4_featuremap.jpg"

# CLASS_NAMES = [
#     "Phình động mạch chủ", "Xẹp phổi", "Vôi hóa", "Tim to", "Đông đặc phổi",
#     "Bệnh phổi kẽ", "Thâm nhiễm", "Mờ phổi", "Nốt/Khối u", "Tổn thương khác",
#     "Tràn dịch màng phổi", "Dày màng phổi", "Tràn khí màng phổi", "Xơ phổi"
# ]

# # ==== LOAD IMAGE ====
# image = Image.open(image_path).convert("RGB")
# orig_w, orig_h = image.size

# # Resize for model input (keep original for visualization)
# resize_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
# image_tensor = resize_transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# # ==== BUILD BACKBONE AND SAVE FEATURE MAP ====
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# backbone = build_mae_backbone(mae_ckpt).to(device)

# # Get encoder feature map
# with torch.no_grad():
#     feature_map = backbone(image_tensor.to(device))  # Expect shape [1, C, H, W]

# # ==== SAVE FEATURE MAP AS HEATMAP ====
# def save_feature_map_as_heatmap(feature_map, save_path, index=0):
#     fmap = feature_map[0]  # [C, H, W]
#     fmap_img = fmap.mean(0).cpu().numpy()  # [H, W]
#     plt.imshow(fmap_img, cmap="viridis")
#     plt.axis("off")
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
#     plt.close()
#     print(f"🧠 Saved feature map to {save_path}")

# save_feature_map_as_heatmap(feature_map, feature_output_path)

# # ==== BUILD DETECTION MODEL ====
# anchor_generator = AnchorGenerator(
#     sizes=((32, 64, 128, 256, 512),),
#     aspect_ratios=((0.5, 1.0, 2.0),)
# )

# transform_model = GeneralizedRCNNTransform(
#     min_size=224,
#     max_size=224,
#     image_mean=[0.485, 0.456, 0.406],
#     image_std=[0.229, 0.224, 0.225],
# )

# model = FasterRCNN(
#     backbone=backbone,
#     num_classes=14,
#     rpn_anchor_generator=anchor_generator,
#     transform=transform_model
# )

# model.load_state_dict(torch.load(model_ckpt, map_location=device))
# model.to(device)
# model.eval()

# # ==== INFERENCE ====
# with torch.no_grad():
#     outputs = model([image_tensor.squeeze(0).to(device)])
#     boxes = outputs[0]["boxes"]
#     scores = outputs[0]["scores"]
#     labels = outputs[0]["labels"]

# # ==== FILTER RESULTS ====
# threshold = 0.1
# keep = scores >= threshold
# boxes = boxes[keep].cpu()
# scores = scores[keep].cpu()
# labels = labels[keep].cpu()

# # ==== RESCALE BOXES TO ORIGINAL SIZE ====
# def rescale_boxes(boxes, from_size, to_size):
#     from_w, from_h = from_size
#     to_w, to_h = to_size
#     scale_x = to_w / from_w
#     scale_y = to_h / from_h
#     return [
#         [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
#         for x1, y1, x2, y2 in boxes.tolist()
#     ]

# rescaled_boxes = rescale_boxes(boxes, (224, 224), (orig_w, orig_h))

# # ==== DRAW AND SAVE RESULTS ====
# def save_bboxes(orig_image, boxes, labels, scores, save_path):
#     draw = ImageDraw.Draw(orig_image)
#     try:
#         font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=24)
#     except:
#         font = ImageFont.load_default(size=100)

#     for box, label, score in zip(boxes, labels, scores):
#         draw.rectangle(box, outline="red", width=5)
#         label_text = f"{CLASS_NAMES[label]}: {score:.2f}"
#         draw.text((box[0], max(0, box[1] - 28)), label_text, fill="yellow", font=font)

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     orig_image.save(save_path)
#     print(f"💾 Saved detection result to {save_path}")

# save_bboxes(image.copy(), rescaled_boxes, labels, scores, output_path)



# import torch
# from torchvision import transforms
# from PIL import Image, ImageDraw, ImageFont
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.transform import GeneralizedRCNNTransform
# from torchvision.utils import save_image
# from torchvision.transforms.functional import to_pil_image
# import os

# from mae_backbone import build_mae_backbone  # Ensure this is available in your path

# # ==== CONFIG ====
# model_ckpt = "/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_20250718_082014/best_model.pth"
# mae_ckpt = "/home/datnvt/project/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=033-valid/loss=0.02.ckpt"
# image_path = "/home/datnvt/project/Medical_CLARA/infer/demo_clara/image_test/test4.png"
# output_dir = "/home/datnvt/project/mae/output/debug_pipeline"
# os.makedirs(output_dir, exist_ok=True)

# CLASS_NAMES = [
#     "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
#     "Consolidation", "Interstitial Lung Disease", "Infiltration", "Lung Opacity", "Nodule/Mass",
#     "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
#     "Pulmonary fibrosis"
# ]

# # ==== LOAD IMAGE ====
# image = Image.open(image_path).convert("RGB")
# orig_w, orig_h = image.size
# image.save(os.path.join(output_dir, "step1_original_input.jpg"))

# # Resize for model input (keep original for visualization)
# resize_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
# image_tensor = resize_transform(image)
# to_pil_image(image_tensor).save(os.path.join(output_dir, "step2_resized_224x224.jpg"))

# # ==== BUILD MODEL ====
# backbone = build_mae_backbone(mae_ckpt)

# # Extract and save MAE feature map (step 3)
# with torch.no_grad():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     backbone.to(device)
#     image_tensor_batch = image_tensor.unsqueeze(0).to(device)
#     feat_dict = backbone(image_tensor_batch)
#     feature_map = feat_dict[0][0]  # shape (C, H, W)
#     vis_feat = feature_map[:3]
#     vis_feat = (vis_feat - vis_feat.min()) / (vis_feat.max() - vis_feat.min())
#     save_image(vis_feat, os.path.join(output_dir, "step3_feature_map.jpg"))

# # Detection model
# anchor_generator = AnchorGenerator(
#     sizes=((32, 64, 128, 256, 512),),
#     aspect_ratios=((0.5, 1.0, 2.0),)
# )

# transform_model = GeneralizedRCNNTransform(
#     min_size=224,
#     max_size=224,
#     image_mean=[0.485, 0.456, 0.406],
#     image_std=[0.229, 0.224, 0.225],
# )

# model = FasterRCNN(
#     backbone=backbone,
#     num_classes=14,
#     rpn_anchor_generator=anchor_generator,
#     transform=transform_model
# )
# model.load_state_dict(torch.load(model_ckpt, map_location=device))
# model.to(device)
# model.eval()

# # ==== INFERENCE ====
# with torch.no_grad():
#     outputs = model([image_tensor.to(device)])
#     boxes = outputs[0]["boxes"]
#     scores = outputs[0]["scores"]
#     labels = outputs[0]["labels"]

# # ==== FILTER RESULTS ====
# threshold = 0.2
# keep = scores >= threshold
# boxes = boxes[keep].cpu()
# scores = scores[keep].cpu()
# labels = labels[keep].cpu()

# # ==== RESCALE BOXES TO ORIGINAL IMAGE ====
# def rescale_boxes(boxes, from_size, to_size):
#     from_w, from_h = from_size
#     to_w, to_h = to_size
#     scale_x = to_w / from_w
#     scale_y = to_h / from_h
#     return [
#         [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
#         for x1, y1, x2, y2 in boxes.tolist()
#     ]

# rescaled_boxes = rescale_boxes(boxes, (224, 224), (orig_w, orig_h))

# # ==== DRAW AND SAVE FINAL OUTPUT ====
# def save_bboxes(orig_image, boxes, labels, scores, save_path):
#     draw = ImageDraw.Draw(orig_image)
#     try:
#         font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=30)
#     except:
#         font = ImageFont.load_default(size=100)

#     for box, label, score in zip(boxes, labels, scores):
#         draw.rectangle(box, outline="red", width=5)
#         label_text = f"{CLASS_NAMES[label]}: {score:.2f}"
#         draw.text((box[0], max(0, box[1] - 30)), label_text, fill="yellow", font=font)

#     orig_image.save(save_path)
#     print(f"✅ Saved final detection result to: {save_path}")

# save_bboxes(image.copy(), rescaled_boxes, labels, scores, os.path.join(output_dir, "step4_output_with_boxes.jpg"))



import torch
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from mae_backbone import build_mae_backbone
import matplotlib.pyplot as plt
import os

# ==== CONFIG ====
# model_ckpt = "/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_20250718_082014/best_model.pth"
# model_ckpt = "/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_20250718_082014/model_epoch_40.pth"
# model_ckpt = "/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_20250720_040426/best_model.pth"
model_ckpt = "/home/datnvt/project/mae/checkpoints_detection_2/detection_experiment_1/last_model.pth"
mae_ckpt = "/home/datnvt/project/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge_2/files/output_ptln/sample-epoch=060-valid/loss=0.02.ckpt"
image_path = "/home/datnvt/project/Medical_CLARA/infer/demo_clara/image_test/test4.png"
output_path = "/home/datnvt/project/mae/output/infer_result_test4_15_09.jpg"

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "Interstitial Lung Disease", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis"
]

# ==== LOAD IMAGE ====
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size

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
threshold = 0.1
keep = scores >= threshold
boxes = boxes[keep].cpu()
scores = scores[keep].cpu()
labels = labels[keep].cpu()

# ==== NMS PER CLASS ====
def nms_per_class(boxes, scores, labels, iou_threshold=0.3):
    keep_indices = []
    unique_classes = labels.unique()
    for cls in unique_classes:
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = torch.where(cls_mask)[0]

        kept = nms(cls_boxes, cls_scores, iou_threshold)
        keep_indices.extend(cls_indices[kept].tolist())
    return keep_indices

keep_nms = nms_per_class(boxes, scores, labels, iou_threshold=0.3)
boxes = boxes[keep_nms]
scores = scores[keep_nms]
labels = labels[keep_nms]

print("📦 Boxes after threshold + NMS:", boxes)
print("🔖 Labels after NMS:", labels)

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
from PIL import ImageFont
# ==== DRAW AND SAVE ====
def save_bboxes(orig_image, boxes, labels, scores, save_path):
    draw = ImageDraw.Draw(orig_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=40)
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        draw.rectangle(box, outline="red", width=4)
        label_text = f"{CLASS_NAMES[label]}: {score:.2f}"
        draw.text((box[0], max(0, box[1] - 28)), label_text, fill="yellow", font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    orig_image.save(save_path)
    print(f"💾 Saved to {save_path}")

save_bboxes(image.copy(), rescaled_boxes, labels, scores, output_path)
