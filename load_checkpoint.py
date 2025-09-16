from huggingface_hub import HfApi, HfFolder, upload_folder
import os

# Đăng nhập (hoặc paste token nếu chưa login)
HfFolder.save_token("REMOVED")

# Thông tin repo
repo_id = "THP2903/ViT_MAE_Huge_Detection"  # thay bằng repo của bạn
repo_type = "model"

# Đường dẫn local
good_ckpt_dir = "/home/datnvt/project/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge_2/files/output_ptln"
# bad_ckpt_dir = "/home/datnvt/project/mae/checkpoints_detection_2/fasterrcnn_20250720_040426-check-point-bad"

# Upload folder good
upload_folder(
    folder_path=good_ckpt_dir,
    repo_id=repo_id,
    path_in_repo="mae",
    repo_type=repo_type
)

# # Upload folder bad
# upload_folder(
#     folder_path=bad_ckpt_dir,
#     repo_id=repo_id,
#     path_in_repo="bad_version_1",
#     repo_type=repo_type
# )


"""
lấy check point từ repo

"""

# from huggingface_hub import REMOVEDhub_download

# # Load checkpoint epoch 120 trong folder good
# ckpt_path = REMOVEDhub_download(
#     repo_id="your-username/mae-fasterrcnn-checkpoints",
#     filename="checkpoint_epoch_120.pth",
#     subfolder="good"
# )
