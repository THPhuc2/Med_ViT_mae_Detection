# import os

# root_dir = "/home/datnvt/project/Medical_CLARA/data_mae"  # đường dẫn chứa các folder từ 1 đến 13

# for i in range(1, 14):  # từ 1 đến 13
#     folder_path = os.path.join(root_dir, str(i))
    
#     if os.path.exists(folder_path):
#         # Đếm số file là ảnh
#         num_images = len([
#             f for f in os.listdir(folder_path)
#             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
#         ])
#         print(f"Folder {i}: {num_images} images")
#     else:
#         print(f"Folder {i} does not exist")
import pandas as pd

def filter_out_class_14(input_csv, output_csv):
    # Đọc dữ liệu
    df = pd.read_csv(input_csv)

    # Bỏ các box không có tọa độ (NaN box)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])

    # Bỏ box class 14
    df_no_14 = df[df["class_id"] != 14]

    # Tìm những ảnh vẫn còn ít nhất 1 box sau khi bỏ 14
    valid_filenames = df_no_14["filename"].unique()

    # Lọc lại toàn bộ df_no_14 chỉ chứa ảnh có ít nhất 1 box (sau khi bỏ 14)
    df_filtered = df_no_14[df_no_14["filename"].isin(valid_filenames)]

    # Ghi ra file mới
    df_filtered.to_csv(output_csv, index=False)
    print(f"✅ Đã lưu file sau khi lọc class 14: {output_csv}")

# Ví dụ dùng:
filter_out_class_14("/home/datnvt/project/Medical_CLARA/data_detection/xray_detection_combined.csv", "/home/datnvt/project/Medical_CLARA/data_detection/xray_filtered_data_no14.csv")
