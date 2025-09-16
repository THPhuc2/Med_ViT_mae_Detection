# import os

# # Thư mục chứa toàn bộ dữ liệu
# data_folder = "/home/datnvt/project/data/data_mae/x_ray_8bit/data_mae"

# # Lấy danh sách folder ảnh (1, 2, ..., 13)
# image_folders = sorted([f for f in os.listdir(data_folder) if f.isdigit()])

# # Xử lý đổi tên file trong thư mục ảnh
# for folder_name in image_folders:
#     folder_path = os.path.join(data_folder, folder_name)
    
#     if os.path.isdir(folder_path):
#         image_files = sorted([f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))])

#         for file in image_files:
#             old_path = os.path.join(folder_path, file)
#             new_name = f"{folder_name}_{file}"  # Thêm số folder vào trước tên file
#             new_path = os.path.join(folder_path, new_name)

#             os.rename(old_path, new_path)

# # Xử lý đổi tên file trong thư mục mask (mask_1, mask_2, ...)
# mask_folders = sorted([f for f in os.listdir(data_folder) if f.startswith("mask_")])

# for mask_folder in mask_folders:
#     mask_folder_path = os.path.join(data_folder, mask_folder)
#     folder_number = mask_folder.split("_")[-1]  # Lấy số cuối cùng (mask_1 → 1)

#     if os.path.isdir(mask_folder_path):
#         mask_files = sorted([f for f in os.listdir(mask_folder_path) if f.endswith((".png", ".jpg", ".jpeg"))])

#         for file in mask_files:
#             old_path = os.path.join(mask_folder_path, file)
#             new_name = f"{folder_number}_{file}"  # Thêm số folder vào trước tên file
#             new_path = os.path.join(mask_folder_path, new_name)

#             os.rename(old_path, new_path)

# print("✅ Đã đổi tên ảnh và mask thành công!")
import os

# ✅ Thư mục chứa dữ liệu bị rename sai
folder = "/home/datnvt/project/data/data_mae/x_ray_8bit/data_mae"

# ✅ Lặp qua tất cả các folder con (1, 2, 3, ...)
for subfolder in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            old_path = os.path.join(subfolder_path, filename)

            # Kiểm tra nếu tên file bắt đầu bằng lặp tiền tố, ví dụ: "1_1_", "2_2_"
            if filename.startswith(f"{subfolder}_{subfolder}_"):
                new_filename = filename.replace(f"{subfolder}_{subfolder}_", f"{subfolder}_", 1)
                new_path = os.path.join(subfolder_path, new_filename)

                print(f"🔄 Đổi tên: {filename} → {new_filename}")
                os.rename(old_path, new_path)

print("✅ Đã khôi phục tên đúng cho ảnh!")
