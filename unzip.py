# import os
# import zipfile
# import re
# import sys

# def unzip_files(input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)

#     for file_name in os.listdir(input_folder):
#         file_path = os.path.join(input_folder, file_name)

#         if file_name.endswith(".zip") and os.path.isfile(file_path):
    
#             match = re.match(r"^(mask_)?(\d+)\.zip$", file_name)
#             if match:
#                 folder_name = match.group(0).replace(".zip", "")  # Tạo tên thư mục giống tên file
#                 dest_folder = os.path.join(output_folder, folder_name)
#                 os.makedirs(dest_folder, exist_ok=True)

#                 try:
#                     with zipfile.ZipFile(file_path, 'r') as zip_ref:
#                         zip_ref.extractall(dest_folder)
#                         print(f"Đã giải nén: {file_name} → {dest_folder}")
#                 except zipfile.BadZipFile:
#                     print(f"Lỗi: {file_name}")

#     print("done")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Sai cú pháp! Cách dùng:")
#         print("python unzip_all.py <input_folder> <output_folder>")
#         sys.exit(1)

#     input_dir = sys.argv[1]
#     output_dir = sys.argv[2]

#     unzip_files(input_dir, output_dir)

import os
import zipfile
import sys

def unzip_all_files(input_folder, output_folder):
    """ Giải nén tất cả file ZIP trong thư mục đầu vào vào thư mục đầu ra """
    if not os.path.isdir(input_folder):
        print(f"❌ Lỗi: {input_folder} không phải là thư mục hợp lệ!")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Duyệt qua tất cả các file trong thư mục đầu vào
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if file_name.endswith(".zip") and os.path.isfile(file_path):
            dest_folder = os.path.join(output_folder, file_name.replace(".zip", ""))
            os.makedirs(dest_folder, exist_ok=True)

            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_folder)
                    print(f"✅ Đã giải nén: {file_name} → {dest_folder}")
            except zipfile.BadZipFile:
                print(f"❌ Lỗi: {file_name} không phải file ZIP hợp lệ!")

    print("🎉 Hoàn thành giải nén tất cả file ZIP!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("❌ Sai cú pháp! Cách dùng:")
        print("👉 python unzip_all.py <input_folder> <output_folder>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    unzip_all_files(input_dir, output_dir)


# import os
# import shutil

# # Thư mục cha chứa các batch
# source_root = "/home/datnvt/project/Medical_CLARA/data_2"
# # Thư mục đích sau khi gộp
# destination = "/home/datnvt/project/Medical_CLARA/all_images"

# # Tạo thư mục đích nếu chưa có
# os.makedirs(destination, exist_ok=True)

# # Lặp qua từng thư mục con (batch_x_y)
# for subfolder in os.listdir(source_root):
#     subfolder_path = os.path.join(source_root, subfolder)
#     if os.path.isdir(subfolder_path):
#         for file in os.listdir(subfolder_path):
#             src_file = os.path.join(subfolder_path, file)
#             dst_file = os.path.join(destination, file)

#             # Nếu trùng tên thì xử lý đổi tên tránh ghi đè
#             if os.path.exists(dst_file):
#                 base, ext = os.path.splitext(file)
#                 i = 1
#                 while os.path.exists(dst_file):
#                     dst_file = os.path.join(destination, f"{base}_{i}{ext}")
#                     i += 1

#             shutil.copy2(src_file, dst_file)

# print("✅ Gộp ảnh xong vào:", destination)
