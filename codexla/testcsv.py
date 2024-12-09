import pandas as pd
import os

folder_path = r'E:\archive\train'

# Kiểm tra xem thư mục có chứa tệp CSV không
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Nếu có tệp CSV, đọc nó
if csv_files:
    data = pd.read_csv(os.path.join(folder_path, csv_files[0]))  # Đọc tệp CSV đầu tiên trong thư mục
    print(data.head())  # Xem 5 dòng đầu tiên của dữ liệu
else:
    print('Không có tệp CSV trong thư mục.')
