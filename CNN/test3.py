import os

dir_path = r'D:\EEE\Dataset\CUAVE_sub\sub1'
file_name = os.listdir(dir_path)
file_path = []
for i in range(0, len(file_name)):
    file_path.append(os.path.join(dir_path, file_name[i]))

print(file_path)