import os

root_path = 'data/labels_detect/val'
del_path = 'data/images'

txt = os.listdir(root_path)

for i in txt:
    f = open(os.path.join(root_path, i), mode='r')
    lines = f.readlines()
    if len(lines) == 0:
        file_path = os.path.join(del_path, i[:-4] + '.jpg')
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print("File not found in the directory")
    f.close()
