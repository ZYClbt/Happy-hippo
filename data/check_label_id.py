import numpy as np
import os
import os.path as osp
import cv2
from tqdm import tqdm

data_dir = '.'

CLASS = {0: 'Azibo', 1: 'Natascha', 2: 'Ohini', 3: 'Swela', 4: 'Frodo', 5: 'Dorien', 6: 'Lome', 7: 'Lobo', 8: 'Kisha', 9: 'Fraukje', 10: 'Riet', 11: 'Sandra', 12: 'Kofi', 13: 'Bambari', 14: 'Tai', 15: 'Corrie', 16: 'Maja'}
counts = 0
train_counts = 0
val_counts = 0

def read_files(file_dir):
    file_path_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_path_list.append(osp.join(root, file))
    file_path_list = sorted(file_path_list, key=lambda x: x)
    return file_path_list


def draw_bbox(frame, classes, body_box, body_color, txt_color=(255, 255, 255), subset='train'):
    global counts, train_counts, val_counts
    line_width = max(round(sum(frame.shape) / 2 * 0.003), 2)

    for name_id, box in zip(classes, body_box):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(frame, p1, p2, body_color, thickness=line_width, lineType=cv2.LINE_AA)
        txt = CLASS[name_id]
        tf = max(line_width - 1, 1)  # font thickness
        w, h = cv2.getTextSize(txt, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(frame, p1, p2, body_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(frame, txt, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, line_width / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
        counts += 1
        if subset == 'train':
            train_counts += 1
        else:
            val_counts += 1
    return frame


def process_label(label_list, height, width):
    ret = {
        'cls': [],
        'bbox': []
    }
    for label in label_list:
        name_id, centerx, centery, bboxw, bboxh = label.strip('\n').split(' ')
        name_id, centerx, centery, bboxw, bboxh = int(name_id), float(centerx)*width, float(centery)*height, float(bboxw)*width, float(bboxh)*height
        # print(name_id, centerx, centery, bboxw, bboxh)
        minx, miny, maxx, maxy = centerx - bboxw/2, centery - bboxh/2, centerx + bboxw/2, centery + bboxh/2
        bbox = np.array([minx, miny, maxx, maxy])
        ret['cls'].append(name_id)
        ret['bbox'].append(bbox)
    return ret

def main():
    os.makedirs(osp.join(data_dir, 'vis_labels_id'), exist_ok=True)
    os.makedirs(osp.join(data_dir, 'vis_labels_id', 'train'), exist_ok=True)
    os.makedirs(osp.join(data_dir, 'vis_labels_id', 'val'), exist_ok=True)
    label_path_list = read_files(osp.join(data_dir, 'labels_id'))
    print('total num of labels-id: ', len(label_path_list))
    
    global counts, train_counts, val_counts
    body_color = (46, 74, 246)

    for label_path in tqdm(label_path_list):
        image_path = label_path.replace('labels_id', 'images').replace('.txt', '.jpg').replace('train/','').replace('val/','')
        assert osp.isfile(image_path), f'check image {image_path}'

        # read image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # read bounding box annotation
        with open(label_path, 'r') as f:
           label = f.readlines()
        label = process_label(label, height, width)

        # visualize bounding box with name id
        subset = 'train' if 'train' in label_path else 'val'
        image = draw_bbox(image, label['cls'], label['bbox'], body_color, subset=subset)
        cv2.imwrite(label_path.replace('labels_id', 'vis_labels_id').replace('.txt', '.jpg'), image)

    print(f'total annot num of ids: {counts} | train {train_counts}, val {val_counts}')

if __name__ == '__main__':
    main()