# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import time

import cv2
import mmengine
import numpy as np


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def convert_wflw_to_coco(ann_dir, out_file):
    annotations = []
    images = []
    cnt = 0


    ann_dir_list = [ann_dir]

    for tv in ann_dir_list:
        ann_dir = '/opt/data/DMS/face6_data/data/WFLW/' + tv
        if "train" in ann_dir:
            landmark_txt = osp.join(ann_dir, 'train.txt')
        else:
            landmark_txt = osp.join(ann_dir, 'test.txt')
        with open(landmark_txt) as f:
            ann_lines = f.readlines()
        ann_datas = {}
        for line in ann_lines:
            img_name = line.split()[0]
            data = line.split()[1:]
            data = [float(x) * 256 for x in data]
            data = np.array(data).reshape(98,2)
            ann_datas[img_name] = data

        for idx, data_tuple in enumerate(mmengine.track_iter_progress(ann_datas.items())):
            img_name,ann_data = data_tuple
            cnt += 1
            file_name = ann_dir + "/" + img_name
            img_path = file_name

            img = cv2.imread(img_path)

            keypoints = []
            for point in ann_data:
                x, y = point
                x, y = float(x), float(y)
                keypoints.append([x, y, 2])
            keypoints = np.array(keypoints)

            x1, y1, _ = np.amin(keypoints, axis=0)
            x2, y2, _ = np.amax(keypoints, axis=0)
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]

            image = {}
            image['id'] = cnt
            image['file_name'] = f'{file_name.split("/")[-1]}'
            image['height'] = img.shape[0]
            image['width'] = img.shape[1]
            images.append(image)

            ann = {}
            ann['keypoints'] = keypoints.reshape(-1).tolist()
            ann['image_id'] = cnt
            ann['id'] = cnt
            ann['num_keypoints'] = len(keypoints)
            ann['bbox'] = bbox
            ann['iscrowd'] = 0
            ann['area'] = int(ann['bbox'][2] * ann['bbox'][3])
            ann['category_id'] = 1

            annotations.append(ann)

    cocotype = {}

    cocotype['info'] = {}
    cocotype['info']['description'] = 'WFLW'
    cocotype['info']['version'] = 1.0
    cocotype['info']['year'] = time.strftime('%Y', time.localtime())
    cocotype['info']['date_created'] = time.strftime('%Y/%m/%d',
                                                     time.localtime())

    cocotype['images'] = images
    cocotype['annotations'] = annotations
    cocotype['categories'] = [{
        'supercategory': 'person',
        'id': 1,
        'name': 'face',
        'keypoints': [],
        'skeleton': []
    }]

    json.dump(
        cocotype,
        open(out_file, 'w'),
        ensure_ascii=False,
        default=default_dump)
    print(f'done {out_file}')


if __name__ == '__main__':
    if not osp.exists('/opt/data/DMS/face6_data/data/WFLW/annotations'):
        os.makedirs('/opt/data/DMS/face6_data/data/WFLW/annotations')
    for tv in ['train', 'test']:
        print(f'processing {tv}')
        convert_wflw_to_coco(tv, f'/opt/data/DMS/face6_data/data/WFLW/annotations/WFLW_{tv}.json')
