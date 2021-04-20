from pathlib import Path
from shutil import copy
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def get_info():
    return {
        'description': 'VOC 2007 & 2012 Dataset',
        'url': '-',
        'version': '1.0',
        'year': '-',
        'contributer': '-',
        'date_created': '-'
    }


def get_licenses():
    return [{
        'url': '-',
        'id': 1,
        'name': '-'
    }]


def get_categories():
    return [{
        'supercategory': name,
        'id': i,
        'name': name} for i, name in enumerate(classes, start=1)
    ]


def get_images_and_annotations(phase, dst_dir):
    images = []
    annotations = []

    data_dir = Path('/home/sato/work/object_detection/data/VOCdevkit')
    dst_image_dir = dst_dir / phase
    dst_image_dir.mkdir(parents=True, exist_ok=True)

    id = 1
    for year in ['2007', '2012']:
        base_dir = data_dir / f'VOC{year}'
        data_list_path = base_dir / 'ImageSets' / 'Main' / f'{phase}.txt'

        with open(data_list_path, 'r') as f:
            file_names = f.read().split('\n')[:-1]

        for file_name in tqdm(file_names):
            new_file_name = f'2007_{file_name}' if year == '2007' else file_name
            image_path = base_dir / 'JPEGImages' / f'{file_name}.jpg'
            copy(image_path, dst_image_dir / f'{new_file_name}.jpg')

            label_path = base_dir / 'Annotations' / f'{file_name}.xml'
            root = ET.parse(label_path).getroot()
            height = int(root.find('size').find('height').text)
            width = int(root.find('size').find('width').text)

            image_id = int(new_file_name)
            images.append({
                'license': 1,
                'file_name': f'{new_file_name}.jpg',
                'coco_url': '-',
                'height': height,
                'width': width,
                'date_captured': '-',
                'flickr_url': '-',
                'id': image_id
            })

            for obj in root.iter('object'):
                category_id = classes.index((obj.find('name').text)) + 1
                xmin = int(obj.find('bndbox').find('xmin').text) - 1
                ymin = int(obj.find('bndbox').find('ymin').text) - 1
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymax = int(obj.find('bndbox').find('ymax').text)

                x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin

                annotations.append({
                    'segmentation': [x, y, x+w, y, x+w, y+h, x, y+h],
                    'area': float(w * h),
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': [x, y, w, h],
                    'category_id': category_id,
                    'id': id
                })
                id += 1

    return images, annotations


def main():
    dst_dir = Path('/home/sato/work/object_detection/data/voc')
    dst_anno_dir = dst_dir / 'annotations'
    dst_anno_dir.mkdir(parents=True, exist_ok=True)
    for phase in ['train', 'val']:
        info = get_info()
        licenses = get_licenses()
        categories = get_categories()
        images, annotations = get_images_and_annotations(phase=phase, dst_dir=dst_dir)

        instances = {
            'info': info,
            'images': images,
            'licenses': licenses,
            'annotations': annotations,
            'categories': categories
        }

        with open(dst_anno_dir / f'instances_{phase}.json', 'w') as f:
            json.dump(instances, f)


if __name__ == '__main__':
    main()
