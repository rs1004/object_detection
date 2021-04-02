from pathlib import Path
from shutil import copy
from tqdm import tqdm
import xml.etree.ElementTree as ET

dst_dir = Path('/home/sato/work/object_detection/data/voc')

data_list_dir = Path('/home/sato/work/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/ImageSets/Main')
image_dir = Path('/home/sato/work/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/JPEGImages')
label_dir = Path('/home/sato/work/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/Annotations')

data = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

for t in ['train', 'val']:
    with open(data_list_dir / (t + '.txt'), 'r') as f:
        data_list = f.read().split('\n')[:-1]

    for file_name in tqdm(data_list):
        image_path = image_dir / (file_name + '.jpg')
        image_dst_dir = dst_dir / t / 'images'
        image_dst_dir.mkdir(parents=True, exist_ok=True)
        copy(image_path, image_dst_dir)

        label_path = label_dir / (file_name + '.xml')
        label_dst_dir = dst_dir / t / 'labels'
        label_dst_dir.mkdir(parents=True, exist_ok=True)
        root = ET.parse(label_path).getroot()
        height = int(root.find('size').find('height').text)
        width = int(root.find('size').find('width').text)

        labels = ['{} {} {} {} {}'.format(
            data.index(obj.find('name').text),
            int(obj.find('bndbox').find('xmin').text) / width,
            int(obj.find('bndbox').find('ymin').text) / height,
            (int(obj.find('bndbox').find('xmax').text) - int(obj.find('bndbox').find('xmin').text)) / width,
            (int(obj.find('bndbox').find('ymax').text) - int(obj.find('bndbox').find('ymin').text)) / height
        ) for obj in root.iter('object')]

        with open(label_dst_dir / (file_name + '.txt'), 'w') as f:
            f.write('\n'.join(labels))
