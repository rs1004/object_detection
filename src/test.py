import argparse
import torch
from shutil import rmtree
from datasets import DetectionDataset, MetaData
from torch.utils.data import DataLoader
from models import SSD
from utils import BBoxPainter
from pathlib import Path
from tqdm import tqdm

# ----------------- パラメータ設定 -----------------
parser = argparse.ArgumentParser()

parser.add_argument('--data_name', help='same as the directory name placed under ./data', default='voc')
parser.add_argument('--out_dir', help='directory to save weight files etc', default='./result')
parser.add_argument('--batch_size', help='batch size of loaded data', type=int, default=32)
parser.add_argument('--input_size', help='input image size to model', type=int, default=300)
parser.add_argument('--version', help='used for output directory name', default='ssd_voc')

args = parser.parse_args()
# --------------------------------------------------

data_dir = f'./data/{args.data_name}'
meta = MetaData(data_dir=data_dir)

test_dir = f'{args.out_dir}/{args.version}/test'
weights_path = f'{args.out_dir}/{args.version}/weights/latest.pth'
rmtree(test_dir, ignore_errors=True)
Path(test_dir).mkdir(parents=True, exist_ok=True)

# データ生成
dataset = DetectionDataset(
    data_dir=data_dir,
    input_size=args.input_size,
    norm_cfg=meta.norm_cfg,
    phase='val'
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    collate_fn=dataset.collate_fn,
    shuffle=False
)

# モデル
model = SSD(num_classes=meta.num_classes)

if Path(weights_path).exists():
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# 推論・評価
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

predictor = model.predict
painter = BBoxPainter(classes=['back'] + meta.classes, save_dir=test_dir)

torch.backends.cudnn.benchmark = True

print(f'''<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
<-><-><-><-><-><-><->   TEST START !   <-><-><-><-><-><-><->
<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
[CONFIG]
- version    : {args.version}
- batch_size : {args.batch_size}
- out_dir    : {args.out_dir}

[RUNTIME]
- {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

[DATASET]
- {args.data_name}

[MODEL]
- {model.__class__.__name__}{args.input_size}

[WEIGHTS]
- {weights_path if Path(weights_path).exists() else 'None'}

<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
''')

model.eval()
torch.set_grad_enabled(False)

num_done = 0
for images, image_metas, gt_bboxes, gt_labels in tqdm(dataloader, total=len(dataloader)):
    # to GPU device
    images = images.to(device)

    # forward
    outputs = model(images)

    # inference + evaluation
    result = predictor(images, image_metas, outputs, norm_cfg=meta.norm_cfg, bbox_painter=painter)
    num_done += len(result)

    if num_done > 20:
        break
