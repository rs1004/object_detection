import argparse
import torch
from shutil import rmtree
from datasets import DetectionDataset
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
test_dir = Path(args.out_dir) / args.version / 'test'
weights_dir = Path(args.out_dir) / args.version / 'weights'
rmtree(test_dir, ignore_errors=True)
test_dir.mkdir(parents=True, exist_ok=True)

with open(Path(data_dir) / 'labels', 'r') as f:
    classes = f.read().split('\n')
    num_classes = len(classes)

# データ生成
dataset = DetectionDataset(
    data_dir=data_dir,
    input_size=args.input_size,
    phase='train'
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    collate_fn=dataset.collate_fn,
    shuffle=False
)

# モデル
model = SSD(num_classes=num_classes)

weights_path = weights_dir / 'latest.pth'
if weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# 推論・評価
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

predictor = model.inference
painter = BBoxPainter(classes=classes, save_dir=test_dir)

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

<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
''')

num_done = 0
for images, bboxes, labels in tqdm(dataloader, total=len(dataloader)):
    model.eval()
    torch.set_grad_enabled(False)

    # to GPU device
    images = images.to(device)

    # forward
    outputs = model(images)

    # inference + evaluation
    num_done = predictor(images, outputs, num_done, bbox_painter=painter)

    if num_done > 20:
        break
