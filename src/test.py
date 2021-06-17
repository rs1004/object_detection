import argparse
import torch
from shutil import rmtree
from pathlib import Path
from tqdm import tqdm

from datasets import DetectionDataset
from torch.utils.data import DataLoader
from models import Model
from utilities import Config, Predictor, Evaluator

from tensorboard.backend.event_processing import event_accumulator


# ----------------- パラメータ設定 -----------------
parser = argparse.ArgumentParser()
parser.add_argument('config_path', help='config file path')
args = parser.parse_args()
# --------------------------------------------------

cfg = Config(args.config_path)

test_dir = cfg.runtime['out_dir'] + '/test'
log_dir = cfg.runtime['out_dir'] + '/logs'
weights_path = cfg.runtime['out_dir'] + '/weights/latest.pth'
rmtree(test_dir, ignore_errors=True)
Path(test_dir).mkdir(parents=True, exist_ok=True)
epoch = '-'
for log_path in Path(log_dir).glob('**/events.out.*'):
    ea = event_accumulator.EventAccumulator(log_path.as_posix())
    ea.Reload()
    if 'loss/train' in ea.Tags()['scalars']:
        epoch = max(event.step for event in ea.Scalars('loss/train')) + 1

# データ生成
dataset = DetectionDataset(
    data_dir=cfg.data['data_dir'],
    pipeline=cfg.data['val_pipeline'],
    fmt=cfg.data['bbox_fmt'],
    phase='val'
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=cfg.runtime['batch_size'],
    collate_fn=dataset.collate_fn,
    shuffle=False
)

# モデル
model = Model(**cfg.model)

if Path(weights_path).exists():
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# 予測・評価
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

predictor = Predictor(**cfg.predictor)
evaluator = Evaluator(**cfg.evaluator)

torch.backends.cudnn.benchmark = True

print(f'''<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
<-><-><-><-><-><-><->   TEST START !   <-><-><-><-><-><-><->
<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
[CONFIG]
- config     : {args.config_path}
- batch_size : {cfg.runtime['batch_size']}
- out_dir    : {cfg.runtime['out_dir']}

[RUNTIME]
- {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

[DATASET]
- data_dir   : {cfg.data['data_dir']}

[MODEL]
- {cfg.model['type']}

[WEIGHTS]
- {weights_path if Path(weights_path).exists() else 'None'}

<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
''')

model.eval()
torch.set_grad_enabled(False)

result = []
for images, image_metas, gt_bboxes, gt_labels in tqdm(dataloader, total=len(dataloader)):
    # to GPU device
    images = images.to(device)

    # forward
    outputs = model(images)

    # prediction + evaluation
    bboxes, scores, class_ids = model.pre_predict(outputs)
    result += predictor.run(image_metas, bboxes, scores, class_ids)

if len(result) > 0:
    evaluator.dump_pred(result)
    evaluator.run(epoch)
else:
    print('No Object Detected. Skip Evaluation')
