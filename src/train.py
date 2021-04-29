import argparse
import torch
from shutil import rmtree
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from datasets import DetectionDataset
from torch.utils.data import DataLoader
from models import Model
from utilities import Config, Optimizer, Scheduler, Predictor, Evaluator

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


def chain(loaders: dict) -> tuple:
    """ dataloader を繋ぐ (train phase ~> val phase を連鎖的に行う)
    Args:
        loaders (dict): {
            'train': train dataloader
            'val': val dataloader
        }
    Yields:
        Iterator[tuple]: (phase, (images tensor, labels tensor))
    """
    for phase in ['train', 'val']:
        for element in loaders[phase]:
            yield phase, element


# ----------------- パラメータ設定 -----------------
parser = argparse.ArgumentParser()
parser.add_argument('config_path', help='config file path')
args = parser.parse_args()
# --------------------------------------------------

cfg = Config(args.config_path)

# 実行準備
log_dir = cfg.runtime['out_dir'] + '/logs'
weights_dir = cfg.runtime['out_dir'] + '/weights'
initial_epoch = 1
if cfg.runtime['resume']:
    for log_path in Path(log_dir).glob('**/events.out.*'):
        ea = event_accumulator.EventAccumulator(log_path.as_posix())
        ea.Reload()
        if 'loss/train' in ea.Tags()['scalars']:
            initial_epoch = max(event.step for event in ea.Scalars('loss/train')) + 1
else:
    for d in [log_dir, weights_dir]:
        rmtree(d, ignore_errors=True)
        Path(d).mkdir(parents=True)

# データ生成
dataloaders = {}
for phase in ['train', 'val']:
    dataset = DetectionDataset(
        data_dir=cfg.data['data_dir'],
        pipeline=cfg.data[f'{phase}_pipeline'],
        fmt=cfg.data['bbox_fmt'],
        phase=phase
    )

    dataloaders[phase] = DataLoader(
        dataset=dataset,
        batch_size=cfg.runtime['batch_size'],
        collate_fn=dataset.collate_fn,
        shuffle=phase == 'train'
    )

# モデル
model = Model(**cfg.model)

weights_path = f'{weights_dir}/latest.pth'
if Path(weights_path).exists():
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# 学習
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = model.loss
optimizer = Optimizer(params=model.get_parameters(), cfg=cfg.optimizer)
scheduler = Scheduler(optimizer=optimizer, cfg=cfg.scheduler)

# 予測・評価
predictor = Predictor(**cfg.predictor)
evaluator = Evaluator(**cfg.evaluator)

torch.backends.cudnn.benchmark = True

print(f'''<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
<-><-><-><-><-><-><-> TRAINING START ! <-><-><-><-><-><-><->
<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
[CONFIG]
- config     : {args.config_path}
- batch_size : {cfg.runtime['batch_size']}
- epochs     : {cfg.runtime['epochs']}
- out_dir    : {cfg.runtime['out_dir']}

[RUNTIME]
- {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

[DATASET]
- data_dir   : {cfg.data['data_dir']}

[MODEL]
- {cfg.model['type']}

[OPTIMIZER]
- {optimizer.__class__.__name__}
- hyper params : {optimizer.defaults}

[SCHEDULER]
- {scheduler.__class__.__name__}
- milestones: lr = {scheduler.get_last_lr()[-1]:.1e} -> {
    ' -> '.join(
        f'{scheduler.get_last_lr()[-1] * pow(scheduler.gamma, i):.1e} ({s} epc~)'
        for i, s in enumerate(scheduler.milestones.keys(), start=1))}

<-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
''')
min_val_loss = 99999
with SummaryWriter(log_dir=log_dir) as writer:
    for epoch in range(initial_epoch, cfg.runtime['epochs'] + initial_epoch):
        losses = {'train': defaultdict(lambda: 0), 'val': defaultdict(lambda: 0)}
        counts = {'train': 0, 'val': 0}
        result = []
        for phase, (images, image_metas, gt_bboxes, gt_labels) in tqdm(
                chain(dataloaders),
                total=sum(len(dl) for dl in dataloaders.values()),
                desc=f'[Epoch {epoch:3}]'):

            if phase == 'train':
                model.train()
                optimizer.zero_grad()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)

            # to GPU device
            images = images.to(device)

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, gt_bboxes, gt_labels)

            if phase == 'train':
                loss['loss'].backward()
                optimizer.step()
            elif epoch % cfg.runtime['eval_interval'] == 0:
                bboxes, confs, class_ids = model.pre_predict(outputs)
                result += predictor.run(images, image_metas, bboxes, confs, class_ids)

            for kind in loss.keys():
                losses[phase][kind] += loss[kind].item() * images.size(0)
            counts[phase] += images.size(0)

        for phase in ['train', 'val']:
            for kind in losses[phase].keys():
                losses[phase][kind] /= counts[phase]

        # tensor board への書き込み
        for phase in ['train', 'val']:
            for kind in losses[phase].keys():
                writer.add_scalar(f'{kind}/{phase}', losses[phase][kind], epoch)
        for i, lr in enumerate(scheduler.get_last_lr(), start=1):
            writer.add_scalar(f'lr/lr_{i}', lr, epoch)

        print(f'  loss     : {losses["train"].pop("loss"):.04f} ({", ".join([f"{kind}: {value:.04f}" for kind, value in losses["train"].items()])})')
        print(f'  val_loss : {losses["val"].pop("loss"):.04f} ({", ".join([f"{kind}: {value:.04f}" for kind, value in losses["val"].items()])})')

        # 評価
        if epoch % cfg.runtime['eval_interval'] == 0:
            if len(result) > 0:
                evaluator.dump_pred(result)
                evaluator.run()
            else:
                print('No Object Detected. Skip Evaluation')

        # 重みファイル保存
        if losses['val']['loss'] < min_val_loss:
            torch.save(model.state_dict(), weights_path)
            min_val_loss = losses['val']['loss']

        # スケジューラ更新
        scheduler.step()
