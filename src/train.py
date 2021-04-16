import argparse
import torch
from datautils import DetectionDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from models import SSD
from pathlib import Path
from tqdm import tqdm


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

parser.add_argument('--work_dir', help='root directory of this repository', default='/home/sato/work/object_detection')
parser.add_argument('--data_name', help='same as the directory name placed under ./data', default='voc')
parser.add_argument('--batch_size', help='batch size of loaded data', type=int, default=2)
parser.add_argument('--input_size', help='input image size to model', type=int, default=300)
parser.add_argument('--num_classes', help='number of classes to be classified', type=int, default=2)
parser.add_argument('--epochs', help='number of epochs', type=int, default=50)
parser.add_argument('--version', help='used for output directory name', default='ssd_voc')

args = parser.parse_args()
# --------------------------------------------------

data_dir = f'{args.work_dir}/data/{args.data_name}'
log_dir = f'{args.work_dir}/result/{args.version}/logs'
weights_dir = Path(f'{args.work_dir}/result/{args.version}/weights')

# データ生成
dataloaders = {}
for phase in ['train', 'val']:
    dataset = DetectionDataset(
        data_dir=data_dir,
        input_size=args.input_size,
        phase=phase
    )

    dataloaders[phase] = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=phase == 'train'
    )

# モデル
model = SSD(num_classes=args.num_classes)

weights_path = weights_dir / 'latest.pth'
if weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# 学習
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = model.loss
optimizer = SGD(params=model.get_parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
scheduler = MultiStepLR(optimizer, milestones=[100, 200])

torch.backends.cudnn.benchmark = True

min_val_loss = 99999
with SummaryWriter(log_dir=log_dir) as writer:
    for epoch in range(1, args.epochs + 1):
        losses = {'train': 0, 'val': 0}
        loss_ls = {'train': 0, 'val': 0}
        loss_cs = {'train': 0, 'val': 0}
        counts = {'train': 0, 'val': 0}
        for phase, (images, bboxes, labels) in tqdm(chain(dataloaders), total=sum(len(dl) for dl in dataloaders.values()), desc=f'[Epoch {epoch:3}]'):
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
            out_ls, out_cs = model(images)
            loss, loss_l, loss_c = criterion(
                out_ls=out_ls,
                out_cs=out_cs,
                batch_bboxes=bboxes,
                batch_labels=labels
            )

            if phase == 'train':
                loss.backward()
                optimizer.step()

            losses[phase] += loss.item()
            loss_ls[phase] += loss_ls.item()
            loss_cs[phase] += loss_cs.item()
            counts[phase] += images.size(0)

        for phase in ['train', 'val']:
            losses[phase] /= counts[phase]
            loss_ls[phase] /= counts[phase]
            loss_cs[phase] /= counts[phase]

        print(f'loss: {losses["train"]:.04f}, val_loss: {losses["val"]:.04f}')

        # tensor board への書き込み
        writer.add_scalar('loss', losses["train"], epoch)
        writer.add_scalar('loss/localization', loss_ls["train"], epoch)
        writer.add_scalar('loss/confidence', loss_cs["train"], epoch)
        writer.add_scalar('val_loss', losses["val"], epoch)
        writer.add_scalar('val_loss/localization', loss_ls["val"], epoch)
        writer.add_scalar('val_loss/confidence', loss_cs["val"], epoch)

        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        # 重みファイル保存
        if losses['val'] < min_val_loss:
            weights_dir.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), weights_dir / 'latest.pth')
            min_val_loss = losses['val']

        # スケジューラ更新
        scheduler.step()
