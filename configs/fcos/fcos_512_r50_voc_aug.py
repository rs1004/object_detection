from pathlib import Path


__data = 'voc07+12'
__input_size = 512
__version = 'fcos_512_r50_voc_aug'

if Path('/content/object_detection').exists():
    __data_dir = '/content/object_detection/data/' + __data
    __out_dir = '/content/drive/MyDrive/result/' + __version
else:
    __data_dir = '/home/sato/work/object_detection/data/' + __data
    __out_dir = '/home/sato/work/object_detection/result/' + __version

# データ
__mean = [0.485, 0.456, 0.406]
__std = [0.229, 0.224, 0.225]
data = dict(
    data_dir=__data_dir,
    bbox_fmt='cxcywh',
    train_pipeline=dict(
        albu=[
            dict(type='ToFloat32'),
            dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
            dict(type='Expand', mean=tuple(v * 255 for v in __mean), ratio_range=(1, 4)),
            dict(type='MinIoURandomCrop'),
            dict(type='Resize', height=__input_size, width=__input_size),
            dict(type='HorizontalFlip'),
        ],
        torch=[
            dict(type='ToTensor'),
            dict(type='Normalize', mean=__mean, std=__std),
            dict(type='Dropout', p=(0.0, 0.05))
        ]
    ),
    val_pipeline=dict(
        albu=[
            dict(type='ToFloat32'),
            dict(type='Resize', height=__input_size, width=__input_size)
        ],
        torch=[
            dict(type='ToTensor'),
            dict(type='Normalize', mean=__mean, std=__std)
        ]
    )
)

# モデル
model = dict(type='fcos', num_classes=20, backbone='resnet50', backbone_weight=None)

# 学習
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = dict(type='MultiStepLRWarmUpRestarts', gamma=0.1, milestones=[75, 90], eta_min=0.0001, T_up=2)
runtime = dict(
    batch_size=24,
    epochs=100,
    out_dir=__out_dir,
    resume=False,
    eval_interval=10
)

# 予測・評価
predictor = dict(
    iou_thresh=0.45
)
evaluator = dict(
    anno_path=__data_dir + '/annotations/instances_val.json',
    pred_path=__out_dir + '/test/pred_val.json'
)
