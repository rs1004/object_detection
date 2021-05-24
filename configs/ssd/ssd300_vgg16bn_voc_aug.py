from pathlib import Path


__data = 'voc07+12'
__input_size = 300
__version = 'ssd300_vgg16bn_voc_aug'

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
            dict(type='ColorJitter', brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
            dict(type='ShiftScaleRotate', shift_limit=0, rotate_limit=0, scale_limit=(-0.75, -0.0), border_mode=0,
                 value=tuple(v * 255 for v in __mean)
                 ),
            dict(type='RandomSizedBBoxSafeCrop', height=__input_size, width=__input_size, erosion_rate=0.48),
            dict(type='HorizontalFlip'),
        ],
        torch=[
            dict(type='ToTensor'),
            dict(type='Normalize', mean=__mean, std=__std),
            dict(type='Dropout', p=(0.0, 0.1))
        ]
    ),
    val_pipeline=dict(
        albu=[
            dict(type='Resize', height=__input_size, width=__input_size)
        ],
        torch=[
            dict(type='ToTensor'),
            dict(type='Normalize', mean=__mean, std=__std)
        ]
    )
)

# モデル
model = dict(type='ssd', num_classes=20, backborn='vgg16_bn', backborn_weight=None)

# 学習
train_conditions = [
    dict(keys=['bn', 'bias', '4_3.0'], weight_decay=0.0),
    dict(keys=['.'])
]
optimizer = dict(type='SGD', lr=0.0026, momentum=0.9, weight_decay=0.0005)
scheduler = dict(type='ExponentialLRWarmUpRestarts', gamma=0.97, eta_min=0.00026, T_up=10)
runtime = dict(
    batch_size=32,
    epochs=100,
    out_dir=__out_dir,
    resume=True,
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
