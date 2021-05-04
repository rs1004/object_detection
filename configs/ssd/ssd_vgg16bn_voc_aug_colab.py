__root_dir = '/content/object_detection'
__data = 'voc'
__input_size = 300
__version = 'ssd_vgg16bn_voc_aug'

__data_dir = __root_dir + '/data/' + __data
__out_dir = '/content/drive/MyDrive/result/' + __version

# データ
__mean = [0.485, 0.456, 0.406]
__std = [0.229, 0.224, 0.225]
data = dict(
    data_dir=__data_dir,
    bbox_fmt='cxcywh',
    train_pipeline=dict(
        albu=[
            dict(type='RandomScale', scale_limit=(0.8, 1.2)),
            dict(type='RandomSizedBBoxSafeCrop', height=__input_size, width=__input_size, erosion_rate=0.3),
            dict(type='HorizontalFlip'),
            dict(type='ColorJitter', brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ],
        torch=[
            dict(type='ToTensor'),
            dict(type='Normalize', mean=__mean, std=__std)
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
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005)
scheduler = dict(type='MultiStepLRWarmUpRestarts', milestones=[50, 75], gamma=0.1, eta_min=0.0001, T_up=10)
runtime = dict(
    batch_size=32,
    epochs=100,
    out_dir=__out_dir,
    resume=False,
    eval_interval=10
)

# 予測・評価
predictor = dict(
    conf_thresh=0.4,
    iou_thresh=0.45
)
evaluator = dict(
    anno_path=__data_dir + '/annotations/instances_val.json',
    pred_path=__out_dir + '/test/pred_val.json'
)
