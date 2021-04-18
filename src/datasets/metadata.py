import yaml
from pathlib import Path


class MetaData:
    """ メタデータを保持するクラス
    Args:
        data_dir (str): 画像データのディレクトリ

    Members:
        classes (list)    : クラスの名前一覧
        num_classes (int) : クラス数
        norm_cfg (dict)   : 画像の標準化で使用　{'mean': [r_mean, g_mean, b_mean],　'std' : [r_std,  g_std,  b_std ]}
    """

    def __init__(self, data_dir: str):
        meta_data = self._load_meta_data(data_dir)

        self.classes = meta_data['classes']
        self.num_classes = len(self.classes)
        self.norm_cfg = meta_data['norm_cfg']

    def _load_meta_data(self, data_dir):
        meta_path = Path(data_dir) / 'meta.yaml'
        with open(meta_path.as_posix(), 'r') as f:
            meta_data = yaml.load(f, Loader=yaml.SafeLoader)

        return meta_data
