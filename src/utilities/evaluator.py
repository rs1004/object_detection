from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
import sys
import json


class SetIO():
    """with構文でI/Oを切り替えるためのクラス"""

    def __init__(self, filename: str):
        self.filename = filename

    def __enter__(self):
        sys.stdout = _STDLogger(out_file=self.filename)

    def __exit__(self, *args):
        sys.stdout = sys.__stdout__


class _STDLogger():
    """カスタムI/O"""

    def __init__(self, out_file='out.log'):
        self.log = open(out_file, "a+")

    def write(self, message):
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        pass


class Evaluator:
    """ 評価を行うクラス

    Args:
        anno_path (str): アノテーションファイル（.json）のパス
        pred_path (str): 予測結果ファイル（.json）のパス

    Outputs:
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = x.xxx
    """

    def __init__(self, anno_path, pred_path):
        self.anno_path = anno_path
        self.pred_path = pred_path

    def run(self, epoch: int):
        out_log_path = Path(self.pred_path).parent / f'eval@{epoch}epc.log'

        cocoGt = COCO(self.anno_path)
        cocoDt = cocoGt.loadRes(self.pred_path)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        with SetIO(out_log_path):
            cocoEval.summarize()
        with open(out_log_path, 'r') as f:
            print(f.read())

    def dump_pred(self, d: dict):
        Path(self.pred_path).parent.mkdir(exist_ok=True, parents=True)
        with open(self.pred_path, 'w') as f:
            json.dump(d, f)
