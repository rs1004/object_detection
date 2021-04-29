from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
import json


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

    def run(self):
        cocoGt = COCO(self.anno_path)
        cocoDt = cocoGt.loadRes(self.pred_path)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def dump_pred(self, d: dict):
        Path(self.pred_path).parent.mkdir(exist_ok=True, parents=True)
        with open(self.pred_path, 'w') as f:
            json.dump(d, f)
