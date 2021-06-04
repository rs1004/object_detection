"""
Below is a modified version of https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/pipelines/transforms.py
It passes through the pipeline of albumentations, but only BBox of "Coco Format" can be processed.
"""

import numpy as np
import cv2


class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from RGB to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to RGB
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 prob=0.5):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.prob = prob

    def __call__(self, *args, force_apply=False, **results):
        assert 'image' in results

        if np.random.uniform(0, 1) > self.prob and not force_apply:
            return results

        image = results['image']

        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-self.brightness_delta,
                                      self.brightness_delta)
            image += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(self.contrast_lower,
                                          self.contrast_upper)
                image *= alpha

        # convert color from RGB to HSV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # random saturation
        if np.random.randint(2):
            image[..., 1] *= np.random.uniform(self.saturation_lower,
                                               self.saturation_upper)

        # random hue
        if np.random.randint(2):
            image[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
            image[..., 0][image[..., 0] > 360] -= 360
            image[..., 0][image[..., 0] < 0] += 360

        # convert color from HSV to RGB
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(self.contrast_lower,
                                          self.contrast_upper)
                image *= alpha

        # randomly swap channels
        if np.random.randint(2):
            image = image[..., np.random.permutation(3)]

        results['image'] = image
        return results


class Expand:
    """Random expand the image & bboxes.
    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 ratio_range=(1, 4),
                 prob=0.5):

        self.ratio_range = ratio_range
        self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.prob = prob

    def __call__(self, *args, force_apply=False, **results):
        assert 'image' in results
        assert 'bboxes' in results

        if np.random.uniform(0, 1) > self.prob and not force_apply:
            return results

        image = results['image']

        h, w, c = image.shape
        ratio = np.random.uniform(self.min_ratio, self.max_ratio)

        # speedup expand when meets large image
        if np.all(self.mean == self.mean[0]):
            expand_image = np.empty((int(h * ratio), int(w * ratio), c),
                                    image.dtype)
            expand_image.fill(self.mean[0])
        else:
            expand_image = np.full((int(h * ratio), int(w * ratio), c),
                                   self.mean,
                                   dtype=image.dtype)
        left = int(np.random.uniform(0, w * ratio - w))
        top = int(np.random.uniform(0, h * ratio - h))
        expand_image[top:top + h, left:left + w] = image

        results['image'] = expand_image
        # expand bboxes
        results['bboxes'] = results['bboxes'] + np.tile((left, top, 0, 0), 1)

        return results


class MinIoURandomCrop:
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding bboxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True,
                 prob=1.0):
        # 1: return ori image
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border
        self.prob = prob

    def __call__(self, *args, force_apply=False, **results):
        assert 'image' in results
        assert 'bboxes' in results
        assert 'labels' in results

        if np.random.uniform(0, 1) > self.prob and not force_apply:
            return results

        image = results['image']
        h, w, c = image.shape
        bboxes = np.array(results['bboxes'])
        bboxes[:, 2:] += bboxes[:, :2]  # xywh -> xyxy
        while True:
            mode = np.random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = np.random.uniform(self.min_crop_size * w, w)
                new_h = np.random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = np.random.uniform(w - new_w)
                top = np.random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = self.bbox_overlaps(
                    patch.reshape(-1, 4), bboxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of bboxes should inside the crop image
                # only adjust bboxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    mask = self.is_center_of_bboxes_in_patch(bboxes, patch)
                    if not mask.any():
                        continue

                    # bboxes = bboxes.copy()
                    mask = self.is_center_of_bboxes_in_patch(bboxes, patch)
                    bboxes = bboxes[mask]
                    if self.bbox_clip_border:
                        bboxes[:, 2:] = bboxes[:, 2:].clip(max=patch[2:])
                        bboxes[:, :2] = bboxes[:, :2].clip(min=patch[:2])
                    bboxes -= np.tile(patch[:2], 2)

                    bboxes[:, 2:] -= bboxes[:, :2]  # xyxy -> xywh
                    results['bboxes'] = bboxes
                    # labels
                    results['labels'] = np.array(results['labels'])[mask]

                # adjust the image no matter whether the gt is empty before crop
                image = image[patch[1]:patch[3], patch[0]:patch[2]]
                results['image'] = image

                return results

    def bbox_overlaps(self, bboxes1, bboxes2, eps=1e-6):
        max_xy = np.minimum(bboxes1[:, 2:], bboxes2[:, 2:])
        min_xy = np.maximum(bboxes1[:, :2], bboxes2[:, :2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf).prod(axis=1)
        area_1 = (bboxes1[:, 2:] - bboxes1[:, :2]).prod(axis=1)
        area_2 = (bboxes2[:, 2:] - bboxes2[:, :2]).prod(axis=1)
        union = area_1 + area_2 - inter
        return inter / (union + eps)

    # adjust bboxes
    def is_center_of_bboxes_in_patch(self, bboxes, patch):
        center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
        mask = ((center[:, 0] > patch[0]) *
                (center[:, 1] > patch[1]) *
                (center[:, 0] < patch[2]) *
                (center[:, 1] < patch[3]))
        return mask
