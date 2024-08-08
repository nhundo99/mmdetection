# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class CocoDataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes': (
        'cube_class_1', 'cube_class_2', 'cube_class_3', 'cube_class_4',
        'cube_class_5', 'cube_class_6', 'cube_class_7', 'cube_class_8',
        'cube_class_9', 'cube_class_10', 'cube_class_11', 'cube_class_12',
        'cube_class_13', 'cube_class_14', 'cube_class_15', 'cube_class_16',
        'cube_class_17', 'cube_class_18', 'cube_class_19', 'cube_class_20',
        'cube_class_21', 'cube_class_22', 'cube_class_23', 'cube_class_24',
        'cube_class_25', 'cube_class_26', 'cube_class_27', 'cube_class_28',
        'cube_class_29', 'cube_class_30', 'cube_class_31', 'cube_class_32',
        'cube_class_33', 'cube_class_34', 'cube_class_35', 'cube_class_36',
        'cube_class_37', 'cube_class_38', 'cube_class_39', 'cube_class_40',
        'cube_class_41', 'cube_class_42', 'cube_class_43', 'cube_class_44',
        'cube_class_45', 'cube_class_46', 'cube_class_47', 'cube_class_48',
        'cube_class_49', 'cube_class_50', 'cube_class_51', 'cube_class_52',
        'cube_class_53', 'cube_class_54', 'cube_class_55', 'cube_class_56',
        'cube_class_57', 'cube_class_58', 'cube_class_59', 'cube_class_60',
        'cube_class_61', 'cube_class_62', 'cube_class_63', 'cube_class_64',
        'cube_class_65', 'cube_class_66', 'cube_class_67', 'cube_class_68',
        'cube_class_69', 'cube_class_70', 'cube_class_71', 'cube_class_72',
        'cube_class_73', 'cube_class_74', 'cube_class_75', 'cube_class_76',
        'cube_class_77', 'cube_class_78', 'cube_class_79', 'cube_class_80',
        'cube_class_81', 'cube_class_82', 'cube_class_83', 'cube_class_84',
        'cube_class_85', 'cube_class_86', 'cube_class_87', 'cube_class_88',
        'cube_class_89', 'cube_class_90', 'cube_class_91', 'cube_class_92',
        'cube_class_93', 'cube_class_94', 'cube_class_95', 'cube_class_96',
        'cube_class_97', 'cube_class_98', 'cube_class_99', 'cube_class_100')
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
