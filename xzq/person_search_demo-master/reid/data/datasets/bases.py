# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import numpy as np


class BaseDataset(object):
    """
    Base class of reid dataset
    """
    # 获取图像数据信息：人员编号数量，图片数量，摄像头数量
    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        # set函数，无序不重复元素集合
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams
    # 获取视频数据信息：人员总数，轨迹统计数量，摄像头数量，是否轨迹统计
    def get_videodata_info(self, data, return_tracklet_stats=False):
        pids, cams, tracklet_stats = [], [], []
        for img_paths, pid, camid in data:
            pids += [pid]
            cams += [camid]
            tracklet_stats += [len(img_paths)]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_stats:
            return num_pids, num_tracklets, num_cams, tracklet_stats
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, query):
        i=1


