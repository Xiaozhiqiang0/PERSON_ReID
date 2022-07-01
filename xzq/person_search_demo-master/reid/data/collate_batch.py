# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
    #
    # 我们可以使用 list() 转换来输出列表。
    #
    # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids
    # torch.stack改变维度，torch.cat不改变维度


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids
