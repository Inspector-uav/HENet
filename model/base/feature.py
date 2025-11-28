from collections import Counter  # 可以用来统计可哈希对象（如列表、元组等）中每个元素的出现次数
import numpy as np


def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None):
    r""" 从VGG中提取中间特征 """
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(feat.clone())
    return feats


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" 从ResNet中提取中间特征"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    layer_nums = np.cumsum(list(Counter(lids).values()))  # [3 7 13 16]
    layer_nums_iter = iter(layer_nums)  # 将layer_nums列表转换为一个迭代器对象
    layer_id = next(layer_nums_iter)  # 使用next()方法来逐个获取layer_nums列表中的元素，表示在ResNet中需要保留输出的层的索引
    layers = [feat]

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        # hid表示索引位置，(bid, lid)表示配对的元组，zip(bottleneck_ids, lids)将对应位置一一对应
        # feat_ids = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        # bid = [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2]
        # lid = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:  # 把layer1过完，从layer2开始保留
            feats.append(feat.clone())  # 第2、3、4层的输出

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        if hid + 1 == layer_id:  # [3 7 13 16]保留了每一层layer最后一个bottleneck的输出
            if layer_id != layer_nums[-1]:
                layer_id = next(layer_nums_iter)
            layers.append(feat.clone())  # 每层最后的输出

    return feats, layers
