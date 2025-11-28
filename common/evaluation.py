import torch


class Evaluator:
    """ 计算预测结果与真实标签的交集和并集 """

    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_mask, target):
        gt_mask = target

        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [], [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            _inter = _pred_mask[_pred_mask == _gt_mask]
            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter.float(), bins=2, min=0, max=1)
                # 计算输入张量的直方图。以min和max为range边界，将其均分成bins个直条
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask.float(), bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask.float(), bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()  # 求矩阵的转置
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union
