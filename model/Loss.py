import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # 计算输入图像和目标图像的边缘
        edge_inputs = self.get_edges(inputs)
        edge_targets = self.get_edges(targets)

        # 计算边缘的BCE Loss
        edge_loss = self.bce_loss(edge_inputs, edge_targets)

        return edge_loss

    def get_edges(self, img):
        sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3).to(img.device)
        sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3).to(img.device)

        edge_x = F.conv2d(img, sobel_x, padding=1)
        edge_y = F.conv2d(img, sobel_y, padding=1)

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge


class IOU(nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return self._iou(pred, target)

    def _iou(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(b):
            Iand1 = torch.sum(target[i, :, :] * pred[i, :, :])
            Ior1 = torch.sum(target[i, :, :]) + torch.sum(pred[i, :, :])
            IoU1 = Iand1 / Ior1
            IoU = IoU + (1 - IoU1)
        if self.size_average:
            return IoU / b
        else:
            return IoU


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.iou_loss = IOU()
        self.edge_loss = EdgeLoss()
        self.dice_loss = DiceLoss()

        # 定义损失的权重，并将它们作为可训练参数
        self.bce_weight = nn.Parameter(torch.tensor(1.0))
        self.iou_weight = nn.Parameter(torch.tensor(1.0))
        self.edge_weight = nn.Parameter(torch.tensor(1.0))
        self.dice_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)

        bce_loss = self.cross_entropy_loss(logit_mask, gt_mask)

        pred_mask = logit_mask.argmax(dim=1).float()

        iou_loss = self.iou_loss(pred_mask, gt_mask)

        total_loss = (self.bce_weight * bce_loss +
                      self.iou_weight * iou_loss)

        return total_loss


# 随机数生成测试
if __name__ == "__main__":
    logit_mask = torch.randn(4, 2, 400, 400, requires_grad=True).to('cuda')
    gt_mask = torch.randint(0, 2, (4, 400, 400)).to('cuda')

    criterion = CombinedLoss().to('cuda')
    loss = criterion(logit_mask, gt_mask)

    print(f"Computed loss: {loss.item()}")
