r""" Implementation of center-pivot 4D convolution """

import torch
import torch.nn as nn


class CenterPivotConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size() #1,3,13,13,13,13
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)#tensor([ 0,  2,  4,  6,  8, 10, 12])
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)#tensor([ 0,  2,  4,  6,  8, 10, 12])
            self.len_h = len(idxh)#7
            self.len_w = len(idxw)#7
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)
        #torch.Size([1, 3, 13, 13, 7, 7])
        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()#torch.Size([1, 3, 13, 13, 7, 7])
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)#torch.Size([49, 3, 13, 13])
        out1 = self.conv1(out1)#torch.Size([49, 16, 13, 13])
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)#16,13,13
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()#torch.Size([1, 3, 13, 13, 13, 13])
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)#torch.Size([169, 3, 13, 13])
        out2 = self.conv2(out2)#torch.Size([169, 16, 7, 7])
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()#torch.Size([1, 16, 13, 13, 7, 7])

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)#torch.Size([1, 16, 7, 7, 4, 4])
            out2 = out2.squeeze()#torch.Size([1, 16, 7, 7, 4, 4])

        y = out1 + out2 #torch.Size([1, 16, 13, 13, 7, 7])
        return y


# import torch
# import torch.nn as nn
#
#
# class CenterPivotConv4d(nn.Module):
#     """ CenterPivot 4D conv """
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
#         super(CenterPivotConv4d, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2],
#                                stride=stride[:2], bias=bias, padding=padding[:2])
#
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:],
#                                stride=stride[2:], bias=bias, padding=padding[2:])
#
#         self.stride34 = stride[2:]
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#
#         # 用 buffer 记录 idx，确保可随 model.to(device) 自动移动
#         self.register_buffer("idx", None)
#         self.idx_initialized = False
#         self.len_h = None
#         self.len_w = None
#
#     def prune(self, ct):
#
#         bsz, ch, ha, wa, hb, wb = ct.size()
#
#         # 第一次 forward 时才初始化 idx
#         if (not self.idx_initialized) or (self.idx is None):
#
#             # 生成 idxh / idxw （自动在 ct.device 上）
#             idxh = torch.arange(0, hb, self.stride[2], device=ct.device)
#             idxw = torch.arange(0, wb, self.stride[3], device=ct.device)
#
#             self.len_h = len(idxh)
#             self.len_w = len(idxw)
#
#             # 生成 idx（注意保持在 ct.device上）
#             idx = (idxw.repeat(self.len_h, 1) +
#                    idxh.repeat(self.len_w, 1).t() * wb).reshape(-1)
#
#             # 更新 buffer：保持 device 同步
#             self.idx = idx
#             self.idx_initialized = True
#
#         # 保证 idx 在正确的 device（虽然 register_buffer 通常会自动处理）
#         idx = self.idx.to(ct.device)
#
#         ct_pruned = (
#             ct.view(bsz, ch, ha, wa, -1)
#               .index_select(4, idx)
#               .view(bsz, ch, ha, wa, self.len_h, self.len_w)
#         )
#
#         return ct_pruned
#
#     def forward(self, x):
#
#         # stride#3/#4 大于 1 时执行 prune
#         if self.stride[2] > 1 or self.stride[3] > 1:
#             out1 = self.prune(x)
#         else:
#             out1 = x
#
#         bsz, inch, ha, wa, hb, wb = out1.size()
#         out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
#         out1 = self.conv1(out1)
#
#         outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
#         out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(
#             0, 3, 4, 5, 1, 2).contiguous()
#
#         # 第二个分支 out2
#         bsz, inch, ha, wa, hb, wb = x.size()
#         out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
#         out2 = self.conv2(out2)
#
#         outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
#         out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(
#             0, 3, 1, 2, 4, 5).contiguous()
#
#         # shape mismatch 修复
#         if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
#             out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
#             out2 = out2.squeeze()
#
#         y = out1 + out2
#         return y