from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from model.base.feature import extract_feat_vgg, extract_feat_res
from model.base.correlation import Correlation

from model.learner import HPNLearner
from model.HEmodel import HEM
from model.FIFmodel import FIFM
from model.CRM import NonBottleneck
import numpy as np


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class HENet(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(HENet, self).__init__()

        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize  # false
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()

        self.hem = HEM(64, 64)
        self.fifm = FIFM(256, 256)

        self.attn1 = NonBottleneck(256, 256)

        reduce_dim = 256
        self.down_sample = nn.Sequential(
            nn.Conv2d(1024 + 512, reduce_dim, kernel_size=1, padding=0, bias=False),  # 1024+512->256
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.init_merge = nn.Sequential(
            nn.Conv2d(577, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))

        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))  # 3,6,4
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.res2_meta = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                                       nn.ReLU(inplace=True))
        self.decoder2 = nn.Sequential(nn.Conv2d(256, 64, (3, 3), padding=(1, 1), bias=True),  # 256
                                      nn.ReLU(),
                                      nn.Dropout2d(p=0.1),
                                      nn.Conv2d(64, 2, (3, 3), padding=(1, 1), bias=True))

    def forward(self, query_img, query_img_th, query_img_d, support_img, support_img_th, support_img_d, support_mask):
        supp_pro_list = []
        final_supp_list = []

        with torch.no_grad():
            query_feats, query_backbone_layers = self.extract_feats(query_img, self.backbone, self.feat_ids,
                                                                    self.bottleneck_ids, self.lids)

            if True:
                query_backbone_layers[3] = F.interpolate(query_backbone_layers[3],
                                                         size=(query_backbone_layers[2].size(2),
                                                               query_backbone_layers[2].size(3)),
                                                         mode='bilinear', align_corners=True)
                query_backbone_layers[4] = F.interpolate(query_backbone_layers[4],
                                                         size=(query_backbone_layers[4].size(2),
                                                               query_backbone_layers[4].size(3)),
                                                         mode='bilinear', align_corners=True)

                query_feat0 = torch.cat([query_backbone_layers[3], query_backbone_layers[2]], 1)

            query_feat0 = self.down_sample(query_feat0)
            quy = self.attn1(query_feat0)

            ########################################################################################

            # RGB特征
            support_feats, support_backbone_layers = self.extract_feats(support_img, self.backbone, self.feat_ids,
                                                                        self.bottleneck_ids, self.lids)
            final_supp_list.append(support_backbone_layers[4])

            support_feats = self.mask_feature(support_feats, support_mask.clone())

            support_layer3 = F.interpolate(support_feats[9],
                                           size=(support_feats[3].size(2), support_feats[3].size(3)),
                                           mode='bilinear', align_corners=True)
            supp_feat = torch.cat([support_layer3, support_feats[3]], 1)

            mask_down = F.interpolate(support_mask.float().unsqueeze(1),
                                      size=(support_feats[3].size(2), support_feats[3].size(3)),
                                      mode='bilinear', align_corners=True)
            supp_feat = self.down_sample(supp_feat)
            supp_pro = Weighted_GAP(supp_feat, mask_down)
            supp_pro_list.append(supp_pro)

            # T特征
            query_feats_th, query_backbone_layers_th = self.extract_feats(query_img_th, self.backbone, self.feat_ids,
                                                                          self.bottleneck_ids, self.lids)
            support_feats_th, support_backbone_layers_th = self.extract_feats(support_img_th, self.backbone,
                                                                              self.feat_ids, self.bottleneck_ids,
                                                                              self.lids)
            support_feats_th = self.mask_feature(support_feats_th, support_mask.clone())

            support_layer3_th = F.interpolate(support_feats_th[9],
                                              size=(support_feats_th[3].size(2), support_feats_th[3].size(3)),
                                              mode='bilinear', align_corners=True)
            supp_feat_th = torch.cat([support_layer3_th, support_feats_th[3]], 1)

            supp_feat_th = self.down_sample(supp_feat_th)
            supp_pro_th = Weighted_GAP(supp_feat_th, mask_down)

            # D特征
            query_feats_d, query_backbone_layers_d = self.extract_feats(query_img_d, self.backbone, self.feat_ids,
                                                                        self.bottleneck_ids, self.lids)
            support_feats_d, support_backbone_layers_d = self.extract_feats(support_img_d, self.backbone,
                                                                            self.feat_ids, self.bottleneck_ids,
                                                                            self.lids)
            support_feats_d = self.mask_feature(support_feats_d, support_mask.clone())

            support_layer3_d = F.interpolate(support_feats_d[9],
                                             size=(support_feats_d[3].size(2), support_feats_d[3].size(3)),
                                             mode='bilinear', align_corners=True)
            supp_feat_d = torch.cat([support_layer3_d, support_feats_d[3]], 1)

            supp_feat_d = self.down_sample(supp_feat_d)
            supp_pro_d = Weighted_GAP(supp_feat_d, mask_down)

            t = (supp_pro - supp_pro_th) ** 2
            t1 = t.view(4, -1)
            tmp = torch.sum(t1).cpu()
            tmp1 = np.sqrt(tmp)
            tmp2 = tmp1.cuda(0)
            aux_loss1 = torch.zeros_like(tmp2).cuda()
            aux_loss1 = aux_loss1 + tmp2

            d = (supp_pro - supp_pro_d) ** 2
            d1 = d.view(4, -1)
            dmp = torch.sum(d1).cpu()
            dmp1 = np.sqrt(dmp)
            dmp2 = dmp1.cuda(0)
            aux_loss2 = torch.zeros_like(dmp2).cuda()
            aux_loss2 = aux_loss2 + dmp2

            corr_query_mask_list = []
            cosine_eps = 1e-7
            for i, tmp_supp_feat in enumerate(final_supp_list):
                resize_size = tmp_supp_feat.size(2)

                tmp_mask = F.interpolate(support_mask.unsqueeze(1).float(), size=(resize_size, resize_size),
                                         mode='bilinear', align_corners=True)

                tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                q = query_backbone_layers[4]
                s = tmp_supp_feat_4
                bsize, ch_sz, sp_sz, _ = q.size()[:]

                tmp_query = q
                tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
                tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                tmp_supp = s
                tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
                tmp_supp = tmp_supp.permute(0, 2, 1)
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / \
                             (similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)

                corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)

                corr_query = F.interpolate(corr_query,
                                           size=(
                                               query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                           mode='bilinear', align_corners=True)

                corr_query_mask_list.append(corr_query)
            corr_query_mask = torch.cat(corr_query_mask_list, 1)
            supp_pro = torch.cat(supp_pro_list, 2)

            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
            corr_th = Correlation.multilayer_correlation(query_feats_th, support_feats_th, self.stack_ids)
            corr_d = Correlation.multilayer_correlation(query_feats_d, support_feats_d, self.stack_ids)

        logit_mask_r = self.hpn_learner(corr)
        logit_mask_d = self.hpn_learner(corr_d)
        logit_mask_th = self.hpn_learner(corr_th)

        logit_mask_th = self.hem(logit_mask_th, logit_mask_r)
        logit_mask_d = self.hem(logit_mask_d, logit_mask_r)

        concat_feat = supp_pro.expand_as(quy)
        merge_feat = torch.cat([quy, concat_feat, corr_query_mask], 1)

        merge_feat = F.interpolate(merge_feat, size=(logit_mask_th.size(2), logit_mask_th.size(3)), mode='bilinear',
                                   align_corners=True)

        merge_feat_th = torch.cat([merge_feat, logit_mask_th], 1)
        merge_feat_d = torch.cat([merge_feat, logit_mask_d], 1)

        merge_th = self.init_merge(merge_feat_th)
        merge_d = self.init_merge(merge_feat_d)

        query_meta_th = self.res2_meta(merge_th) + merge_th
        query_meta_d = self.res2_meta(merge_d) + merge_d

        query_meta = self.fifm(query_meta_th, query_meta_d)

        logit_mask = self.decoder2(query_meta)

        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask, aux_loss1, aux_loss2

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear',
                                 align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):

        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        bsz = logit_mask_agg.size(0)  # 1
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]  # tensor([5])
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])  # tensor([[5], [1]]),torch.Size([2, 1])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)  # torch.Size([1, 1, 1])
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        if self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, size=gt_mask.size()[1:], mode='bilinear', align_corners=True)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.train()