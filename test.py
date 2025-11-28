import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2
from model.HENet import HENet
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from common import dataset_mask_test
from model.Loss import CombinedLoss

def masking(img, mask, color=None, alph=0.5):
    if color is None:
        color = [0, 0, 255]
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask == 255] = color
    out = cv2.addWeighted(img_layer, alph, out, 1 - alph, 0, out)
    return out


def visulize(path_rgbq, target, q_pred, i):
    target = target.float()

    q_image = cv2.imread(path_rgbq[0])
    q_h_ori = np.size(q_image, 0)
    q_w_ori = np.size(q_image, 1)

    q_pred = F.interpolate(q_pred.unsqueeze(0).float(), size=(q_h_ori, q_w_ori), mode='bilinear',
                           align_corners=True).squeeze(0)

    target = F.interpolate(target.unsqueeze(0).float(), size=(q_h_ori, q_w_ori), mode='bilinear',
                           align_corners=True).squeeze(0)

    q_mask = target.squeeze().numpy()

    q_mask_rgb = q_mask
    q_mask_rgb = q_mask_rgb[:q_h_ori, :q_w_ori] * 255
    q_mask_rgb = q_mask_rgb.astype(np.uint8)
    q_mask_rgb = masking(q_image.copy(), q_mask_rgb, color=[0, 128, 255])
    path1 = './visual_maps/' + str(i + 1) + '.png'
    cv2.imwrite(path1, q_mask_rgb)

    q_pre_mask = q_pred.squeeze().cpu().numpy()
    q_pre_mask = (q_pre_mask[:q_h_ori, :q_w_ori] * 255).astype(np.uint8)
    path_pred = './visual_maps/' + str(i + 1) + '.png'
    cv2.imwrite(path_pred, q_pre_mask)

    q_pre_mask_rgb = masking(q_image.copy(), q_pre_mask, color=[0, 0, 255])
    path4 = './visual_maps/' + str(i + 1) + '.png'
    cv2.imwrite(path4, q_pre_mask_rgb)


def segmentation(model, dataloader, list):

    utils.fix_randseed(1)
    average_meter = AverageMeter(list)
    loss_function = CombinedLoss()
    for idx, (input, input_th, input_d, target, s_input, s_input_th, s_input_d, s_mask, subcls, index,
              paths_rgb, paths_th, pathq_rgb, pathq_th) in enumerate(dataloader):

        s_mask1 = s_mask.squeeze(1)
        target = target.squeeze(1)

        s_input1 = s_input.squeeze(1)
        s_input1 = s_input1.cuda(non_blocking=True)

        s_input_th1 = s_input_th.squeeze(1)
        s_input_th1 = s_input_th1.cuda(non_blocking=True)

        s_input_d1 = s_input_d.squeeze(1)
        s_input_d1 = s_input_d1.cuda(non_blocking=True)

        s_mask1 = s_mask1.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        input_th = input_th.cuda(non_blocking=True)
        input_d = input_d.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        subcls = subcls.cuda(non_blocking=True)
        logit_mask_agg, aux_loss1, aux_loss2 = model(input, input_th, input_d, s_input1, s_input_th1, s_input_d1,
                                                     s_mask1)

        pred_mask = logit_mask_agg.argmax(dim=1)

        visulize(pathq_rgb, target.cpu(), pred_mask.cpu(), idx)
        loss1 = loss_function(logit_mask_agg, target)
        loss = loss1 + aux_loss1 + aux_loss2

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, target)
        average_meter.update(area_inter, area_union, subcls, loss.detach().clone())

    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='./dataset/test')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='./logs/')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_original_imgsize', action='store_true')
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    model = HENet(args.backbone, args.use_original_imgsize)
    model.eval()
    Logger.log_params(model)

    if args.fold == 3:
        sub_list = list(range(0, 15))
        sub_val_list = list(range(15, 20))
    elif args.fold == 2:
        sub_list = list(range(0, 10)) + list(range(15, 20))
        sub_val_list = list(range(10, 15))
    elif args.fold == 1:
        sub_list = list(range(0, 5)) + list(range(10, 20))
        sub_val_list = list(range(5, 10))
    elif args.fold == 0:
        sub_list = list(range(5, 20))
        sub_test_list = list(range(0, 1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))

    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    dataloader_test = dataset_mask_test.Dataset(data_dir=args.datapath, fold=args.fold)
    test_sampler = None
    test_loader = torch.utils.data.DataLoader(dataloader_test, batch_size=1, shuffle=False,
                                              num_workers=args.nworker, pin_memory=True, sampler=test_sampler)

    with torch.no_grad():
        test_miou, test_fb_iou = segmentation(model, test_loader, sub_test_list)
    Logger.info('test: mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Segmentation Testing ====================')
