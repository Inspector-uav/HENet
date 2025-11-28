import argparse
import torch.optim as optim
import torch.nn as nn
import torch

from model.HENet import HENet
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from common import dataset_mask_train, dataset_mask_val
from common.vis import Visualizer

from model.Loss import CombinedLoss


def train(epoch, model, dataloader, optimizer, sub_list, training, shot, loss_function):
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(sub_list)

    if (shot > 1) and (not training):
        for idx, (input, input_th, input_d, target, s_input, s_input_th, s_input_d, s_mask, subcls, index) in enumerate(
                dataloader):

            logit_mask_agg = 0
            target = target.squeeze(1)
            s_input_th = s_input_th.cuda(non_blocking=True)
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            input_th = input_th.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            subcls = subcls.cuda(non_blocking=True)
            history = history.cuda(non_blocking=True)
            for s_idx in range(shot):
                print(s_idx)
                logit_mask, _, _ = model(input, input_th, s_input[:, s_idx, :, :, :], s_input_th[:, s_idx, :, :, :],
                                         s_mask[:, s_idx, :, :], history)

                logit_mask_agg += logit_mask.argmax(dim=1).clone()

            bsz = logit_mask_agg.size(0)
            max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
            max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
            max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
            pred_mask = logit_mask_agg.float() / max_vote
            pred_mask[pred_mask < 0.5] = 0
            pred_mask[pred_mask >= 0.5] = 1

            for j in range(s_mask.shape[0]):
                sub_index = index[j]
                dataloader_val.history_mask_list[sub_index] = pred_mask[j].data.cpu()

            area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), target)
            average_meter.update(area_inter, area_union, subcls, loss=None)
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=1)


    else:
        for idx, (input, input_th, input_d, target, s_input, s_input_th, s_input_d, s_mask, subcls, index) in enumerate(
                dataloader):
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

            if training:
                for j in range(s_mask.shape[0]):
                    sub_index = index[j]
                    dataloader_trn.history_mask_list[sub_index] = pred_mask[j].data.cpu()
                loss1 = loss_function(logit_mask_agg, target)

                loss = loss1 + aux_loss1 + aux_loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                area_inter, area_union = Evaluator.classify_prediction(pred_mask, target)
                average_meter.update(area_inter, area_union, subcls, loss.detach().clone())
                average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=1)  # 每50个iteration显示一次结果

            else:
                for j in range(s_mask.shape[0]):
                    sub_index = index[j]
                    dataloader_val.history_mask_list[sub_index] = pred_mask[j].data.cpu()
                area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), target)
                average_meter.update(area_inter, area_union, subcls, loss=None)
                average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=1)

    average_meter.write_result('Training' if training else 'Validation', epoch)  # 负责输出一个epoch的结果
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='./dataset/train')
    parser.add_argument('--benchmark', type=str, default='pascal')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', action='store_false')
    parser.add_argument('--train_h', type=int, default=400)
    parser.add_argument('--train_w', type=int, default=400)
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    model = HENet(args.backbone, False)
    Logger.log_params(model)

    if args.fold == 3:
        sub_list = list(range(0, 12))
        sub_val_list = list(range(12, 16))
    elif args.fold == 2:
        sub_list = list(range(0, 8)) + list(range(12, 16))
        sub_val_list = list(range(8, 12))
    elif args.fold == 1:
        sub_list = list(range(0, 4)) + list(range(8, 16))
        sub_val_list = list(range(4, 8))
    elif args.fold == 0:
        sub_list = list(range(4, 16))
        sub_val_list = list(range(0, 4))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    loss_function = CombinedLoss()

    optimizer = optim.Adam([
        {"params": model.parameters(), "lr": args.lr},
        {"params": loss_function.parameters(), "lr": args.lr * 0.01}
    ])
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    dataloader_trn = dataset_mask_train.Dataset(data_dir=args.datapath, fold=args.fold,
                                                input_size=(args.train_h, args.train_w), normalize_mean=mean,
                                                normalize_std=std)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataloader_trn, batch_size=args.bsz, shuffle=(train_sampler is None),
                                               num_workers=args.nworker, pin_memory=False, sampler=train_sampler,
                                               drop_last=True)

    dataloader_val = dataset_mask_val.Dataset(data_dir=args.datapath, fold=args.fold,
                                              input_size=(args.train_h, args.train_w), normalize_mean=mean,
                                              normalize_std=std, shot=args.shot)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(dataloader_val, batch_size=1, shuffle=False,
                                             num_workers=args.nworker, pin_memory=False, sampler=val_sampler)

    best_val_miou = float('-inf')
    best_val_fb_miou = float('-inf')
    best_val_loss = float('inf')

    best_train_miou = float('-inf')
    best_train_fb_miou = float('-inf')
    best_train_loss = float('inf')

    best_epoch = float('inf')
    for epoch in range(args.niter):

        print(
            f"[Epoch {epoch}] BCE Weight: {loss_function.bce_weight.item():.6f}, IOU Weight: {loss_function.iou_weight.item():.6f}")

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, train_loader, optimizer, sub_list, training=True, shot=5, loss_function=loss_function)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, val_loader, optimizer, sub_val_list, training=False,
                                                   shot=args.shot)

        if trn_miou > best_train_miou:
            best_train_miou = trn_miou
            best_train_fb_miou = trn_fb_iou
            best_epoch = epoch
            Logger.save_model_miou(model, best_epoch, trn_miou)

        print("%d epoch ,%.2f is the best M-iou, %.2f is the best FB-iou" % (
            best_epoch, best_train_miou, best_train_fb_miou))

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou}, epoch)
        Logger.tbd_writer.flush()

    Logger.info('==================== Finished Training ====================')
