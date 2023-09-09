# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import random
import sys
from typing import Iterable

import torch
import torch.nn as nn

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device==torch.cuda:
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Helper function to adjust the ema momentum
def adjust_ema_momentum(epoch, args):
    if epoch<100:
        ret_decay = args.ema_decay + epoch/100 * (0.9999 - args.ema_decay)
    else:
        ret_decay = 0.9999 + min(300, epoch-100)/300 * (0.00009)
    
    return ret_decay
    
# Helper function to turn masked areas into indices
def mask_to_index(masks):
    bz = masks.shape[0]
    mask1 = 1 - masks.float()
    mask1 += torch.rand(masks.shape).cuda()
    idx_shuffle = torch.argsort(1-mask1.view(bz, -1), dim=1)
    rand_roll = random.randint(0,3)
    idx_shuffle = idx_shuffle.roll(rand_roll, dims=1)
    return idx_shuffle

def bmae_train_one_epoch(model: torch.nn.Module,
                         data_loader: Iterable, optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, loss_scaler,
                         model_ema,
                         ntrain_steps_per_epoch,
                         weight_schedule,
                         log_writer=None,
                         args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('decay', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    patch_size=4
    print_freq = 50

    accum_iter = args.accum_iter
    win_size = args.input_size//patch_size
    seq_len = win_size**2
    mask_len = args.mask_ratio * (args.input_size//patch_size)**2
    

    optimizer.zero_grad()
    
    # model_ema.decay = adjust_ema_momentum(epoch, args)
    metric_logger.update(decay=model_ema.decay)
    print("EMA decay value: ", model_ema.decay)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            
            iteration=epoch*ntrain_steps_per_epoch + data_iter_step
            for param_group in enumerate(optimizer.param_groups):
                if 'weight_decay' in param_group:
                    param_group['weight_decay'] = weight_schedule[iteration]

        samples = samples.to(device, non_blocking=True)

        # 192 is the encoder dimension for the tiny deit
        temp_len = int(mask_len)
        feature_weight = args.feature_weight * (1+epoch)/args.epochs
        feature_weight = min(feature_weight*4, args.feature_weight)
        bz, _, H, W = samples.shape
        LN = nn.LayerNorm(192, eps=1e-6, elementwise_affine=False).cuda()
        
        with torch.cuda.amp.autocast():
            loss, pred, feature_pred, mask = model(samples)
            
            with torch.no_grad():
                ids_shuffle = mask_to_index(mask)
                ids_unshuffle = torch.argsort(ids_shuffle, dim=1)
                
                loss_mask = [0]*(seq_len-temp_len) + [1]*temp_len
                loss_mask = torch.Tensor(loss_mask).reshape(1, seq_len).cuda().repeat(bz,1)
                loss_mask = torch.gather(loss_mask, dim=1, index = ids_unshuffle)
                loss_mask = loss_mask.reshape(-1, 1, win_size, win_size).to(torch.float)
                loss_mask = loss_mask.reshape(-1, 1, win_size, win_size).to(torch.float)
                
                weight_mask = torch.nn.functional.interpolate(loss_mask, (H, W), mode='nearest')
                perc_mask = loss_mask.reshape(-1, seq_len, 1)
                
            with torch.no_grad():
                assert model_ema is not None
                feature_model = model_ema.ema
                feature_model.eval()
                feature_gt = feature_model.get_feature(samples)
                assert feature_gt.shape == feature_pred.shape
                
                rec_loss = torch.mean(perc_mask * ((LN(feature_gt) - LN(feature_pred))**2))/torch.mean(loss_mask)

            loss = loss + feature_weight*rec_loss
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            model_ema.update(model)

        if device==torch.cuda:
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        # lr = 2.5e-4 # seems like the best lr
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("feature weight: ", feature_weight)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}