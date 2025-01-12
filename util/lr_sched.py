# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# Cosine scheduler that will be used for weight decay
def weight_scheduler(base_value, final_value, epochs, niter_per_epoch, args,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = args.warmup_epochs * niter_per_epoch
    print("Set warmup steps = %d" % warmup_iters)
    if args.warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    # Formula from pytorch https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    iters = np.arange(epochs * niter_per_epoch - warmup_iters)
    schedule = np.array([final_value+0.5*(base_value-final_value)*(1+math.cos(math.pi*i/(len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_epoch
    return schedule