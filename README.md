# Bootstrapped MAE Documentation

This is the repository containing the implementation of the Bootstrapped MAE algorithm for the coding assignment. Results (both checkpoints and tensorboard log outputs) from our runs are saved in the `results` folder. To view the tensorboard log outputs, just type `$ tensorboard --logdir /results` (or the full path to the results if this doesn't work) into the command line and click the link that is printed out. If you want to reproduce the results shown in the report, please follow the instructions below.

## Table of Contents

- [Bootstrapped MAE Documentation](#bootstrapped-mae-documentation)
  - [Table of Contents](#table-of-contents)
  - [Dependencies](#dependencies)
  - [Baseline MAE Algorithm](#baseline-mae-algorithm)
  - [Bootstrapped MAE Algorithm](#bootstrapped-mae-algorithm)
  - [Bootstrapped MAE + EMA Algorithm](#bootstrapped-mae--ema-algorithm)

## Dependencies

This repository is based off of the original MAE codebase [https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae) and thus has the same dependency requirements as the original code base. We list the dependencies again below for convenience:

- Torch 1.7.0
- Torchvision 0.8.1
- Timm 0.3.2
- Tensorboard (any version)

## Baseline MAE Algorithm

To run the pretraining, fine-tuning, and linear-probing on the baseline MAE algorithm, please first change directory to the `bash_files` directory, and then run the following commands (in order) in the command line:

```
$ chmod +x mae_train.sh mae_eval_linear.sh mae_eval_finetune.sh
$ ./mae_train.sh
$ ./mae_eval_finetune.sh
$ ./mae_eval_linear.sh
```

Once these commands are run in the command line, the model will start training/fine-tuning/linear-probing. Its progress can be viewed using tensorboard by using the following command:

```
tensorboard --logdir {your_directory_path}
```

where `your_directory_path` is the path of the directory containing all the results. Then click the link that is printed out in the command line to view the tensorboard outputs in your browser.

## Bootstrapped MAE Algorithm

To run the pretraining, fine-tuning, and linear-probing on the bootstrapped MAE algorithm (i.e. MAE-1, MAE-2, ... algorithm), first change directory to the `bash_files` directory, and then run the following commands in the command line:

```
$ chmod +x bmaeiter_train.sh bmaeiter_eval_finetune.sh bmaeiter_eval_linprobe.sh
$ ./bmaeiter_train.sh
$ ./bmaeiter_eval_finetune.sh
$ ./bmaeiter_linear.sh
```

Similar to the baseline MAE algorithm, the results can be tracked real-time using tensorboard by running the following command:

```
tensorboard --logdir {your_directory_path}
```

## Bootstrapped MAE + EMA Algorithm

The steps to running the pretraining, fine-tuning, and linear-probing on the bootstrapped MAE + EMA algorithm is very similar to the baseline MAE algorithm. First change directory to the `bash_files` directory, and then run the following commands in the command line:

```
$ chmod +x Bmae_train.sh Bmae_eval_finetune.sh Bmae_eval_linear.sh
$ ./Bmae_train.sh
$ ./Bmae_eval_finetune.sh
$ ./Bmae_linear.sh
```

Similar to the baseline MAE algorithm, the results can be tracked real-time using tensorboard by running the following command:

```
tensorboard --logdir {your_directory_path}
```