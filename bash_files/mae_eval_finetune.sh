#  Run the desired python file, use default parameters from file
# Output and log dirs can be changed by changing --output_dir and --log_dir
python3 ../01_main_finetune.py --data_path ../datasets/ --batch_size 64 --epochs 100 --distributed --output_dir ../results/output_mae_finetune --log_dir ../results/output_mae_finetune --finetune ../results/output_mae_pretrain/checkpoint-199.pth