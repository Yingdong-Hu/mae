# Run the desired python file, use default parameters from file
# Output and log dirs can be changed by changing --output_dir and --log_dir
python3 ../02_main_linprobe.py --data_path ../datasets/ --batch_size 64 --epochs 100 --distributed --output_dir ../results/output_mae_linprobe --log_dir ../results/output_mae_linprobe --finetune ../results/output_mae_pretrain/checkpoint-199.pth