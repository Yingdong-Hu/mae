#  Run the desired python file, use default parameters from file
# Output and log dirs can be changed by changing --output_dir and --log_dir

python3 ../03_bmae_pretrain.py --data_path ../datasets/ --batch_size 64 --epochs 200 --distributed --output_dir ../results/output_bmae_pretrain --log_dir ../results/output_bmae_pretrain