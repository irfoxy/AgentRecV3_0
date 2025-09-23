# export PYTHONPATH=/home/icfoxy/projects/AgentRecV2_0/model:$PYTHONPATH
# 在包含 model/SASRec.py 的工程根目录运行：
python ../train.py \
  --data_dir ../data/ml100k/processed \
  --train_file train.csv \
  --save_dir ../model/saved \
  --hidden_units 256 \
  --num_blocks 3 \
  --num_heads 4 \
  --dropout_rate 0.1 \
  --maxlen 50 \
  --batch_size 128 \
  --epochs 100 \
  --lr 5e-4 \
  --norm_first