# FLambas



### Installation
```bash
pip install -r requirements.txt
```

## Train and evaluate
```
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model Mamba_fft \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 1024 \
  --itr 1 \
  --learning_rate 0.00005 \
  --i_or_cos 1 \
  --base 96 \
  --gpu 0
```
