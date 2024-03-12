import os
if __name__ == '__main__':
    # os.system("python pretrain.py --output_dir=./saved_models --model_name=pretrain.bin --do_train --train_data_file=./dataset/train.jsonl --eval_data_file=./dataset/valid.jsonl --test_data_file=./dataset/test.jsonl --epochs 75 --encoder_block_size 512 --decoder_block_size 256 --train_batch_size 8 --eval_batch_size 8 --learning_rate 2e-5 --max_grad_norm 1.0 --n_gpu 1 --evaluate_during_training --seed 123456  2>&1 | tee train.log")
    os.system(
        "python finetune_awi_alarm.py --load_model_from_checkpoint --checkpoint_model_name pretrain.bin --output_dir=./saved_models --model_name=awi.bin --do_train --train_data_file=./dataset/awi_dataset/awi_train.jsonl --eval_data_file=./dataset/awi_dataset/awi_val.jsonl --test_data_file=./dataset/awi_dataset/awi_test.jsonl --epochs 75 --encoder_block_size 512 --decoder_block_size 256 --train_batch_size 8 --eval_batch_size 8 --learning_rate 2e-5 --max_grad_norm 1.0 --n_gpu 1 --evaluate_during_training --seed 123456  2>&1 | tee train.log")