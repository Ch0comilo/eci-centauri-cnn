#!/bin/bash

set -e

MODEL=AstroCNNModel
CFG=final_alpha_1_tuned
NAME=fa1t

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=${MODEL} \
        --config_name=${CFG} \
        --train_files='/home/astronet/tfrecords-new+old/train/*' \
        --eval_files='/home/astronet/tfrecords-new+old/test/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/${NAME}_38_run_2/${i}"
done

for i in {1..10}
do
    echo "Training Y1 model ${i}"
    python astronet/train.py \
        --model=${MODEL} \
        --config_name=${CFG} \
        --train_files='/home/astronet/tfrecords-new+old/train/*' \
        --eval_files='/home/astronet/tfrecords-new+old/test/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/${NAME}_38_y1_run_2/${i}"
done

for i in {1..10}
do
    echo "Training Y2 model ${i}"
    python astronet/train.py \
        --model=${MODEL} \
        --config_name=${CFG} \
        --train_files='/home/astronet/tfrecords-new+old/train/*' \
        --eval_files='/home/astronet/tfrecords-new+old/test/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/${NAME}_38_y2_run_2/${i}"
done

for i in {1..10}
do
    echo "Training Y3 model ${i}"
    python astronet/train.py \
        --model=${MODEL} \
        --config_name=${CFG} \
        --train_files='/home/astronet/tfrecords-new+old/train/*' \
        --eval_files='/home/astronet/tfrecords-new+old/test/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/${NAME}_38_y3_run_2/${i}"
done

