#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

TRAIN_DIR=./lists/list_semi_ocim_train_OCM-I.txt.hollow.C
TEST_DIR=./lists/list_semi_ocim_test_OCM-I.txt
MODEL_DIR=./checkpoints/epcr_leave_one_om-i/

python main_ocim_cdcn.py \
--dataset ocim \
--data_dir ${TRAIN_DIR} \
--test_dir ${TEST_DIR} \
--image_size 256 \
--model simsiam_semi_cdcn_meanteacherV11 \
--proj_layers 2 \
--backbone cdcn \
--optimizer sgd \
--weight_decay 0.0005 \
--momentum 0.9 \
--warmup_epoch 0 \
--warmup_lr 0 \
--base_lr 0.03 \
--final_lr 0.01 \
--num_epochs 800 \
--stop_at_epoch 800 \
--batch_size 64 \
--num_workers 32 \
--head_tail_accuracy \
--hide_progress \
--print_freq 10 \
--output_dir ${MODEL_DIR} \
# --debug












