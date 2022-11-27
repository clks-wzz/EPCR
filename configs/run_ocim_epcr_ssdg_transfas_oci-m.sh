#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

TRAIN_DIR=./lists/list_semi_ocim_train_SSDG_OCI-M.txt
TEST_DIR=./lists/list_semi_ocim_test_OCI-M.txt
MODEL_DIR=./checkpoints/epcr_ssdg_transfas_ocim_oci-m/

python main_ocim_cdcn_ssdg_fp16.py \
--amp \
--dataset ocim_ssdg_sparse_uniform \
--data_dir ${TRAIN_DIR} \
--test_dir ${TEST_DIR} \
--image_size 224 \
--model simsiam_semi_cdcn_meanteacherV11_ssdg_fp16 \
--proj_layers 2 \
--backbone transfas \
--optimizer sgd \
--weight_decay 0.0005 \
--momentum 0.9 \
--warmup_epoch 0 \
--warmup_lr 0 \
--base_lr 0.03 \
--final_lr 0.01 \
--num_epochs 800 \
--stop_at_epoch 800 \
--batch_size 36 \
--num_workers 16 \
--head_tail_accuracy \
--hide_progress \
--print_freq 10 \
--output_dir ${MODEL_DIR} \
# --debug












