#!/bin/bash

# Pretrained network to load and finetune
PRETRAINED=${1}

# To find PRETRAINED_ARGS, we first find the base directory of the pretrained model 
# in PRETRAINED path, and then we append 'args.json'
PRETRAINED_ARGS=${PRETRAINED%/*}/args.json

# Dataset to be used
DATASET=${2:-colored_mnist}
# Dataset difficulty (C if sMPI3D, else percentage of bias-conflicting samples)
PCT=${3:-"1"}

# Validation score to monitor and model selection
# 'val_0' corresponds to the validation dataset used (0 is the iid one)
# 'u_balanced' corresponds to bias-unsupervised group-balanced metrics
# 's_balanced' would be bias-supervised group-balanced metrics
# 'acc1' corresponds to top1 accuracy of the target task
VAL_METRIC=${4:-"val_0/u_balanced/acc1"}
# Save path for the model and logs
ROOT_PATH=${5:-"./experiments"}
# Path to the dataset
DATA_PATH=${6:-"./datasets/$DATASET"}

# Target variables for downstream online classification
# For all datasets, except mpi3d, target variable is 0 label and bias is 1
# Last label, corresponds to the target task...
TASK='1 0'

OPTIMIZER=adamw
BATCH_SIZE=256
METHOD=ula
BACKBONE=resnet18
SCHEDULER="warmup_cosine --scheduler_interval step --warmup_epochs 0 --warmup_start_lr 0.0"


# for i in `seq 5`; do
i=1

# Defaults are going to be overwritten according to the dataset
if [[ $DATASET == "mpi3d" ]]; then
  PCT=${3:-3}
  TRAIN_DATA_PATH="$DATA_PATH/${DATASET}_real_train_K${PCT}_seed${i}.npz"
  VAL_DATA_PATH="$DATA_PATH/${DATASET}_real_valid_K${PCT}_seed${i}.npz $DATA_PATH/${DATASET}_real_test_K${PCT}_seed${i}.npz"
  TEST_DATA_PATH="$DATA_PATH/${DATASET}_real_test_K${PCT}_seed${i}.npz $DATA_PATH/${DATASET}_real_ood_K${PCT}_seed${i}.npz"

  MAX_PRETRAIN_EPOCHS=50
  MAX_EPOCHS=150

  # Shape is task 1 in mpi3d
  TASK='0 1'
elif [[ $DATASET == "celeba" ]]; then
  TRAIN_DATA_PATH="$DATA_PATH"
  VAL_DATA_PATH="$DATA_PATH"
  TEST_DATA_PATH="$DATA_PATH"

  BACKBONE=resnet50

  MAX_PRETRAIN_EPOCHS=50
  MAX_EPOCHS=150
  BATCH_SIZE=256
  OPTIMIZER=adamw
elif [[ $DATASET == "waterbirds" ]]; then
  TRAIN_DATA_PATH="$DATA_PATH"
  VAL_DATA_PATH="$DATA_PATH"
  TEST_DATA_PATH="$DATA_PATH"

  BACKBONE=pre_resnet50

  MAX_PRETRAIN_EPOCHS=50
  MAX_EPOCHS=250
  BATCH_SIZE=64
  OPTIMIZER=sgd
  SCHEDULER="none"
elif [[ $DATASET == "colored_mnist" ]]; then
  TRAIN_DATA_PATH="$DATA_PATH/${PCT}pct/align_conflict.npz"
  VAL_DATA_PATH="$DATA_PATH/${PCT}pct/valid.npz $DATA_PATH/test.npz"
  TEST_DATA_PATH="$DATA_PATH/test.npz"

  BACKBONE=mlp

  MAX_PRETRAIN_EPOCHS=10
  MAX_EPOCHS=110
elif [[ $DATASET == "corrupted_cifar10" ]]; then
  TRAIN_DATA_PATH="$DATA_PATH/${PCT}pct/align_conflict.npz"
  VAL_DATA_PATH="$DATA_PATH/${PCT}pct/valid.npz $DATA_PATH/test.npz"
  TEST_DATA_PATH="$DATA_PATH/test.npz"

  MAX_PRETRAIN_EPOCHS=100
  MAX_EPOCHS=200
else
  TRAIN_DATA_PATH=$DATA_PATH
  VAL_DATA_PATH=$DATA_PATH
  TEST_DATA_PATH="$DATA_PATH"

  MAX_PRETRAIN_EPOCHS=100
  MAX_EPOCHS=100
fi


GROUP="baseline_${METHOD}_${DATASET}_K${PCT}_${BACKBONE}"
export WANDB_RUN_GROUP=$GROUP
export ORION_DB_ADDRESS="${ROOT_PATH}/${GROUP}/orion_db.pkl"

EXPNAME="test_${GROUP}"

## Uncomment the following two lines and specify hyperparameter priors
## if you want to perform a hyperparameter search
# EXPNAME="${GROUP}_{trial.hash_params}"
# orion -v hunt -n ${GROUP} --config=./orion_config.yaml \
python3 ./../main_train.py \
    --name ${EXPNAME} \
    --method ${METHOD} \
    --dataset ${DATASET} \
    --backbone ${BACKBONE} \
    --train_data_path $TRAIN_DATA_PATH \
    --valid_data_path $VAL_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --checkpoint_dir="${ROOT_PATH}/${GROUP}/${EXPNAME}" \
    --save_checkpoint \
    --auto_resume \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --num_workers 4 \
    --task $TASK \
    --model_selection_metric="${VAL_METRIC}" \
    --model_selection_mode="max" \
    --select_best_model \
    --max_epochs $MAX_EPOCHS \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --lr 1e-4 \
    --weight_decay 0.0 \
    --batch_size $BATCH_SIZE \
    --augment minimal \
    --student_temperature 1.0 \
    --teacher_temperature 1.0 \
    --gen0_max_steps -1 \
    --gen0_max_epochs $MAX_PRETRAIN_EPOCHS \
    --biased_network_state $PRETRAINED \
    --biased_network_args $PRETRAINED_ARGS \
    --train_mode finetune

# done
