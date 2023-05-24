#!/bin/bash

# Dataset to be used
DATASET=${1:-colored_mnist}
# Dataset difficulty (C if sMPI3D, else percentage of bias-conflicting samples)
PCT=${2:-"1"}

# Validation score to monitor and model selection
# 'val_0' corresponds to the validation dataset used (0 is the iid one)
# 'base' corresponds to features extracted from the backbone (no projection head)
# 'linear' for a linear probing classifier
# 'acc1_0' corresponds to top1 accuracy of the task 0
VAL_METRIC=${3:-"val_0/base/linear/acc1_0"}
# Save path for the model and logs
ROOT_PATH=${4:-"./experiments"}
# Path to the dataset
DATA_PATH=${5:-"./datasets/$DATASET"}

# Target variables for downstream online classification
# For all datasets, except mpi3d, target variable is 0 label and bias is 1
TASK='1 0'

QUEUE_SIZE=32768
OPTIMIZER=lars
SCHEDULER="warmup --scheduler_interval step --warmup_epochs 5 --warmup_start_lr 1e-5"
BATCH_SIZE=256
PROJ_OUT_DIM=256
MAX_EPOCHS=100
BACKBONE=resnet18
METHOD=mocov2plus

# Default augmentation parameters
MIN_SCALE=0.08
JITTER_PROB=0.8
GRAY_SCALE_PROB=0.2
GAUSSIAN_PROB=0.5
SOLARIZATION_PROB=0.0
FLIP_PROB=0.5


# for i in `seq 5`; do
i=1

# Defaults are going to be overwritten according to the dataset
if [[ $DATASET == "mpi3d" ]]; then
  PCT=${2:-3}
  # Shape is task 1 in mpi3d
  VAL_METRIC=${4:-"val_0/base/linear/acc1_1"}
  TRAIN_DATA_PATH="$DATA_PATH/${DATASET}_real_train_K${PCT}_seed${i}.npz"
  VAL_DATA_PATH="$DATA_PATH/${DATASET}_real_valid_K${PCT}_seed${i}.npz $DATA_PATH/${DATASET}_real_test_K${PCT}_seed${i}.npz"
  TEST_DATA_PATH="$DATA_PATH/${DATASET}_real_test_K${PCT}_seed${i}.npz $DATA_PATH/${DATASET}_real_ood_K${PCT}_seed${i}.npz"

  MAX_EPOCHS=100

  MIN_SCALE=0.5
elif [[ $DATASET == "celeba" ]]; then
  TRAIN_DATA_PATH="$DATA_PATH"
  VAL_DATA_PATH="$DATA_PATH"
  TEST_DATA_PATH="$DATA_PATH"

  BACKBONE=resnet50
  MAX_EPOCHS=100

  MIN_SCALE=0.5
  GRAY_SCALE_PROB=0.0
  GAUSSIAN_PROB=0.0
  SOLARIZATION_PROB=0.2
elif [[ $DATASET == "waterbirds" ]]; then
  TRAIN_DATA_PATH="$DATA_PATH"
  VAL_DATA_PATH="$DATA_PATH"
  TEST_DATA_PATH="$DATA_PATH"

  BACKBONE=resnet50
  # Dataset is small
  QUEUE_SIZE=4096

  MIN_SCALE=0.5
elif [[ $DATASET == "colored_mnist" ]]; then
  TRAIN_DATA_PATH="$DATA_PATH/${PCT}pct/align_conflict.npz"
  VAL_DATA_PATH="$DATA_PATH/${PCT}pct/valid.npz $DATA_PATH/test.npz"
  TEST_DATA_PATH="$DATA_PATH/test.npz"

  BACKBONE=mlp
  MAX_EPOCHS=100

  MIN_SCALE=0.7
  JITTER_PROB=0.0
  GRAY_SCALE_PROB=0.0
  GAUSSIAN_PROB=0.5
  FLIP_PROB=0.0
elif [[ $DATASET == "corrupted_cifar10" ]]; then
  TRAIN_DATA_PATH="$DATA_PATH/${PCT}pct/align_conflict.npz"
  VAL_DATA_PATH="$DATA_PATH/${PCT}pct/valid.npz $DATA_PATH/test.npz"
  TEST_DATA_PATH="$DATA_PATH/test.npz"
  MAX_EPOCHS=500
  MIN_SCALE=0.08
else
  TRAIN_DATA_PATH="$DATA_PATH"
  VAL_DATA_PATH="$DATA_PATH"
  TEST_DATA_PATH="$DATA_PATH"
fi


GROUP="baseline_${METHOD}_${DATASET}_K${PCT}_${BACKBONE}"
export WANDB_RUN_GROUP=$GROUP
export ORION_DB_ADDRESS="${ROOT_PATH}/${GROUP}/orion_db.pkl"


AUGMENTATION="--num_crops_per_aug 2 --augment=strong --min_scale ${MIN_SCALE} --color_jitter_prob ${JITTER_PROB} --brightness 0.4 --contrast 0.4 --saturation 0.4 --hue 0.1 --gray_scale_prob ${GRAY_SCALE_PROB} --gaussian_prob ${GAUSSIAN_PROB} --solarization_prob ${SOLARIZATION_PROB} --equalization_prob 0.0 --horizontal_flip_prob ${FLIP_PROB}"


if [[ $BACKBONE == "resnet18" ]]; then
  LR=0.9
  WD=0.0001
  CLR=0.1
  CWD=0.0
  ETA=0.002
  TEMPERATURE=0.1
elif [[ $BACKBONE == "resnet50" ]]; then
  LR=0.3
  WD=0.00003
  CLR=0.1
  CWD=0.0
  ETA=0.002
  TEMPERATURE=0.1
elif [[ $BACKBONE == "mlp" ]]; then
  LR=1.0
  WD=0.001
  CLR=0.1
  CWD=0.0
  ETA=0.002
  TEMPERATURE=0.1
fi

EXPNAME="test_${GROUP}"

## Uncomment the following two lines and specify hyperparameter priors
## if you want to perform a hyperparameter search
# EXPNAME="${GROUP}_{trial.hash_params}"
# orion -v hunt -n ${GROUP} --config=./orion_config.yaml \
python3 ./../main_pretrain.py \
    --name ${EXPNAME} \
    --method $METHOD \
    --dataset ${DATASET} \
    --backbone ${BACKBONE} \
    --train_data_path $TRAIN_DATA_PATH \
    --valid_data_path $VAL_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --checkpoint_dir="${ROOT_PATH}/${GROUP}/${EXPNAME}" \
    --checkpoint_frequency 10 \
    --save_checkpoint \
    --auto_resume \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --num_workers 4 \
    --model_selection_metric="${VAL_METRIC}" \
    --model_selection_mode="max" \
    --task $TASK \
    --max_epochs $MAX_EPOCHS \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --eta_lars $ETA --exclude_bias_n_norm --grad_clip_lars \
    --lr $LR \
    --weight_decay $WD \
    --classifier_lr $CLR --classifier_wd $CWD \
    --batch_size $BATCH_SIZE \
    $AUGMENTATION \
    --proj_output_dim $PROJ_OUT_DIM \
    --proj_hidden_dim 2048 \
    --queue_size $QUEUE_SIZE \
    --temperature $TEMPERATURE \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier

# done
