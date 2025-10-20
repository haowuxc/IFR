# bash scripts/eval_diffcapag_coco.sh MODEL_NAME DEVICE setting_id
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

MODEL_NAME=$1
EXP_NAME=train_${MODEL_NAME}
DEVICE=$2
SETTING=$3
NOCAPS_OUT_PATH=align_results/$EXP_NAME/setting_$SETTING
# NOCAPS_OUT_PATH=fb_results/$EXP_NAME/setting_$SETTING

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")


NOCAPS_LOG_FILE="$NOCAPS_OUT_PATH/NOCAPS_${TIME_START}.log"
COCO_LOG_FILE="$NOCAPS_OUT_PATH/COCO_${TIME_START}.log"

EVAL_SCRIPT="eval_ifr.py"

# python -u eval_ref.py \
python -u $EVAL_SCRIPT \
--device cuda:0 \
--name_of_datasets coco \
--path_of_val_datasets data/coco/test_captions.json \
--image_folder /mnt/petrelfs/wuhao2/datasets/data/coco2014/val2014/ \
--out_path=$NOCAPS_OUT_PATH \
--model $MODEL_NAME \
|& tee -a  ${NOCAPS_LOG_FILE}

# CUDA_VISIBLE_DEVICES=$DEVICE python -u eval_ag_diffcap.py \
# --device cuda:0 \
# --name_of_datasets coco \
# --path_of_val_datasets data/coco/test_captions.json \
# --image_folder /nas/shared/sport/wuhao/dataset/data/coco2014/val2014/ \
# --out_path=$NOCAPS_OUT_PATH \
# --model $MODEL_NAME \
# |& tee -a  ${NOCAPS_LOG_FILE}

COCO_PATTERN="${NOCAPS_OUT_PATH}/COCO_*.log"

# Function to check if a log pattern has any matching file
is_log_present() {
  local pattern=$1
  
  # Check if any file matching the pattern exists
  compgen -G "$pattern" > /dev/null && return 0 || return 1
}

# if ! is_log_present "$COCO_PATTERN"; then
echo "==========================NOCAPS COCO================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/coco*captions.json |& tee -a  ${COCO_LOG_FILE}
