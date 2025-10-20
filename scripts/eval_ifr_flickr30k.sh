# bash scripts/eval_diffcap_flickr30k.sh MODEL_NAME DEVICE setting_id
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
COCO_LOG_FILE="$NOCAPS_OUT_PATH/flickr_${TIME_START}.log"

EVAL_SCRIPT="eval_ifr.py"

# python -u eval_ref.py \
python -u $EVAL_SCRIPT \
--device cuda:0 \
--name_of_datasets flickr30k \
--path_of_val_datasets data/flickr30k/test_captions.json \
--image_folder /mnt/petrelfs/wuhao2/datasets/data/flickr30k_images/flickr30k_images/ \
--out_path=$NOCAPS_OUT_PATH \
--model $MODEL_NAME \
|& tee -a  ${NOCAPS_LOG_FILE}


FLICKR_PATTERN="${NOCAPS_OUT_PATH}/flickr_*.log"

# if ! is_log_present "$FLICKR_PATTERN"; then
echo "==========================NOCAPS FLICKR================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/flickr*captions.json |& tee -a  ${COCO_LOG_FILE}