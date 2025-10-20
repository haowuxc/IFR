# bash scripts/eval_diffcap_coco.sh MODEL_NAME DEVICE setting_id
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
# COCO_LOG_FILE="$NOCAPS_OUT_PATH/COCO_${TIME_START}.log"

EVAL_SCRIPT="eval_ifr.py"

# python -u eval_ref.py \
python -u $EVAL_SCRIPT \
--device cuda:0 \
--name_of_datasets nocaps \
--path_of_val_datasets data/nocaps/nocaps_corpus.json \
--image_folder /mnt/petrelfs/wuhao2/datasets/data/nocaps/val/ \
--out_path=$NOCAPS_OUT_PATH \
--model $MODEL_NAME \
|& tee -a  ${NOCAPS_LOG_FILE}

# CUDA_VISIBLE_DEVICES=$DEVICE python -u eval_ag_diffcap.py \
# --device cuda:0 \
# --name_of_datasets nocaps \
# --path_of_val_datasets data/nocaps/nocaps_corpus.json \
# --image_folder /nas/shared/sport/wuhao/dataset/data/nocaps/val/ \
# --out_path=$NOCAPS_OUT_PATH \
# --model $MODEL_NAME \
# |& tee -a  ${NOCAPS_LOG_FILE}

# Define patterns to identify existing logs for each domain
INDOMAIN_PATTERN="${NOCAPS_OUT_PATH}/NOCAPS_*_indomain.log"
NEARDOMAIN_PATTERN="${NOCAPS_OUT_PATH}/NOCAPS_*_neardomain.log"
OUTDOMAIN_PATTERN="${NOCAPS_OUT_PATH}/NOCAPS_*_outdomain.log"
OVERALL_PATTERN="${NOCAPS_OUT_PATH}/NOCAPS_*_overall.log"

# Function to check if a log pattern has any matching file
is_log_present() {
  local pattern=$1
  
  # Check if any file matching the pattern exists
  compgen -G "$pattern" > /dev/null && return 0 || return 1
}


# if ! is_log_present "$INDOMAIN_PATTERN"; then
echo "==========================NOCAPS IN-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/indomain*captions.json |& tee -a  ${NOCAPS_LOG_FILE/.log/_indomain.log}
# else
#     echo "Skipping IN-DOMAIN evaluation as a log file with a similar pattern already exists."
# fi

# if ! is_log_present "$NEARDOMAIN_PATTERN"; then
echo "==========================NOCAPS NEAR-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/neardomain*captions.json |& tee -a  ${NOCAPS_LOG_FILE/.log/_neardomain.log}
# else
#     echo "Skipping NEAR-DOMAIN evaluation as a log file with a similar pattern already exists."
# fi

# if ! is_log_present "$OUTDOMAIN_PATTERN"; then
echo "==========================NOCAPS OUT-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/outdomain*captions.json |& tee -a  ${NOCAPS_LOG_FILE/.log/_outdomain.log}
# else
#     echo "Skipping OUT-DOMAIN evaluation as a log file with a similar pattern already exists."
# fi

# if ! is_log_present "$OVERALL_PATTERN"; then
echo "==========================NOCAPS ALL-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/overall*captions.json |& tee -a  ${NOCAPS_LOG_FILE/.log/_overall.log}
