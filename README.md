# IFR

## Environment
```bash
# Using conda
conda env create -f environment.yaml
conda activate ifr
pip install -r requirements_ifr.txt
```

## Training
```
bash scripts/train_ifr.sh CONFIG_FILE GPU_ID
```
## Evaluation
```
# Download evaluation code
wget https://github.com/FeiElysia/ViECap/releases/download/checkpoints/evaluation.zip
unzip evaluation.zip

# Run evaluation
bash scripts/eval_ifr_dataset.sh MODEL_NAME DEVICE setting_id
```
**Note**: The `setting_id` parameter corresponds to experiment labels. Check `scripts/eval_ifr_dataset.sh` for available options and their meanings.
