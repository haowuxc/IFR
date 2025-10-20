import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import random
import sys
import argparse
import numpy as np
import utils
from optims import LinearWarmupCosineLRScheduler, set_optimizer
import yaml
import json

from dataset.coco_dataset import (
    COCODataset,
    AlternateBatchSampler,
)


import models.evcap as evcap
import models.ifr as ifr
import pprint
from omegaconf import OmegaConf
import subprocess
import time
import shutil
from pathlib import Path


snapshot_code = [
    "./train_ifr.py",
    "./eval_ag_diffcap.py",
    "./scripts/train_diffcap_ag.sh",
    "./models",
]


def create_py_file_snapshot(source_list, target_dir):
    """
    Create a snapshot of .py files from a list of files or folders to a target directory.

    Parameters:
    - source_list (list of str): A list of file or folder paths to include in the snapshot.
    - target_dir (str): The directory where the .py files will be backed up.
    """
    # Convert target_dir to Path object and ensure it exists
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    for source in source_list:
        source_path = Path(source)

        if not source_path.exists():
            print(f"Warning: '{source}' does not exist. Skipping.")
            continue

        if source_path.is_file() and (
            source_path.suffix == ".py" or source_path.suffix == ".sh"
        ):
            # If the source is a single .py file, copy it directly
            dest_path = target_path / source_path.name
            shutil.copy2(source_path, dest_path)
        elif source_path.is_dir():
            # If the source is a directory, copy all .py files recursively
            for py_file in source_path.rglob("*.py"):
                relative_path = py_file.relative_to(
                    source_path
                )  # Maintain relative structure
                dest_path = target_path / source_path.name / relative_path
                dest_path.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Create necessary directories
                shutil.copy2(py_file, dest_path)
        else:
            print(f"Warning: '{source}' is not a .py file or a directory. Skipping.")

    print(f"Snapshot of .py files created in: '{target_path.resolve()}'.")


def run_bash_commands(model_name, setting_id, device="0"):
    """
    Run three bash commands using the returned model_name and setting_id from the train function.
    :param model_name: The name of the model.
    :param setting_id: The setting ID for the model.
    :param device: The device (GPU or CPU) identifier (default is "0").
    """

    # Define the bash commands you want to run
    commands = [
        f"bash scripts/eval_diffcapag_flickr30k.sh {model_name} {device} {setting_id}",
        f"bash scripts/eval_diffcapag_nocaps.sh {model_name} {device} {setting_id}",
        f"bash scripts/eval_diffcapag_coco.sh {model_name} {device} {setting_id}",
    ]

    for command in commands:
        try:
            # Run the command using subprocess
            result = subprocess.run(command, shell=True, text=True)

            # Check the result
            if result.returncode == 0:
                print(f"Command succeeded: {command}")
                print(result.stdout)  # Print the output of the command
            else:
                print(f"Command failed: {command}")
                print(result.stderr)  # Print the error message
        except Exception as e:
            print(f"An error occurred while running the command: {command}")
            print(str(e))


from common.dist_utils import (
    get_rank,
    init_distributed_mode,
    get_world_size,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def save_checkpoint(model, optimizer, cur_epoch, output_dir):
    """
    Save the checkpoint at the current epoch.
    """
    model_no_ddp = model
    param_grad_dic = {k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()}
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": cur_epoch,
    }
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, output_dir))
    torch.save(save_obj, output_dir)


def train(dataset, model, args):
    device = torch.device(f"cuda:{get_rank()}")
    batch_size = args.bs
    epochs = args.epochs
    accum_grad_iters = 1
    output_dir = args.out_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if args.distributed:

        if args.model.lower() == "refineragcap_batch":
            distributed_sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
            )
            sampler = AlternateBatchSampler(
                distributed_sampler=distributed_sampler,
                dataset1_len=len(dataset.datasets[0]),
                dataset2_len=len(dataset.datasets[1]),
                batch_size=args.bs,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                num_replicas=get_world_size(),
                rank=get_rank(),
            )
        model = model.to(device)
        if args.model.lower() in ["refigcap"]:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[get_rank()], find_unused_parameters=True
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[get_rank()]
            )

    else:
        sampler = None

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
    )

    if args.model.lower() == "refineragcap_batch":
        # for i, batch in enumerate(train_dataloader):
        #     breakpoint()
        #     # print(batch["initial_caption"])
        #     if i > 100:
        exit(0)
    model.train()
    optimizer = set_optimizer(
        model, init_lr=args.init_lr, weight_decay=args.weight_decay
    )
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=epochs,
        iters_per_epoch=len(train_dataloader),
        min_lr=args.min_lr,
        init_lr=args.init_lr,
        decay_rate=None,
        warmup_start_lr=args.warmup_start_lr,
        warmup_steps=args.warmup_steps,
    )
    # Initialize AMP if enabled
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    use_amp = scaler is not None
    print(f"Using AMP: {use_amp}")

    # Setup metric logging
    loss_metrics = ["loss", "loss_cap", "loss_info", "loss_div"]
    if "weight" in args and "div_all" in args.weight:
        loss_metrics.append("loss_div_all")

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        
        # Initialize metric logger
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
        
        for metric in loss_metrics:
            metric_logger.add_meter(
                metric, 
                utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
            )

        metric_logger.update(loss=1000.0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        print_freq = 50
        header = "Train Epoch: [{}]".format(epoch)


        for idx, samples in enumerate(
            metric_logger.log_every(train_dataloader, print_freq, header)
        ):
            # breakpoint()
            # optimizer.zero_grad()
            samples["image"] = samples["image"].to(device)
            scheduler.step(cur_epoch=epoch, cur_step=idx)
            with torch.cuda.amp.autocast(enabled=use_amp):
                losses = model(samples)
                loss = losses["loss"]

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (idx + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            # Update additional loss metrics if they exist
            for metric in loss_metrics[1:]:  # Skip 'loss' which we already updated
                if metric in losses:
                    metric_value = losses[metric].item() if hasattr(losses[metric], 'item') else losses[metric]
                    metric_logger.update(**{metric: metric_value})

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())

        # save the averaged stats to a file
        with open(os.path.join(output_dir, f"train.log"), "w") as f:
            f.write(str(metric_logger.global_avg()) + "\n")

        if epoch == epochs - 1:
            output_dir_model = os.path.join(output_dir, f"{epoch:03d}.pt")
            save_checkpoint(model, optimizer, epoch, output_dir_model)
    return model


def save_config(args, out_path):
    i = 1
    while os.path.exists(os.path.join(out_path, f"setting_{i}")):
        i += 1

    new_out_path = os.path.join(out_path, f"setting_{i}")
    settings_id = i
    # args['out_dir'] = new_out_path
    os.makedirs(new_out_path, exist_ok=True)

    OmegaConf.save(args, os.path.join(new_out_path, "config.yaml"))

    return new_out_path, settings_id


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __getitem__(self, item):
        return self.__dict__[item]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="./align_results",
        help="output directory for saving model checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=24)
    parser.add_argument("--is_rn", dest="is_rn", action="store_true")
    parser.add_argument("--device", default="cuda", help="gpu for training")
    parser.add_argument("--distributed", default=True)
    parser.add_argument("--amp", default=True)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--num_query_token_txt", type=int, default=8)
    parser.add_argument("--topn", type=int, default=3)
    # lr args
    # parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--disable_random_seed",
        action="store_true",
        default=False,
        help="set random seed for reproducing",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="set random seed for reproducing"
    )
    parser.add_argument(
        "--freeze_vit",
        action="store_false",
        default=True,
        help="freeze vision transformer",
    )
    parser.add_argument(
        "--freeze_qformer",
        action="store_false",
        default=True,
        help="freeze query transformer",
    )

    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--llm", type=str, default="lmsys/vicuna-13b-v1.3", help="Path to config file"
    )
    parser.add_argument(
        "--ext_path",
        type=str,
        default="ext_data/ext_memory_lvis.pkl",
        help="Path to external memory file",
    )

    parser.add_argument(
        "--num_query_token", type=int, default=32, help="Path to config file"
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--map",
        action="store_true",
        default=False,
        help="whether map aliyun path to s1",
    )
    args = parser.parse_args()

    return args


def print_config(config):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config.__dict__)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Starts ...")
    print(" # PID :", os.getpid())

    args = parse_arguments()
    args_dict = OmegaConf.create(vars(args))

    if args.config:
        config = OmegaConf.load(args.config)

    else:
        raise ValueError("Please provide a config file")
    # breakpoint()
    args = OmegaConf.merge(args_dict, config)

    config["config_path"] = args.config.split("/")[-1]
    if "raw_rag_text_input" not in config:
        config.raw_rag_text_input = False
    if "prompt" not in config:
        prompt_path = (
            "prompts/prompt_imgtxtfuse.txt"
            if config.raw_rag_text_input
            else "prompts/prompt_evcap.txt"
        )
    else:
        prompt_path = config.prompt
    # breakpoint()

    # breakpoint()
    # if "rag" in self.model.lower()
    args.out_dir = "align_results/train_{}".format(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    out_dir, setting_id = save_config(config, args.out_dir)
    # breakpoint()
    args.out_dir = out_dir
    # args = Config(**config)

    create_py_file_snapshot(snapshot_code, out_dir)

    # print config
    print(args)

    if not args.disable_random_seed:
        set_seed(args.seed)
    init_distributed_mode(args)

    # breakpoint()
    if args.map:
        data_root = "/mnt/petrelfs/wuhao2/datasets/data/coco2014"
        ckpt_path = "/mnt/petrelfs/wuhao2/models/ckpts"

    else:
        data_root = "/nas/shared/sport/wuhao/dataset/data/coco2014"
        ckpt_path = "/nas/shared/sport/wuhao/model/ckpts"


    dataset = COCODataset(data_root=data_root)

    model_type = os.path.join(ckpt_path, args.llm)
    # breakpoint()
    if os.path.exists(os.path.join(out_dir, "000.pt")):
        print(f"Model {args.model} already trained")
        exit(0)
    # breakpoint()
    if args.model.lower() == "evcap":  # is evcap_gt
        model = evcap.EVCap(
            ext_path="ext_data/ext_memory_lvis.pkl",
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=args.freeze_vit,
            freeze_qformer=args.freeze_qformer,
            num_query_token=args.num_query_token,
            num_query_token_txt=args.num_query_token_txt,
            topn=args.topn,
            llama_model=model_type,
            prompt_path="prompts/prompt_evcap.txt",
            prompt_template="###Human: {} ###Assistant: ",
            max_txt_len=128,
            end_sym="\n",
            low_resource=False,
            device_8bit=0,
        )
    elif args.model.lower() == "feataligncap":
        model = ifr.FeatAlignCap(
            ext_path=args.ext_path,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=args.freeze_vit,
            freeze_qformer=args.freeze_qformer,
            num_query_token=args.num_query_token,
            num_query_token_txt=args.num_query_token_txt,
            topn=args.topn,
            llama_model=model_type,
            prompt_path=prompt_path,
            prompt_template="###Human: {} ###Assistant: ",
            max_txt_len=128,
            end_sym="\n",
            low_resource=False,
            device_8bit=0,
            config=args,
        )
    elif args.model.lower() == "feataligncatcap":
        model = ifr.FeatAlignCatCap(
            ext_path=args.ext_path,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=args.freeze_vit,
            freeze_qformer=args.freeze_qformer,
            num_query_token=args.num_query_token,
            num_query_token_txt=args.num_query_token_txt,
            topn=args.topn,
            llama_model=model_type,
            prompt_path=prompt_path,
            prompt_template="###Human: {} ###Assistant: ",
            max_txt_len=128,
            end_sym="\n",
            low_resource=False,
            device_8bit=0,
            config=args,
        )
    else:
        raise ValueError(f"model {args.model} not implemented")

    if "init_lr" not in args:
        args.init_lr = 1e-4
    if "weight_decay" not in args:
        args.weight_decay = 0.05
    if "min_lr" not in args:
        args.min_lr = 8e-5
    if "warmup_steps" not in args:
        args.warmup_steps = 5000
    if "warmup_start_lr" not in args:
        args.warmup_start_lr = 1e-6

    train_start_time = time.time()
    train(dataset, model, args)
    train_time = time.time() - train_start_time
    print("training time: {}h".format(train_time / 3600))



if __name__ == "__main__":
    main()
