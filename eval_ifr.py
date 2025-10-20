import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from search import beam_search
import random
import numpy as np

import models.evcap as evcap
import models.ifr as ifr


from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
from datasets import load_dataset
from omegaconf import OmegaConf

import yaml


def is_json_file_empty(file_path, min_len=50):
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
            if len(data) > min_len:
                return False
            return True
            # Check if the data is empty
            # return not bool(data)
        except json.JSONDecodeError:
            # The file exists but contains no valid JSON
            return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return transform(img).unsqueeze(0)


def validation_whoops(
    args,
    model,
    tokenizer,
    dataset_dir="/mnt/hwfile/sport/wuhao/dataset",
    img_dir="/mnt/hwfile/sport/wuhao/dataset/nlphuji/whoops/whoops_images",
) -> None:

    device = args.device

    predicts = []
    examples = load_dataset(
        os.path.join(dataset_dir, "nlphuji/whoops"),
        data_files=os.path.join(dataset_dir, "nlphuji/whoops/whoops_dataset.csv"),
    )
    model.eval()
    # breakpoint()
    for example in examples["train"]:
        image_id = example["image_id"]
        captions = example["crowd_captions"]
        predict = {}

        print("\n")
        print(image_id)
        print("GT: ", captions)
        image = os.path.join(img_dir, f"{image_id}.png")
        image = preprocess_image(image).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            if "rag" in args.model.lower():
                qform_all_proj, atts_qform_all_proj, ragcaps = model.encode_img(image)
                predict["rag_caps"] = ragcaps[0][:10]
                prompt_embeds, atts_prompt = model.prompt_wrap(
                    qform_all_proj, atts_qform_all_proj, model.prompt_list, ragcaps
                )
            else:
                qform_all_proj, atts_qform_all_proj = model.encode_img(image)
                prompt_embeds, atts_prompt = model.prompt_wrap(
                    qform_all_proj, atts_qform_all_proj, model.prompt_list
                )
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = (
                torch.ones([batch_size, 1], device=image.device)
                * tokenizer.bos_token_id
            )
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)
            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence = beam_search(
                embeddings=embeddings,
                tokenizer=tokenizer,
                beam_width=args.beam_width,
                model=model.llama_model,
            )
            sentence = sentence[0]
            print("Pred: ", sentence)

        predict["split"] = "valid"
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    out_json_path = os.path.join(
        args.out_path, f"{args.name_of_datasets}_generated_captions.json"
    )
    with open(out_json_path, "w") as outfile:
        json.dump(predicts, outfile, indent=4)


def validation_coco_flickr30k(args, inpath, model, tokenizer) -> None:

    device = args.device
    with open(inpath, "r") as infile:
        annotations = json.load(infile)
    predicts = []
    for idx, item in tqdm(enumerate(annotations)):
        image_id = item
        captions = annotations[item]
        image_path = args.image_folder + image_id
        predict = {}

        print("\n")
        print(image_path)
        print("GT: ", captions)
        image = preprocess_image(image_path).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            if "rag" in args.model.lower():
                qform_all_proj, atts_qform_all_proj, ragcaps = model.encode_img(image)
                predict["rag_caps"] = ragcaps[0][:10]
                prompt_embeds, atts_prompt = model.prompt_wrap(
                    qform_all_proj, atts_qform_all_proj, model.prompt_list, ragcaps
                )
            else:
                qform_all_proj, atts_qform_all_proj = model.encode_img(image)
                prompt_embeds, atts_prompt = model.prompt_wrap(
                    qform_all_proj, atts_qform_all_proj, model.prompt_list
                )
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = (
                torch.ones([batch_size, 1], device=image.device)
                * tokenizer.bos_token_id
            )
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)
            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence = beam_search(
                embeddings=embeddings,
                tokenizer=tokenizer,
                beam_width=args.beam_width,
                model=model.llama_model,
            )  # List[str]
            sentence = sentence[0]
            print("Pred: ", sentence)

        predict["split"] = "valid"
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)


    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    out_json_path = os.path.join(
        args.out_path, f"{args.name_of_datasets}_generated_captions.json"
    )
    with open(out_json_path, "w") as outfile:
        json.dump(predicts, outfile, indent=4)


def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def validation_nocaps(args, inpath, model, tokenizer) -> None:
    device = args.device
    overall_cap = os.path.join(args.out_path, f"overall_generated_captions.json")
    indomain_cap = os.path.join(args.out_path, f"indomain_generated_captions.json")
    neardomain_cap = os.path.join(args.out_path, f"neardomain_generated_captions.json")
    outdomain_cap = os.path.join(args.out_path, f"outdomain_generated_captions.json")

    if (
        os.path.exists(overall_cap)
        and os.path.exists(indomain_cap)
        and os.path.exists(neardomain_cap)
        and os.path.exists(outdomain_cap)
    ):
        print("Already generated captions")
        exit(0)

    with open(inpath, "r") as infile:
        annotations = json.load(infile)
    indomain = []
    neardomain = []
    outdomain = []
    overall = []
    model.eval()
    for idx, annotation in tqdm(enumerate(annotations)):
        # ann = img_info[idx]
        # image_file = ann['image']
        # img_id = ann['img_id']
        image_id = annotation["image_id"]
        split = annotation["split"]
        captions = annotation["caption"]
        predict = {}
        print("\n")
        image_path = args.image_folder + "/" + image_id
        print(image_path)
        print("GT: ", captions)
        image = preprocess_image(image_path).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            if "rag" in args.model.lower():
                qform_all_proj, atts_qform_all_proj, ragcaps = model.encode_img(image)
                predict["rag_caps"] = ragcaps[0][:10]
                prompt_embeds, atts_prompt = model.prompt_wrap(
                    qform_all_proj, atts_qform_all_proj, model.prompt_list, ragcaps
                )
            else:
                qform_all_proj, atts_qform_all_proj = model.encode_img(image)
                prompt_embeds, atts_prompt = model.prompt_wrap(
                    qform_all_proj, atts_qform_all_proj, model.prompt_list
                )  # (self, img_embeds, batch_names, atts_img, prompt_list):
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = (
                torch.ones([batch_size, 1], device=image.device)
                * tokenizer.bos_token_id
            )
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)

            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)

            sentence_ = beam_search(
                embeddings=embeddings,
                tokenizer=tokenizer,
                beam_width=args.beam_width,
                model=model.llama_model,
            )
            sentence_ = sentence_[0]
            sentence = sentence_.split("#")[0]
            print("Pred: ", sentence)

            # predict = {}
            predict["split"] = split
            predict["image_name"] = image_id
            predict["captions"] = captions
            predict["prediction"] = sentence

            overall.append(predict)

            if split == "in_domain":
                indomain.append(predict)
            elif split == "near_domain":
                neardomain.append(predict)
            elif split == "out_domain":
                outdomain.append(predict)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    with open(
        os.path.join(args.out_path, f"overall_generated_captions.json"), "w"
    ) as outfile:
        json.dump(overall, outfile, indent=4)
    with open(
        os.path.join(args.out_path, f"indomain_generated_captions.json"), "w"
    ) as outfile:
        json.dump(indomain, outfile, indent=4)
    with open(
        os.path.join(args.out_path, f"neardomain_generated_captions.json"), "w"
    ) as outfile:
        json.dump(neardomain, outfile, indent=4)
    with open(
        os.path.join(args.out_path, f"outdomain_generated_captions.json"), "w"
    ) as outfile:
        json.dump(outdomain, outfile, indent=4)


@torch.no_grad()
def main(args) -> None:
    args = OmegaConf.create(vars(args))
    # initializing
    device = args.device
    # loading model
    # model_type = "vicuna-13b-v1.3"
    caption_file = (
        os.path.join(args.out_path, f"{args.name_of_datasets}_generated_captions.json")
        if args.name_of_datasets != "nocaps"
        else os.path.join(args.out_path, "overall_generated_captions.json")
    )

    if os.path.exists(caption_file):
        # load the json file
        empty = is_json_file_empty(caption_file)
        if not empty:
            print("Already generated captions")
            exit(0)
        else:
            print("Caption file is empty, re-generating captions")

    else:
        print("Caption file does not exist, loading model to generate captions")

    ckpt = os.path.join(args.out_path, "000.pt")
    if not os.path.exists(ckpt):
        raise ValueError("Invalid ckpt path")

    # ckpt = 'results/train_{}/000.pt'.format(args.model)
    # load config
    cfg = os.path.join(os.path.dirname(ckpt), "config.yaml")
    if os.path.exists(cfg):
        config = OmegaConf.load(cfg)
        args = OmegaConf.merge(args, config)

    ckpt_path = "/mnt/petrelfs/wuhao2/models/ckpts"
    # ckpt_path = "/nas/shared/sport/wuhao/model/ckpts"

    model_type = os.path.join(ckpt_path, args.llm)

    prompt_path = "prompts/prompt_evcap.txt"


    print("load:", ckpt)

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
    state_dict = torch.load(ckpt, map_location=device)["model"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    inpath = args.path_of_val_datasets
    tokenizer = model.llama_tokenizer

    if args.name_of_datasets == "nocaps":
        validation_nocaps(args, inpath, model, tokenizer)
    if args.name_of_datasets == "coco" or args.name_of_datasets == "flickr30k":
        validation_coco_flickr30k(args, inpath, model, tokenizer)
    if args.name_of_datasets == "whoops":
        validation_whoops(args, model, tokenizer)


if __name__ == "__main__":
    print("Starts ...")
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--name_of_datasets",
        default="coco",
        choices=("coco", "flickr30k", "nocaps", "whoops"),
    )
    parser.add_argument(
        "--path_of_val_datasets", default="./data/coco/val_captions.json"
    )
    # parser.add_argument('--image_folder', default = './annotations/coco/val2014/')
    parser.add_argument(
        "--image_folder",
        default="/mnt/petrelfs/wuhao2/datasets/data/coco2014/val2014/",
    )
    parser.add_argument("--out_path", help="model path and save log path")
    parser.add_argument("--topn", type=int, default=9)
    parser.add_argument("--beam_width", type=int, default=5, help="width of beam")
    parser.add_argument(
        "--random_seed", type=int, default=42, help="set random seed for reproducing"
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
    parser.add_argument("--model", type=str, default="evcap", help="model type")
    parser.add_argument("--num_query_token_txt", type=int, default=8)
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

    args = parser.parse_args()
    set_seed(args.random_seed)
    print("args: {}\n".format(vars(args)))
    main(args)
