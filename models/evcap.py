import logging
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import random
from models.blip2 import Blip2Base, disabled_train
from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import pickle
import faiss
import re

import collections
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM


def filter_retrieval_results_old(retrievals, min_frequency=1, min_returned=3):
    """
    Filters the retrievals based on a minimum frequency of occurrence for entities and actions.
    Ensures that at least `min_returned` entities and actions are returned.

    Args:
    retrievals (list of dict): List of retrieval results where each dict contains 'entity' and 'action' as lists.
    min_frequency (int): The minimum frequency for entities/actions to be considered.
    min_returned (int): The minimum number of entities and actions to return.

    Returns:
    list: Filtered list of retrievals with at least `min_returned` entities and actions.
    """
    # Flatten lists in the retrievals to count frequency
    entity_counter = collections.Counter()
    action_counter = collections.Counter()

    for item in retrievals:
        entity_counter.update(item["entity"])
        action_counter.update(item["action"])

    # Select entities and actions based on frequency and ensure minimum count
    selected_entities = [
        e for e, count in entity_counter.most_common() if count >= min_frequency
    ]
    selected_actions = [
        a for a, count in action_counter.most_common() if count >= min_frequency
    ]

    # Ensure at least `min_returned` entities and actions are selected
    if len(selected_entities) < min_returned:
        additional_entities = [e for e in entity_counter if e not in selected_entities]
        selected_entities += additional_entities[
            : min_returned - len(selected_entities)
        ]

    if len(selected_actions) < min_returned:
        additional_actions = [a for a in action_counter if a not in selected_actions]
        selected_actions += additional_actions[: min_returned - len(selected_actions)]

    return selected_entities[:min_returned], selected_actions[:min_returned]


def filter_retrieval_results(retrievals, min_frequency=1, min_returned=3):
    """
    Filters the retrievals based on a minimum frequency of occurrence for entities and actions.
    Ensures that at least `min_returned` entities and actions are returned.

    Args:
    retrievals (list of dict): List of retrieval results where each dict contains 'entity' and 'action' as lists.
    min_frequency (int): The minimum frequency for entities/actions to be considered.
    min_returned (int): The minimum number of entities and actions to return.

    Returns:
    list: Filtered list of retrievals with at least `min_returned` entities and actions.
    """
    # Flatten lists in the retrievals to count frequency
    entity_counter = collections.Counter()
    action_counter = collections.Counter()
    scene_counter = collections.Counter()

    if "scene" in retrievals[0]:
        for item in retrievals:
            entity_counter.update(item["entity"])
            action_counter.update(item["action"])
            scene_counter.update(item["scene"])
        has_scene = True
    else:
        for item in retrievals:
            entity_counter.update(item["entity"])
            action_counter.update(item["action"])
        has_scene = False

    # Select entities and actions based on frequency and ensure minimum count
    selected_entities = [
        e for e, count in entity_counter.most_common() if count >= min_frequency
    ]
    selected_actions = [
        a for a, count in action_counter.most_common() if count >= min_frequency
    ]
    if has_scene:
        selected_scenes = [
            s for s, count in scene_counter.most_common() if count >= min_frequency
        ]

    # Ensure at least `min_returned` entities and actions are selected
    if len(selected_entities) < min_returned:
        additional_entities = [e for e in entity_counter if e not in selected_entities]
        selected_entities += additional_entities[
            : min_returned - len(selected_entities)
        ]

    if len(selected_actions) < min_returned:
        additional_actions = [a for a in action_counter if a not in selected_actions]
        selected_actions += additional_actions[: min_returned - len(selected_actions)]

    if len(selected_scenes) < min_returned:
        additional_scenes = [s for s in scene_counter if s not in selected_scenes]
        selected_scenes += additional_scenes[: min_returned - len(selected_scenes)]

    if has_scene:
        data = {
            "entity": selected_entities[:min_returned],
            "action": selected_actions[:min_returned],
            "scene": selected_scenes[:min_returned],
        }
    else:
        data = {
            "entity": selected_entities[:min_returned],
            "action": selected_actions[:min_returned],
            "scene": [],
        }
    # data = {'entity': selected_entities[:min_returned], 'action': selected_actions[:min_returned], 'scene': selected_scenes[:min_returned]}

    # return selected_entities[:min_returned], selected_actions[:min_returned]
    return data


def filter_retrieval_results_sim(retrievals, sim_tensor, min_returned=3):
    """
    Filters the retrievals based on weighted similarity values.
    Ensures that at least `min_returned` entities, actions, and scenes are returned.

    Args:
    retrievals (list of dict): List of retrieval results where each dict contains 'entity', 'action', and 'scene' as lists.
    sim_tensor (torch.Tensor or numpy array): Tensor containing similarity weights for retrievals.
    min_returned (int): The minimum number of entities and actions to return.

    Returns:
    dict: Filtered list of retrievals with at least `min_returned` entities, actions, and scenes.
    """
    # Initialize weighted counters for entities, actions, and scenes
    entity_counter = collections.defaultdict(float)
    action_counter = collections.defaultdict(float)
    scene_counter = collections.defaultdict(float)

    has_scene = "scene" in retrievals[0]

    # Accumulate weighted values from sim_tensor for each entity, action, and scene
    for i, item in enumerate(retrievals):
        weight = sim_tensor[
            i
        ].item()  # Same weight for entity, action, and scene for this retrieval

        # Update entity weights
        for entity in item["entity"]:
            entity_counter[entity] += weight

        # Update action weights
        for action in item["action"]:
            action_counter[action] += weight

        # Update scene weights if present
        if has_scene:
            for scene in item["scene"]:
                scene_counter[scene] += weight

    # Sort by the accumulated weighted sum
    selected_entities = sorted(entity_counter, key=entity_counter.get, reverse=True)
    selected_actions = sorted(action_counter, key=action_counter.get, reverse=True)
    selected_scenes = (
        sorted(scene_counter, key=scene_counter.get, reverse=True) if has_scene else []
    )

    # Ensure at least `min_returned` entities and actions are selected
    if len(selected_entities) < min_returned:
        additional_entities = [e for e in entity_counter if e not in selected_entities]
        selected_entities += additional_entities[
            : min_returned - len(selected_entities)
        ]

    if len(selected_actions) < min_returned:
        additional_actions = [a for a in action_counter if a not in selected_actions]
        selected_actions += additional_actions[: min_returned - len(selected_actions)]

    if has_scene and len(selected_scenes) < min_returned:
        additional_scenes = [s for s in scene_counter if s not in selected_scenes]
        selected_scenes += additional_scenes[: min_returned - len(selected_scenes)]

    return {
        "entity": selected_entities[:min_returned],
        "action": selected_actions[:min_returned],
        "scene": selected_scenes[:min_returned] if has_scene else [],
    }


def create_caption_from_retrievals_old(
    retrievals, min_returned=3, template="{entity} and {action}"
):
    """
    Forms sentences based on the filtered retrieval results using a template.

    Args:
    retrievals (list of dict): List of retrieval results where each dict contains 'entity' and 'action' as lists.
    min_returned (int): The minimum number of entities and actions to ensure in the caption.
    template (str): Template for forming sentences. Default is "{entity} and {action}".

    Returns:
    str: Generated sentence or caption.
    """
    # Filter retrievals to ensure at least `min_returned` entities and actions
    entities, actions = filter_retrieval_results(retrievals, min_returned=min_returned)

    # Create sentences by flexibly combining entities and actions
    entity_part = ", ".join(entities)
    action_part = ", ".join(actions)

    # Use the template to create a coherent caption
    caption = template.format(entity=entity_part, action=action_part)

    return caption


def create_caption_from_retrievals(
    retrievals,
    min_returned=3,
    template="{entity} and {action}",
    scene_template=" in {scene}",
    rt_attribute=False,
):
    """
    Forms sentences based on the filtered retrieval results using a template.

    Args:
    retrievals (list of dict): List of retrieval results where each dict contains 'entity' and 'action' as lists.
    min_returned (int): The minimum number of entities and actions to ensure in the caption.
    template (str): Template for forming sentences. Default is "{entity} and {action}".

    Returns:
    str: Generated sentence or caption.
    """
    # Filter retrievals to ensure at least `min_returned` entities and actions
    data = filter_retrieval_results(retrievals, min_returned=min_returned)
    entities = data["entity"]
    actions = data["action"]
    scenes = data["scene"]
    # entities, actions = filter_retrieval_results(retrievals, min_returned=min_returned)

    # Create sentences by flexibly combining entities and actions
    entity_part = ", ".join(entities)
    action_part = ", ".join(actions)
    scene_part = ", ".join(scenes)

    if rt_attribute == "object":
        return entity_part
    elif rt_attribute == "action":
        return action_part
    elif rt_attribute == "scene":
        return scene_part
    else:
        caption = template.format(entity=entity_part, action=action_part)

    # # Use the template to create a coherent caption
    # caption = template.format(entity=entity_part, action=action_part)

    if len(scenes) > 0:
        caption += scene_template.format(scene=scene_part)
        # breakpoint()

    return caption


def create_caption_from_retrievals_sim(
    retrievals,
    sim_tensor,
    min_returned=3,
    template="{entity} and {action}",
    scene_template=" in {scene}",
    rt_attribute=False,
):
    """
    Forms sentences based on the weighted similarity values and retrieval results using a template.

    Args:
    retrievals (list of dict): List of retrieval results where each dict contains 'entity', 'action', and 'scene' as lists.
    sim_tensor (torch.Tensor or numpy array): Similarity tensor with weights corresponding to the retrievals.
    min_returned (int): The minimum number of entities and actions to ensure in the caption.
    template (str): Template for forming sentences. Default is "{entity} and {action}".
    scene_template (str): Template for adding scene information. Default is " in {scene}".

    Returns:
    str: Generated sentence or caption.
    """
    # Ensure sim_tensor and retrievals match in size
    assert (
        len(retrievals) == sim_tensor.shape[0]
    ), "Sim tensor must match the number of retrievals."

    # Filter retrievals using weighted sum and similarity
    data = filter_retrieval_results_sim(
        retrievals, sim_tensor, min_returned=min_returned
    )
    entities = data["entity"]
    actions = data["action"]
    scenes = data["scene"]

    # Create sentences by combining entities and actions
    entity_part = ", ".join(entities)
    action_part = ", ".join(actions)
    scene_part = ", ".join(scenes)

    if rt_attribute == "object":
        return entity_part
    elif rt_attribute == "action":
        return action_part
    elif rt_attribute == "scene":
        return scene_part
    elif rt_attribute == "object_action":
        return entity_part + " and " + action_part
    elif rt_attribute == "all":
        caption = entity_part + " and " + action_part
    elif rt_attribute == "object_scene":
        caption = entity_part
    elif rt_attribute == "action_scene":
        caption = action_part
        # # Use the template to create a coherent caption
        # caption = template.format(entity=entity_part, action=action_part)

    # Add scene information if available
    if len(scenes) > 0:
        caption += scene_template.format(scene=scene_part)

    return caption


class EVCap(Blip2Base):

    def __init__(
        self,
        ext_path,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model="",
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template="###Human: {} ###Assistant: ",
        max_txt_len=160,
        end_sym="\n",
        low_resource=False,
        device_8bit=0,
    ):
        super().__init__()

        self.low_resource = low_resource
        self.topn = topn
        print("topn:", self.topn)

        ##### Image
        print("Loading VIT")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print("Loading VIT Done")

        print("Loading Q-Former")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = True
            logging.info("freeze Qformer")
        print("Loading Q-Former Done")

        ##### Text
        self.bert_tokenizer = self.init_tokenizer()
        self.Qformer_txt, self.query_tokens_txt = self.init_Qformer_txt(
            num_query_token_txt, self.Qformer.config.hidden_size
        )
        self.Qformer_txt.resize_token_embeddings(len(self.bert_tokenizer))
        self.Qformer_txt.cls = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        if freeze_qformer:
            for name, param in self.Qformer_txt.named_parameters():
                param.requires_grad = False
            self.Qformer_txt = self.Qformer_txt.eval()
            self.Qformer_txt.train = disabled_train
            self.query_tokens_txt.requires_grad = True
            logging.info("freeze Qformer")
        print("query_tokens_txt", self.query_tokens_txt.shape)
        print("Loading Q-Former Done")
        print("Loading Q-Former_txt Done")

        ##### Caption generation
        print("Loading LLAMA")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
                ignore_mismatched_sizes=True,
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                device_map="auto",
                ignore_mismatched_sizes=True,
            )

        # frozen llama model
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print("Loading LLAMA Done")

        ###
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        # breakpoint()

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, "r") as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt
            ]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print("Load {} training prompts".format(len(self.prompt_list)))
            print("Prompt Example \n{}".format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        print(ext_path)
        with open(ext_path, "rb") as f:
            ext_base_img, self.ext_base_img_id = pickle.load(f)
            print(ext_base_img.shape, len(self.ext_base_img_id))
            feature_library_cpu = ext_base_img.cpu().numpy()
            faiss.normalize_L2(feature_library_cpu)
            self.feat_index = faiss.IndexFlatIP(feature_library_cpu.shape[1])
            self.feat_index.add(feature_library_cpu)
            print(f"loaded external base image")

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def prompt_wrap(self, img_embeds, atts_img, prompt_list):
        if prompt_list:
            batch_size = img_embeds.shape[0]
            emb_lists = []
            for i in range(batch_size):
                prompt = random.choice(prompt_list)
                p_before, p_after = prompt.split("<ImageHere>", 1)
                self.llama_tokenizer.padding_side = "right"
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(img_embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(
                    p_before_tokens.input_ids
                )
                p_after_embeds = self.llama_model.model.embed_tokens(
                    p_after_tokens.input_ids
                )
                img_embeds_i = img_embeds[i].unsqueeze(0)
                wrapped_embed_i = torch.cat(
                    [p_before_embeds, img_embeds_i, p_after_embeds], dim=1
                )
                emb_lists.append(wrapped_embed_i)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.llama_model.model.embed_tokens(
                torch.tensor(
                    self.llama_tokenizer.pad_token_id, device=img_embeds.device
                )
            )
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros(
                [len(emb_lens), max(emb_lens)],
                dtype=torch.int,
                device=img_embeds.device,
            )
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, : emb_lens[i]] = emb
                wrapped_atts[i, : emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img

    def pre_name(self, caption):
        caption = re.sub(
            r"([_!,'\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption

    def retrieve_similar_features(
        self, query_features, feat_index, image_id, top_k=5, sub_top_k=32
    ):
        batch_size, nums, dims = query_features.shape
        query_features = query_features.view(-1, dims)

        query_features_cpu = query_features.detach().cpu().numpy()
        faiss.normalize_L2(query_features_cpu)
        top_k_similarities, top_k_indices = feat_index.search(query_features_cpu, top_k)

        top_k_indices = torch.tensor(top_k_indices).to(device=query_features.device)
        top_k_similarities = torch.tensor(top_k_similarities).to(
            device=query_features.device
        )
        top_k_similarities = top_k_similarities.view(batch_size, -1)

        indices = top_k_indices.view(batch_size, -1)

        re_txt_list_all = []
        for batch_i in range(batch_size):
            indices_list = indices[batch_i]
            re_txt_batch_list = []
            for i in indices_list:
                re_txt_batch_list.append(image_id[i])
            re_txt_list_all.append(re_txt_batch_list)

        sorted_batched_ret = []
        for listA, listB in zip(top_k_similarities, re_txt_list_all):
            sorted_listA, indices = listA.sort(descending=True)
            sorted_listB = [self.pre_name(listB[idx]) for idx in indices]
            sorted_listB = sorted_listB[:sub_top_k]
            sorted_batched_ret.append(sorted_listB)
        return sorted_batched_ret

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )  # -1 means keep the original size
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            re_txt_list_all = self.retrieve_similar_features(
                query_output_img, self.feat_index, self.ext_base_img_id
            )
            re_txt_list_batch = []
            for sublist in re_txt_list_all:
                sublist_new = []
                for item in sublist:
                    if item not in sublist_new:
                        sublist_new.append(item)
                        if len(sublist_new) > self.topn:
                            break
                re_txt_list_batch.append(" [SEP] ".join(sublist_new))

            text = self.bert_tokenizer(
                re_txt_list_batch,
                truncation=True,
                padding="longest",
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            query_tokens_txt = self.query_tokens_txt.expand(
                image_embeds.shape[0], -1, -1
            )
            query_atts_txt = torch.ones(
                query_tokens_txt.size()[:-1], dtype=torch.long
            ).to(image_embeds.device)

            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            query_output_img_atts = torch.cat(
                [query_atts_txt, query_output_img_atts], dim=1
            )

            attention_mask = text.attention_mask
            query_outputs_txt = self.Qformer_txt.bert(
                text.input_ids,
                query_embeds=query_tokens_txt,
                attention_mask=attention_mask,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            # the attention_mask is for input tokens but the encoder_attention_mask is for encoder hidden states(here is query_output_img)
            query_output_txt = query_outputs_txt.last_hidden_state[
                :, : query_tokens_txt.size(1), :
            ]  # only keep the txt part

            query_output_all = torch.cat([query_output_img, query_output_txt], dim=1)
            qform_all_proj = self.llama_proj(query_output_all)
            atts_qform_all_proj = torch.ones(
                qform_all_proj.size()[:-1], dtype=torch.long
            ).to(device)
        return qform_all_proj, atts_qform_all_proj

    def forward(self, samples):
        ##### Image
        image = samples["image"]
        qform_all_proj, atts_qform_all_proj = self.encode_img(
            image
        )  # query_img and query_txt
        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(
                qform_all_proj, atts_qform_all_proj, self.prompt_list
            )  # (self, img_embeds, batch_names, atts_img, prompt_list):

        ##### Caption generation
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]  # construct GT text
        text_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(image.device)

        bos = (
            torch.ones(
                [qform_all_proj.shape[0], 1],
                dtype=text_tokens.input_ids.dtype,
                device=text_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]], dtype=torch.long
            )
            .to(image.device)
            .fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, text_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_prompt, text_tokens.attention_mask], dim=1
        )

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"output": outputs[0], "loss": loss}


class EVRCap(EVCap):
    """use llm to refine the generated caption"""

    def __init__(
        self,
        ext_path,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model="",
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template="###Human: {} ###Assistant: ",
        max_txt_len=160,
        end_sym="\n",
        low_resource=False,
        device_8bit=0,
        config=None,
    ):
        super().__init__(
            ext_path,
            vit_model,
            q_former_model,
            img_size,
            drop_path_rate,
            use_grad_checkpoint,
            vit_precision,
            freeze_vit,
            freeze_qformer,
            num_query_token,
            num_query_token_txt,
            topn,
            llama_model,
            prompt_path,
            prompt_template,
            max_txt_len,
            end_sym,
            low_resource,
            device_8bit,
        )

        self.config = config
        self.r2s_prompt = "Generate a short sentence based on context below. Example: 'Several images of someone holding a remote control in their hand.'\n\n- Objects: [OBJECTS]\n\nGenerated Sentence:"
        # self.r2s_prompt = "Generate a short sentence based on context below. Example: 'Several images of someone holding a remote control in their hand.\n\n- Objects: [OBJECTS]\n- Actions: [ACTIONS]\n\nGenerated Sentence:"

    def RT2Cap(self, RTs):
        prompts = []
        for RT in RTs:
            sublist_new = []
            for item in RT:
                if item not in sublist_new:
                    sublist_new.append(item)
                    if len(sublist_new) > self.topn:
                        break
            objects = ", ".join(sublist_new)
            prompt = self.r2s_prompt.replace("[OBJECTS]", objects)
            prompts.append(prompt)

        self.llama_tokenizer.padding_side = "left"

        inputs = self.llama_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.llama_model.generate(
            **inputs,
            max_length=self.max_txt_len,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,  # Adjust temperature for diversity
            do_sample=True,  # Enable sampling to use temperature
            pad_token_id=self.llama_tokenizer.eos_token_id,  # Handle padding correctly
        )

        # decode the output
        captions = [
            self.llama_tokenizer.decode(outputs, skip_special_tokens=True)
            .split("Generated Sentence:")[1]
            .replace('"', "")
            .replace("'", "")
            .strip()
            .split(".")[0]
            .strip()
            for outputs in outputs
        ]

        # filtered_captions = []
        # for i, caption in enumerate(captions):
        #     if len(caption.split()) > 2:
        #         filtered_captions.append(caption)
        #     else:
        #         filtered_captions.append("an image with" + objects[i])

        return captions

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )  # -1 means keep the original size
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            re_txt_list_all = self.retrieve_similar_features(
                query_output_img, self.feat_index, self.ext_base_img_id
            )

            re_txt_list_batch = self.RT2Cap(re_txt_list_all)
            # breakpoint()

            text = self.bert_tokenizer(
                re_txt_list_batch,
                truncation=True,
                padding="longest",
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            query_tokens_txt = self.query_tokens_txt.expand(
                image_embeds.shape[0], -1, -1
            )
            query_atts_txt = torch.ones(
                query_tokens_txt.size()[:-1], dtype=torch.long
            ).to(image_embeds.device)

            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            query_output_img_atts = torch.cat(
                [query_atts_txt, query_output_img_atts], dim=1
            )

            attention_mask = text.attention_mask
            query_outputs_txt = self.Qformer_txt.bert(
                text.input_ids,
                query_embeds=query_tokens_txt,
                attention_mask=attention_mask,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            # the attention_mask is for input tokens but the encoder_attention_mask is for encoder hidden states(here is query_output_img)
            query_output_txt = query_outputs_txt.last_hidden_state[
                :, : query_tokens_txt.size(1), :
            ]  # only keep the txt part

            query_output_all = torch.cat([query_output_img, query_output_txt], dim=1)
            qform_all_proj = self.llama_proj(query_output_all)
            atts_qform_all_proj = torch.ones(
                qform_all_proj.size()[:-1], dtype=torch.long
            ).to(device)
        return qform_all_proj, atts_qform_all_proj


class EVCapGT(EVCap):
    def encode_img(self, image, prompt):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )  # -1 means keep the original size
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            # re_txt_list_all  = self.retrieve_similar_features(query_output_img, self.feat_index, self.ext_base_img_id)
            # re_txt_list_batch = []
            # for sublist in re_txt_list_all:
            #     sublist_new = []
            #     for item in sublist:
            #         if item not in sublist_new:
            #             sublist_new.append(item)
            #             if len(sublist_new)>self.topn:
            #                 break
            #     re_txt_list_batch.append(" [SEP] ".join(sublist_new))

            text = self.bert_tokenizer(
                prompt,
                truncation=True,
                padding="longest",
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            query_tokens_txt = self.query_tokens_txt.expand(
                image_embeds.shape[0], -1, -1
            )
            query_atts_txt = torch.ones(
                query_tokens_txt.size()[:-1], dtype=torch.long
            ).to(image_embeds.device)

            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            query_output_img_atts = torch.cat(
                [query_atts_txt, query_output_img_atts], dim=1
            )

            attention_mask = text.attention_mask
            query_outputs_txt = self.Qformer_txt.bert(
                text.input_ids,
                query_embeds=query_tokens_txt,
                attention_mask=attention_mask,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            # the attention_mask is for input tokens but the encoder_attention_mask is for encoder hidden states(here is query_output_img)
            query_output_txt = query_outputs_txt.last_hidden_state[
                :, : query_tokens_txt.size(1), :
            ]  # only keep the txt part

            query_output_all = torch.cat([query_output_img, query_output_txt], dim=1)
            qform_all_proj = self.llama_proj(query_output_all)
            atts_qform_all_proj = torch.ones(
                qform_all_proj.size()[:-1], dtype=torch.long
            ).to(device)
        return qform_all_proj, atts_qform_all_proj

    def forward(self, samples):
        ##### Image
        image = samples["image"]
        qform_all_proj, atts_qform_all_proj = self.encode_img(
            image, samples["text_input"]
        )  # query_img and query_txt
        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(
                qform_all_proj, atts_qform_all_proj, self.prompt_list
            )  # (self, img_embeds, batch_names, atts_img, prompt_list):

        ##### Caption generation
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]  # construct GT text
        text_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(image.device)

        bos = (
            torch.ones(
                [qform_all_proj.shape[0], 1],
                dtype=text_tokens.input_ids.dtype,
                device=text_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]], dtype=torch.long
            )
            .to(image.device)
            .fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, text_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_prompt, text_tokens.attention_mask], dim=1
        )

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"output": outputs[0], "loss": loss}


class EVCapGT_concat(EVCap):
    def encode_img(self, image, prompt):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )  # -1 means keep the original size
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            # re_txt_list_all  = self.retrieve_similar_features(query_output_img, self.feat_index, self.ext_base_img_id)
            # re_txt_list_batch = []
            # for sublist in re_txt_list_all:
            #     sublist_new = []
            #     for item in sublist:
            #         if item not in sublist_new:
            #             sublist_new.append(item)
            #             if len(sublist_new)>self.topn:
            #                 break
            #     re_txt_list_batch.append(" [SEP] ".join(sublist_new))

            text = self.bert_tokenizer(
                prompt,
                truncation=True,
                padding="longest",
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            query_tokens_txt = self.query_tokens_txt.expand(
                image_embeds.shape[0], -1, -1
            )
            query_atts_txt = torch.ones(
                query_tokens_txt.size()[:-1], dtype=torch.long
            ).to(image_embeds.device)

            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            query_output_img_atts = torch.cat(
                [query_atts_txt, query_output_img_atts], dim=1
            )

            attention_mask = text.attention_mask
            query_outputs_txt = self.Qformer_txt.bert(
                text.input_ids,
                query_embeds=query_tokens_txt,
                attention_mask=attention_mask,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            # the attention_mask is for input tokens but the encoder_attention_mask is for encoder hidden states(here is query_output_img)
            query_output_txt = query_outputs_txt.last_hidden_state[
                :, : query_tokens_txt.size(1), :
            ]  # only keep the txt part

            query_output_all = torch.cat([query_output_img, query_output_txt], dim=1)
            qform_all_proj = self.llama_proj(query_output_all)
            atts_qform_all_proj = torch.ones(
                qform_all_proj.size()[:-1], dtype=torch.long
            ).to(device)
        return qform_all_proj, atts_qform_all_proj


class BaseCap(EVCap):
    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )  # -1 means keep the original size
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            # breakpoint()
            #

            query_output_all = query_output_img
            qform_all_proj = self.llama_proj(query_output_all)
            atts_qform_all_proj = torch.ones(
                qform_all_proj.size()[:-1], dtype=torch.long
            ).to(device)
            # breakpoint()
        return qform_all_proj, atts_qform_all_proj


class ECACap(Blip2Base):
    """
    external compositional attributes memory captioning

    ext_path: path to the external memory (the value is a dict with entity and action rather than object only in EVCap)
    """

    def __init__(
        self,
        ext_path,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model="",
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template="###Human: {} ###Assistant: ",
        max_txt_len=160,
        end_sym="\n",
        low_resource=False,
        device_8bit=0,
        config=None,
    ):
        super().__init__()

        self.low_resource = low_resource
        self.topn = topn
        print("topn:", self.topn)

        ##### Image
        print("Loading VIT")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print("Loading VIT Done")

        print("Loading Q-Former")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = True
            logging.info("freeze Qformer")
        print("Loading Q-Former Done")

        ##### Text
        self.bert_tokenizer = self.init_tokenizer()
        self.Qformer_txt, self.query_tokens_txt = self.init_Qformer_txt(
            num_query_token_txt, self.Qformer.config.hidden_size
        )
        self.Qformer_txt.resize_token_embeddings(len(self.bert_tokenizer))
        self.Qformer_txt.cls = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        if freeze_qformer:
            for name, param in self.Qformer_txt.named_parameters():
                param.requires_grad = False
            self.Qformer_txt = self.Qformer_txt.eval()
            self.Qformer_txt.train = disabled_train
            self.query_tokens_txt.requires_grad = True
            logging.info("freeze Qformer")
        print("query_tokens_txt", self.query_tokens_txt.shape)
        print("Loading Q-Former Done")
        print("Loading Q-Former_txt Done")

        ##### Caption generation
        print("Loading LLAMA")
        # breakpoint()
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
                ignore_mismatched_sizes=True,
                # local_files_only=True,
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                device_map="auto",
                ignore_mismatched_sizes=True,
                local_files_only=True,
            )

        self.llama_tokenizer.bos_token_id = (
            1
            if self.llama_tokenizer.bos_token_id is None
            else self.llama_tokenizer.bos_token_id
        )

        # frozen llama model
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print("Loading LLAMA Done")

        ###
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, "r") as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt
            ]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print("Load {} training prompts".format(len(self.prompt_list)))
            print("Prompt Example \n{}".format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        print(ext_path)
        with open(ext_path, "rb") as f:
            data = pickle.load(f)

            ext_base_img = data["tensor"]
            self.ext_base_attr_list = data["attr_list"]
            feature_library_cpu = ext_base_img.cpu().numpy()
            faiss.normalize_L2(feature_library_cpu)
            self.feat_index = faiss.IndexFlatIP(feature_library_cpu.shape[1])
            self.feat_index.add(feature_library_cpu)
            print(f"loaded external base image from {ext_path}")
            # ext_base_img, self.ext_base_img_id = pickle.load(f)
            # print(ext_base_img.shape, len(self.ext_base_img_id))
            # feature_library_cpu = ext_base_img.cpu().numpy()
            # faiss.normalize_L2(feature_library_cpu)
            # self.feat_index = faiss.IndexFlatIP(feature_library_cpu.shape[1])
            # self.feat_index.add(feature_library_cpu)
            # print(f"loaded external base image")

        self.config = config

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def prompt_wrap(self, img_embeds, atts_img, prompt_list):
        if prompt_list:
            batch_size = img_embeds.shape[0]
            emb_lists = []
            for i in range(batch_size):
                prompt = random.choice(prompt_list)
                p_before, p_after = prompt.split("<ImageHere>", 1)
                self.llama_tokenizer.padding_side = "right"
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(img_embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(
                    p_before_tokens.input_ids
                )
                p_after_embeds = self.llama_model.model.embed_tokens(
                    p_after_tokens.input_ids
                )
                img_embeds_i = img_embeds[i].unsqueeze(0)
                wrapped_embed_i = torch.cat(
                    [p_before_embeds, img_embeds_i, p_after_embeds], dim=1
                )
                emb_lists.append(wrapped_embed_i)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.llama_model.model.embed_tokens(
                torch.tensor(
                    self.llama_tokenizer.pad_token_id, device=img_embeds.device
                )
            )
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros(
                [len(emb_lens), max(emb_lens)],
                dtype=torch.int,
                device=img_embeds.device,
            )
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, : emb_lens[i]] = emb
                wrapped_atts[i, : emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img

    def pre_name(self, caption):
        caption = re.sub(
            r"([_!,'\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption

    def retrieve_similar_features(
        self, query_features, feat_index, image_id, top_k=5, sub_top_k=32
    ):
        """
        return: [[dict,...],...]
        """
        batch_size, nums, dims = query_features.shape
        query_features = query_features.view(
            -1, dims
        )  # include patch feature when retrieval

        query_features_cpu = query_features.detach().cpu().numpy()
        faiss.normalize_L2(query_features_cpu)
        top_k_similarities, top_k_indices = feat_index.search(query_features_cpu, top_k)

        top_k_indices = torch.tensor(top_k_indices).to(device=query_features.device)
        top_k_similarities = torch.tensor(top_k_similarities).to(
            device=query_features.device
        )
        top_k_similarities = top_k_similarities.view(batch_size, -1)

        indices = top_k_indices.view(batch_size, -1)

        re_txt_list_all = []
        for batch_i in range(batch_size):
            indices_list = indices[batch_i]
            re_txt_batch_list = []
            for (
                i
            ) in (
                indices_list
            ):  # the index is for image_id rather than topk_indices/topk_similarities
                re_txt_batch_list.append(image_id[i])
            re_txt_list_all.append(re_txt_batch_list)  # [dict,...]

        sorted_batched_ret = []
        for listA, listB in zip(top_k_similarities, re_txt_list_all):
            sorted_listA, indices = listA.sort(descending=True)
            sorted_listB = [listB[idx] for idx in indices]
            sorted_listB = sorted_listB[:sub_top_k]
            sorted_batched_ret.append(sorted_listB)
        return sorted_batched_ret

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )  # -1 means keep the original size
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            re_txt_list_all = self.retrieve_similar_features(
                query_output_img, self.feat_index, self.ext_base_attr_list
            )

            # process the entity and action
            re_txt_list_batch = [
                create_caption_from_retrievals(re_txt, min_returned=self.config["topn"])
                for re_txt in re_txt_list_all
            ]

            # old implementation
            # re_txt_list_batch = []
            # for sublist in re_txt_list_all:
            #     sublist_new = []
            #     for item in sublist:
            #         if item not in sublist_new:
            #             sublist_new.append(item)
            #             if len(sublist_new)>self.topn:
            #                 break
            #     re_txt_list_batch.append(" [SEP] ".join(sublist_new))

            text = self.bert_tokenizer(
                re_txt_list_batch,
                truncation=True,
                padding="longest",
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            query_tokens_txt = self.query_tokens_txt.expand(
                image_embeds.shape[0], -1, -1
            )
            query_atts_txt = torch.ones(
                query_tokens_txt.size()[:-1], dtype=torch.long
            ).to(image_embeds.device)

            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            query_output_img_atts = torch.cat(
                [query_atts_txt, query_output_img_atts], dim=1
            )

            attention_mask = text.attention_mask
            query_outputs_txt = self.Qformer_txt.bert(
                text.input_ids,
                query_embeds=query_tokens_txt,
                attention_mask=attention_mask,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            # the attention_mask is for input tokens but the encoder_attention_mask is for encoder hidden states(here is query_output_img)
            query_output_txt = query_outputs_txt.last_hidden_state[
                :, : query_tokens_txt.size(1), :
            ]  # only keep the txt part

            query_output_all = torch.cat([query_output_img, query_output_txt], dim=1)
            qform_all_proj = self.llama_proj(query_output_all)
            atts_qform_all_proj = torch.ones(
                qform_all_proj.size()[:-1], dtype=torch.long
            ).to(device)
        return qform_all_proj, atts_qform_all_proj

    def forward(self, samples):
        ##### Image
        image = samples["image"]
        qform_all_proj, atts_qform_all_proj = self.encode_img(
            image
        )  # query_img and query_txt
        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(
                qform_all_proj, atts_qform_all_proj, self.prompt_list
            )  # (self, img_embeds, batch_names, atts_img, prompt_list):

        ##### Caption generation
        # self.llama_tokenizer.padding_side = "left"
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]  # construct GT text
        text_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(image.device)

        # breakpoint()
        bos = (
            torch.ones(
                [qform_all_proj.shape[0], 1],
                dtype=text_tokens.input_ids.dtype,
                device=text_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]], dtype=torch.long
            )
            .to(image.device)
            .fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, text_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_prompt, text_tokens.attention_mask], dim=1
        )

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        # breakpoint()

        # # debug the predict text
        # predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        # decoded_texts = [
        #     self.llama_tokenizer.decode(ids) for ids in predicted_token_ids
        # ]

        # for i in range(len(decoded_texts)):
        #     print(f"Predicted: {decoded_texts[i]}")
        #     # print(f"GT: {text[i]}")
        #     # print()

        return {"output": outputs[0], "loss": loss}


class ECARCap(ECACap):
    def __init__(
        self,
        ext_path,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model="",
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template="###Human: {} ###Assistant: ",
        max_txt_len=160,
        end_sym="\n",
        low_resource=False,
        device_8bit=0,
        config=None,
    ):
        super().__init__(
            ext_path,
            vit_model,
            q_former_model,
            img_size,
            drop_path_rate,
            use_grad_checkpoint,
            vit_precision,
            freeze_vit,
            freeze_qformer,
            num_query_token,
            num_query_token_txt,
            topn,
            llama_model,
            prompt_path,
            prompt_template,
            max_txt_len,
            end_sym,
            low_resource,
            device_8bit,
            config,
        )

        self.r2s_prompt = "Generate a short sentence based on context below. Example: 'Several images of someone holding a remote control in their hand.\n\n- Objects: [OBJECTS]\n- Actions: [ACTIONS]\n- Scenes: [SCENES]\n\nGenerated Sentence:"

    def RT2Cap(self, RTs):
        entities_actions = [filter_retrieval_results(RT) for RT in RTs]

        entities = [ea["entity"] for ea in entities_actions]
        actions = [ea["action"] for ea in entities_actions]
        if "scene" in entities_actions[0]:
            scenes = [ea["scene"] for ea in entities_actions]
            has_scene = True
        else:
            has_scene = False
            scenes = [[] for _ in range(len(entities))]

        entity_part = [", ".join(entity) for entity in entities]
        action_part = [", ".join(action) for action in actions]
        scene_part = [", ".join(scene) for scene in scenes]

        prompts = []
        for i in range(len(RTs)):
            prompt = self.r2s_prompt.replace("[OBJECTS]", entity_part[i]).replace(
                "[ACTIONS]", action_part[i]
            )
            prompt = (
                prompt.replace("[SCENES]", scene_part[i])
                if len(scenes[i]) > 0
                else prompt.replace("\n- Scenes: [SCENES]", "")
            )
            # if len(scenes[i]) == 0:
            #     print(prompt)

            prompts.append(prompt)

        self.llama_tokenizer.padding_side = "left"
        inputs = self.llama_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.llama_model.generate(
            **inputs,
            max_length=self.max_txt_len,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,  # Adjust temperature for diversity
            do_sample=True,  # Enable sampling to use temperature
            pad_token_id=self.llama_tokenizer.eos_token_id,  # Handle padding correctly
        )

        # decode the output
        captions = [
            self.llama_tokenizer.decode(outputs, skip_special_tokens=True)
            .split("Generated Sentence:")[1]
            .replace('"', "")
            .replace("'", "")
            .strip()
            .split(".")[0]
            .strip()
            for outputs in outputs
        ]

        # filtered_captions = []
        # for i, caption in enumerate(captions):
        #     if len(caption.split()) > 2:
        #         filtered_captions.append(caption)
        #     else:
        #         filtered_captions.append("an image with" + objects[i])

        return captions

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )  # -1 means keep the original size
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            re_txt_list_all = self.retrieve_similar_features(
                query_output_img, self.feat_index, self.ext_base_attr_list
            )

            re_txt_list_batch = self.RT2Cap(re_txt_list_all)
            # breakpoint()

            text = self.bert_tokenizer(
                re_txt_list_batch,
                truncation=True,
                padding="longest",
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            query_tokens_txt = self.query_tokens_txt.expand(
                image_embeds.shape[0], -1, -1
            )
            query_atts_txt = torch.ones(
                query_tokens_txt.size()[:-1], dtype=torch.long
            ).to(image_embeds.device)

            query_output_img_atts = torch.ones(
                query_output_img.size()[:-1], dtype=torch.long
            ).to(device)
            query_output_img_atts = torch.cat(
                [query_atts_txt, query_output_img_atts], dim=1
            )

            attention_mask = text.attention_mask
            query_outputs_txt = self.Qformer_txt.bert(
                text.input_ids,
                query_embeds=query_tokens_txt,
                attention_mask=attention_mask,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            # the attention_mask is for input tokens but the encoder_attention_mask is for encoder hidden states(here is query_output_img)
            query_output_txt = query_outputs_txt.last_hidden_state[
                :, : query_tokens_txt.size(1), :
            ]  # only keep the txt part

            query_output_all = torch.cat([query_output_img, query_output_txt], dim=1)
            qform_all_proj = self.llama_proj(query_output_all)
            atts_qform_all_proj = torch.ones(
                qform_all_proj.size()[:-1], dtype=torch.long
            ).to(device)
        return qform_all_proj, atts_qform_all_proj
