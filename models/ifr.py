"""
mt: multi-task learning (caption & diffusion/generation)
"""

import logging
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

# from torchvision.transforms import functional as F
import random
from models.blip2 import Blip2Base, disabled_train
from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import pickle
import faiss
import os
import re
from sentence_transformers import SentenceTransformer

from transformers import CLIPTextModel, CLIPTokenizer, AutoImageProcessor, AutoModel
from models.evcap import EVCap
import models.clip_bank as clip_bank
from models.evcap import (
    create_caption_from_retrievals,
    create_caption_from_retrievals_sim,
)
import numpy as np
from collections import Counter, defaultdict, OrderedDict
import copy
from sklearn.decomposition import PCA
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch.nn as nn


DINO_FEATURE_DIM = 768

def select_most_different_features(feat1, feat2, num_selected_token):
    """
    Selects the most different features from feat2 compared to feat1 based on average similarity.

    Args:
        feat1 (torch.Tensor): Tensor of shape [N, num_query1, dim]
        feat2 (torch.Tensor): Tensor of shape [N, num_query2, dim]
        num_selected_token (int): Number of features to select from feat2.

    Returns:
        torch.Tensor: Selected features from feat2 of shape [N, num_selected_token, dim]
    """
    N, num_query1, dim = feat1.shape
    num_query2 = feat2.shape[1]

    # Normalize features to compute cosine similarity
    feat1_norm = torch.nn.functional.normalize(
        feat1, p=2, dim=-1
    )  # [N, num_query1, dim]
    feat2_norm = torch.nn.functional.normalize(
        feat2, p=2, dim=-1
    )  # [N, num_query2, dim]

    # Compute cosine similarity between feat2 and feat1 -> [N, num_query2, num_query1]
    similarity = torch.bmm(feat2_norm, feat1_norm.transpose(1, 2))

    # Compute average similarity for each feature in feat2 -> [N, num_query2]
    avg_similarity = similarity.mean(dim=2)

    # Select indices of the `num_selected_token` features with the lowest average similarity
    selected_indices = torch.topk(
        -avg_similarity, num_selected_token, dim=1
    ).indices  # Negative to get lowest values

    # Gather selected features from feat2
    selected_features = torch.gather(
        feat2, 1, selected_indices.unsqueeze(-1).expand(-1, -1, dim)
    )

    return selected_features  # Shape [N, num_selected_token, dim]


def pca_feature_reduction(tensor, source_dim, target_dim):
    """
    Reduces the feature dimension of a tensor using PCA.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, T, F], where F is the feature dimension.
        source_dim (int): The index of the dimension to be reduced.
        target_dim (int): The target feature dimension after PCA.

    Returns:
        torch.Tensor: Tensor with reduced feature dimension.
    """
    # Move the source dimension to the last position
    tensor = tensor.permute(
        *[i for i in range(tensor.ndim) if i != source_dim], source_dim
    )
    original_shape = tensor.shape
    flat_tensor = tensor.reshape(-1, original_shape[-1])

    # Apply PCA
    pca = PCA(n_components=target_dim)
    reduced_features = pca.fit_transform(flat_tensor.cpu().numpy())

    # Convert back to tensor and reshape to original format
    reduced_tensor = torch.tensor(
        reduced_features, dtype=tensor.dtype, device=tensor.device
    )
    new_shape = original_shape[:-1] + (target_dim,)
    reduced_tensor = reduced_tensor.reshape(new_shape)

    # Move the dimensions back to their original order
    inv_permute = list(range(tensor.ndim - 1))
    inv_permute.insert(source_dim, tensor.ndim - 1)
    reduced_tensor = reduced_tensor.permute(*inv_permute)

    return reduced_tensor


# Ensure the input tensor is cast to float32 before performing QR decomposition
def safe_qr(matrix):
    """
    Perform QR decomposition safely by casting to float32 if necessary.

    Args:
        matrix (torch.Tensor): Input matrix of shape [M, N].

    Returns:
        Q (torch.Tensor): Orthogonal matrix Q from QR decomposition.
        R (torch.Tensor): Upper triangular matrix R from QR decomposition.
    """
    original_dtype = matrix.dtype  # Save the original dtype (e.g., torch.float16)
    if matrix.dtype == torch.float16:
        matrix = matrix.to(torch.float32)  # Cast to float32 for numerical stability

    # Perform QR decomposition
    Q, R = torch.linalg.qr(matrix)

    # Cast back to the original dtype if necessary
    Q = Q.to(original_dtype)
    R = R.to(original_dtype)

    return Q, R


def pca_feature_reduction_gpu(tensor, source_dim, target_dim):
    """
    Reduces the feature dimension of a tensor using PCA on GPU.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, T, F], where F is the feature dimension.
        source_dim (int): The index of the dimension to be reduced.
        target_dim (int): The target feature dimension after PCA.

    Returns:
        torch.Tensor: Tensor with reduced feature dimension.
    """
    # Move the source dimension to the last position
    permute_order = [i for i in range(tensor.ndim) if i != source_dim] + [source_dim]
    tensor = tensor.permute(*permute_order)
    original_shape = tensor.shape
    flat_tensor = tensor.reshape(-1, original_shape[-1])

    # Ensure the tensor is on the GPU
    device = tensor.device
    flat_tensor = flat_tensor.to(device)
    # breakpoint()

    # Apply PCA using torch.pca_lowrank
    U, S, Vt = torch.pca_lowrank(flat_tensor, q=target_dim, center=True, niter=2)

    # Select the top `target_dim` principal components
    reduced_features = torch.matmul(U[:, :target_dim], torch.diag(S[:target_dim]))

    # Convert back to tensor and reshape to original format
    new_shape = original_shape[:-1] + (target_dim,)
    reduced_tensor = reduced_features.reshape(new_shape)

    # Move the dimensions back to their original order
    inv_permute = list(range(tensor.ndim - 1))
    inv_permute.insert(source_dim, tensor.ndim - 1)
    reduced_tensor = reduced_tensor.permute(*inv_permute)

    return reduced_tensor


def compute_compression_loss(
    raw_features, compressed_features, raw_features_dino, normalize=True
):
    """
    Computes the information retention and diversity losses for token compression over a batch.

    Args:
        raw_features (torch.Tensor): Raw image features of shape [batch, num_token, feat_dim]
        compressed_features (torch.Tensor): Compressed features of shape [batch, small_num_tokens, feat_dim]
        lambda_diversity (float): Weight for the diversity loss term (default: 1.0)
        normalize (bool): Whether to normalize features for cosine similarity (default: True)

    Returns:
        dict: Dictionary containing:
            - 'info_loss': Information retention loss (L_info), averaged over batch
            - 'diversity_loss': Diversity loss (L_div), averaged over batch
            - 'total_loss': Combined loss (L_info + lambda * L_div), averaged over batch
    """
    # Get dimensions
    if raw_features is not None:
        batch_size, num_token, feat_dim = raw_features.shape
        batch_size_comp, small_num_tokens, feat_dim_comp = compressed_features.shape
        assert batch_size == batch_size_comp, "Batch sizes must match"
        assert feat_dim == feat_dim_comp, "Feature dimensions must match"
    else:
        batch_size_comp, small_num_tokens, feat_dim_comp = compressed_features.shape

    # Normalize features if specified (for cosine similarity)
    if normalize:
        raw_features = (
            F.normalize(raw_features, p=2, dim=-1) if raw_features is not None else None
        )  # [batch, num_token, feat_dim]
        compressed_features = F.normalize(
            compressed_features, p=2, dim=-1
        )  # [batch, small_num_tokens, feat_dim]
        raw_features_dino = (
            F.normalize(raw_features_dino, p=2, dim=-1)
            if raw_features_dino is not None
            else None
        )

    # clip info loss
    if raw_features is not None:
        # --- Information Retention Loss (L_info) ---
        # Compute pairwise cosine similarity: [batch, num_token, small_num_tokens]
        # Reshape for batched matmul: [batch, num_token, feat_dim] @ [batch, feat_dim, small_num_tokens]
        similarity_matrix = torch.bmm(raw_features, compressed_features.transpose(1, 2))
        # For each raw token, take the max similarity to any compressed token: [batch, num_token]
        max_similarities = similarity_matrix.max(dim=2)[0]
        # Average over tokens and batch
        info_loss = -max_similarities.mean(dim=1).mean()  # Scalar (mean over batch)
    else:
        info_loss = torch.tensor(0.0, device=compressed_features.device)

    # dino info loss
    # breakpoint()
    if raw_features_dino is not None:
        similarity_matrix = torch.bmm(
            raw_features_dino, compressed_features.transpose(1, 2)
        )
        max_similarities = similarity_matrix.max(dim=2)[0]

        info_loss_dino = -max_similarities.mean(dim=1).mean()
    else:
        info_loss_dino = torch.tensor(0.0, device=compressed_features.device)

    # --- Diversity Loss (L_div) ---
    # Compute pairwise cosine similarities: [batch, small_num_tokens, small_num_tokens]
    compressed_sim_matrix = torch.bmm(
        compressed_features, compressed_features.transpose(1, 2)
    )
    # Zero out diagonal and get upper triangle (j < k)
    mask = torch.triu(torch.ones(small_num_tokens, small_num_tokens), diagonal=1).bool()
    mask = mask.to(compressed_features.device)  # Ensure mask is on the same device
    # Extract pairwise similarities for each batch: [batch, num_pairs]
    pairwise_sims = compressed_sim_matrix[:, mask]
    num_pairs = small_num_tokens * (small_num_tokens - 1) // 2
    if num_pairs > 0:
        # Average over pairs and batch
        diversity_loss = pairwise_sims.sum(dim=1) / num_pairs  # [batch]
        diversity_loss = diversity_loss.mean()  # Scalar (mean over batch)
    else:
        diversity_loss = torch.tensor(
            0.0, device=compressed_features.device
        )  # Edge case

    # --- Combined Loss ---
    # total_loss = info_loss + lambda_diversity * diversity_loss

    return {
        "info_loss": info_loss,
        "diversity_loss": diversity_loss,
        "info_loss_dino": info_loss_dino,
    }

def compute_div_loss(
    compressed_features, normalize=True
):
    """
    Computes the diversity losses.

    Args:
        compressed_features (torch.Tensor): Compressed features of shape [batch, small_num_tokens, feat_dim]
        lambda_diversity (float): Weight for the diversity loss term (default: 1.0)
        normalize (bool): Whether to normalize features for cosine similarity (default: True)

    Returns:
        dict: Dictionary containing:
            - 'info_loss': Information retention loss (L_info), averaged over batch
            - 'diversity_loss': Diversity loss (L_div), averaged over batch
            - 'total_loss': Combined loss (L_info + lambda * L_div), averaged over batch
    """
    # Get dimensions
    _, small_num_tokens, _ = compressed_features.shape

    # Normalize features if specified (for cosine similarity)
    if normalize:
        compressed_features = F.normalize(
            compressed_features, p=2, dim=-1
        )  # [batch, small_num_tokens, feat_dim]

    # --- Diversity Loss (L_div) ---
    # Compute pairwise cosine similarities: [batch, small_num_tokens, small_num_tokens]
    compressed_sim_matrix = torch.bmm(
        compressed_features, compressed_features.transpose(1, 2)
    )
    # Zero out diagonal and get upper triangle (j < k)
    mask = torch.triu(torch.ones(small_num_tokens, small_num_tokens), diagonal=1).bool()
    mask = mask.to(compressed_features.device)  # Ensure mask is on the same device
    # Extract pairwise similarities for each batch: [batch, num_pairs]
    pairwise_sims = compressed_sim_matrix[:, mask]
    num_pairs = small_num_tokens * (small_num_tokens - 1) // 2
    if num_pairs > 0:
        # Average over pairs and batch
        diversity_loss = pairwise_sims.sum(dim=1) / num_pairs  # [batch]
        diversity_loss = diversity_loss.mean()  # Scalar (mean over batch)
    else:
        diversity_loss = torch.tensor(
            0.0, device=compressed_features.device
        )  # Edge case

    return diversity_loss


def compute_compression_loss_seperate(
    raw_features,
    compressed_features,
    raw_features_dino,
    normalize=True,
    num_clip_token=32,
):
    """
    Computes the information retention and diversity losses for token compression over a batch. consider that only the clip feature is optimized

    Args:
        raw_features (torch.Tensor): Raw image features of shape [batch, num_token, feat_dim]
        compressed_features (torch.Tensor): Compressed features of shape [batch, small_num_tokens, feat_dim]
        lambda_diversity (float): Weight for the diversity loss term (default: 1.0)
        normalize (bool): Whether to normalize features for cosine similarity (default: True)
        num_clip_token: 32 (fixed)

    Returns:
        dict: Dictionary containing:
            - 'info_loss': Information retention loss (L_info), averaged over batch
            - 'diversity_loss': Diversity loss (L_div), averaged over batch
            - 'total_loss': Combined loss (L_info + lambda * L_div), averaged over batch
    """
    # Get dimensions
    # breakpoint()
    if raw_features is not None:
        batch_size, num_token, feat_dim = raw_features.shape
        batch_size_comp, small_num_tokens, feat_dim_comp = compressed_features.shape
        assert batch_size == batch_size_comp, "Batch sizes must match"
        assert feat_dim == feat_dim_comp, "Feature dimensions must match"
    else:
        batch_size_comp, small_num_tokens, feat_dim_comp = compressed_features.shape

    # Normalize features if specified (for cosine similarity)
    if normalize:
        raw_features = (
            F.normalize(raw_features, p=2, dim=-1) if raw_features is not None else None
        )  # [batch, num_token, feat_dim]
        compressed_features = F.normalize(
            compressed_features, p=2, dim=-1
        )  # [batch, small_num_tokens, feat_dim]
        raw_features_dino = (
            F.normalize(raw_features_dino, p=2, dim=-1)
            if raw_features_dino is not None
            else None
        )

    compressed_clip_features = compressed_features[:, :num_clip_token]
    compressed_dino_features = compressed_features[:, num_clip_token:]

    # clip info loss
    if raw_features is not None:
        # --- Information Retention Loss (L_info) ---
        # Compute pairwise cosine similarity: [batch, num_token, small_num_tokens]
        # Reshape for batched matmul: [batch, num_token, feat_dim] @ [batch, feat_dim, small_num_tokens]
        similarity_matrix = torch.bmm(
            raw_features, compressed_clip_features.transpose(1, 2)
        )
        # For each raw token, take the max similarity to any compressed token: [batch, num_token]
        max_similarities = similarity_matrix.max(dim=2)[0]
        # Average over tokens and batch
        info_loss = -max_similarities.mean(dim=1).mean()  # Scalar (mean over batch)
    else:
        info_loss = torch.tensor(0.0, device=compressed_features.device)

    # dino info loss
    # breakpoint()
    if raw_features_dino is not None:
        similarity_matrix = torch.bmm(
            raw_features_dino, compressed_clip_features.transpose(1, 2)
        )
        select_dino_similarity_matrix = torch.bmm(
            compressed_dino_features, compressed_clip_features.transpose(1, 2)
        )

        max_similarities = similarity_matrix.max(dim=2)[0]
        select_dino_max_similarities = select_dino_similarity_matrix.max(dim=2)[0]

        info_loss_dino = (
            -max_similarities.mean(dim=1).mean()
            + select_dino_max_similarities.mean(dim=1).mean()
        )
    else:
        info_loss_dino = torch.tensor(0.0, device=compressed_features.device)

    # --- Diversity Loss (L_div) ---
    # Compute pairwise cosine similarities: [batch, small_num_tokens, small_num_tokens]
    compressed_sim_matrix = torch.bmm(
        compressed_features, compressed_features.transpose(1, 2)
    )
    # Zero out diagonal and get upper triangle (j < k)
    mask = torch.triu(torch.ones(small_num_tokens, small_num_tokens), diagonal=1).bool()
    mask = mask.to(compressed_features.device)  # Ensure mask is on the same device
    # Extract pairwise similarities for each batch: [batch, num_pairs]
    pairwise_sims = compressed_sim_matrix[:, mask]
    num_pairs = small_num_tokens * (small_num_tokens - 1) // 2
    if num_pairs > 0:
        # Average over pairs and batch
        diversity_loss = pairwise_sims.sum(dim=1) / num_pairs  # [batch]
        diversity_loss = diversity_loss.mean()  # Scalar (mean over batch)
    else:
        diversity_loss = torch.tensor(
            0.0, device=compressed_features.device
        )  # Edge case

    # --- Combined Loss ---
    # total_loss = info_loss + lambda_diversity * diversity_loss

    return {
        "info_loss": info_loss,
        "diversity_loss": diversity_loss,
        "info_loss_dino": info_loss_dino,
    }


def compute_compression_loss2(
    raw_features,
    compressed_features,
    raw_features_dino,
    normalize=True,
    temperature=1.0,
    diversity_power=2,
    sim_offset=0.5,
):
    """
    Compression loss with normalized information retention and controlled diversity.

    Args:
        sim_offset: Shifts similarity range to [0, 1] when using normalize=True
    """
    device = compressed_features.device

    # Feature normalization
    if normalize:
        if raw_features is not None:
            raw_features = F.normalize(raw_features, p=2, dim=-1)
        if raw_features_dino is not None:
            raw_features_dino = F.normalize(raw_features_dino, p=2, dim=-1)
        compressed_features = F.normalize(compressed_features, p=2, dim=-1)

    # Information Retention Loss (normalized)
    if raw_features is not None:
        # Shift similarity to positive range [0, 1] when normalized
        similarity = torch.bmm(raw_features, compressed_features.transpose(1, 2))
        if normalize:
            similarity = (similarity + 1) * 0.5  # Shift from [-1,1] to [0,1]

        # Stable smooth maximum calculation
        logits = similarity * temperature
        log_sum_exp = torch.logsumexp(logits, dim=2)  # [batch, num_token]

        # Normalization factor (batch_size, num_compressed)
        num_compressed = compressed_features.size(1)
        smooth_max = (log_sum_exp / temperature) - torch.log(
            torch.tensor(num_compressed, device=device)
        )

        info_loss = -smooth_max.mean()  # Now bounded in [-log(N), 0]

        # Additional normalization (scale to ~[-1, 0] range)
        max_scale = torch.log(torch.tensor(num_compressed, device=device))
        info_loss = info_loss / max_scale  # Final range ~[-1, 0]
    else:
        info_loss = torch.tensor(0.0, device=device)

    if raw_features_dino is not None:
        # Shift similarity to positive range [0, 1] when normalized
        similarity = torch.bmm(raw_features_dino, compressed_features.transpose(1, 2))
        if normalize:
            similarity = (similarity + 1) * 0.5  # Shift from [-1,1] to [0,1]

        # Stable smooth maximum calculation
        logits = similarity * temperature
        log_sum_exp = torch.logsumexp(logits, dim=2)  # [batch, num_token]

        # Normalization factor (batch_size, num_compressed)
        num_compressed = compressed_features.size(1)
        smooth_max = (log_sum_exp / temperature) - torch.log(
            torch.tensor(num_compressed, device=device)
        )

        info_loss_dino = -smooth_max.mean()  # Now bounded in [-log(N), 0]

        # Additional normalization (scale to ~[-1, 0] range)
        max_scale = torch.log(torch.tensor(num_compressed, device=device))
        info_loss_dino = info_loss_dino / max_scale  # Final range ~[-1, 0]
    else:
        info_loss_dino = torch.tensor(0.0, device=device)

    # Diversity Loss (controlled magnitude)
    small_num = compressed_features.size(1)
    if small_num > 1:
        mask = torch.triu(
            torch.ones(small_num, small_num, device=device), diagonal=1
        ).bool()
        sim_matrix = torch.bmm(compressed_features, compressed_features.transpose(1, 2))

        if normalize:
            sim_matrix = (sim_matrix + 1) * 0.5  # [0,1] range

        pairwise_sims = sim_matrix[:, mask]

        # Calculate diversity loss based on power
        if diversity_power == 1:
            diversity_loss = pairwise_sims.mean()
        elif diversity_power == 2:
            diversity_loss = (pairwise_sims**2).mean()
        else:
            diversity_loss = torch.abs(pairwise_sims).mean()
    else:
        diversity_loss = torch.tensor(0.0, device=device)

    return {
        "info_loss": info_loss,
        "diversity_loss": diversity_loss,
        "info_loss_dino": info_loss_dino,
    }


class TokenDownsamplerPool(nn.Module):
    def __init__(self, input_dim, num_tokens_out, pooling_type="mean"):
        """
        Args:
            input_dim (int): The feature dimension (L) of the input tokens.
            num_tokens_out (int): The target number of tokens after downsampling (C').
            pooling_type (str): Type of pooling to use - 'mean', 'max', or 'sum'.
        """
        super(TokenDownsamplerPool, self).__init__()
        self.num_tokens_out = num_tokens_out
        self.pooling_type = pooling_type

        # Calculate the stride for pooling
        # We'll need this to determine how to split the tokens
        assert num_tokens_out > 0, "num_tokens_out must be positive"

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [N, C, L].

        Returns:
            torch.Tensor: Downsampled tensor of shape [N, C', L].
        """
        N, C, L = x.shape

        # If we're asked to return the same number of tokens, just return input
        if C == self.num_tokens_out:
            return x

        # If we need to downsample
        if C > self.num_tokens_out:
            # Calculate how many tokens to merge for each output token
            tokens_per_group = C // self.num_tokens_out
            remainder = C % self.num_tokens_out

            # Split tokens into groups
            # First handle the remainder by distributing 1 extra token to the first 'remainder' groups
            split_sizes = [
                tokens_per_group + 1 if i < remainder else tokens_per_group
                for i in range(self.num_tokens_out)
            ]

            # Split the tokens along the sequence dimension
            grouped_tokens = torch.split(x, split_sizes, dim=1)

            # Apply pooling to each group
            pooled_tokens = []
            for group in grouped_tokens:
                if self.pooling_type == "mean":
                    pooled = group.mean(dim=1, keepdim=True)
                elif self.pooling_type == "max":
                    pooled = group.max(dim=1, keepdim=True)[0]
                elif self.pooling_type == "sum":
                    pooled = group.sum(dim=1, keepdim=True)
                else:
                    raise ValueError(f"Unknown pooling type: {self.pooling_type}")
                pooled_tokens.append(pooled)

            # Concatenate all pooled tokens
            return torch.cat(pooled_tokens, dim=1)

        # If we need to upsample (just repeat tokens in this simple version)
        else:
            repeat_factor = self.num_tokens_out // C
            remainder = self.num_tokens_out % C

            # Repeat the tokens
            repeated = x.repeat(1, repeat_factor, 1)

            # Handle remainder by taking the first 'remainder' tokens
            if remainder > 0:
                repeated = torch.cat([repeated, x[:, :remainder, :]], dim=1)

            return repeated


class TokenDownsampler(nn.Module):
    def __init__(self, input_dim, num_tokens_out):
        """
        Args:
            input_dim (int): The feature dimension (L) of the input tokens.
            num_tokens_out (int): The target number of tokens after downsampling (C').
        """
        super(TokenDownsampler, self).__init__()

        # Learnable query vectors for attention (C' x L)
        self.learnable_queries = nn.Parameter(torch.randn(num_tokens_out, input_dim))

        # Small linear layer to project token features for attention computation
        self.attention_proj = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [N, C, L].

        Returns:
            torch.Tensor: Downsampled tensor of shape [N, C', L].
        """
        N, C, L = x.shape  # Batch size, number of tokens, feature dimension

        # Step 1: Project input tokens through a linear layer for attention computation
        projected_tokens = self.attention_proj(x)  # Shape: [N, C, L]

        # Step 2: Compute attention scores between tokens and learnable queries
        # Reshape learnable queries to match batch size: [1, C', L] -> [N, C', L]
        queries = self.learnable_queries.unsqueeze(0).expand(
            N, -1, -1
        )  # Shape: [N, C', L]

        # Compute dot-product attention scores
        attention_scores = torch.bmm(
            queries, projected_tokens.transpose(1, 2)
        )  # Shape: [N, C', C]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Normalize over C

        # Step 3: Aggregate tokens using attention weights
        downsampled_tokens = torch.bmm(attention_weights, x)  # Shape: [N, C', L]

        return downsampled_tokens


"""
Key Parameters:
- imgtxt_fusion: whether use Q-Former_text
- raw_rag_text_input: whether input raw RAG text to LLM
- cot: add a new model. (base model with rag text(as output) + cot)
"""


class FeatAlignCap(Blip2Base):
    """
    Base model by align compressed feature to original clip feature and dino feature
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

        # Image
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
        # breakpoint()
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

        # Caption generation
        print("Loading LLAMA")
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            llama_model, use_fast=False
        )
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            # self.llama_model = LlamaForCausalLM.from_pretrained(
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                # device_map="auto"
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

        config["num_query_token"] = num_query_token
        config["num_query_token_txt"] = num_query_token_txt

        self.pattern_dictionary = {"None": [""]}
        config.actual_bs = len(self.pattern_dictionary[config.visual_pattern])

        if config.weight["info_dino"] != 0:
            # self.dino_processor = AutoImageProcessor.from_pretrained(
            #     "facebook/dinov2-base"
            # )
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
            for param in self.dino_model.parameters():
                param.requires_grad = False
        else:
            self.dino_model = None

        if config.weight["info"] != 0 and config.feat_ds == "linear":
            self.feature_downsample = nn.Linear(1408, 768)
            # self.feature_downsample = nn.Linear(768, 1408)
        if "gpu_pca" not in config:
            config.gpu_pca = False
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
                # breakpoint()
                p_before, p_after = prompt.split("<ImageHere>", 1)
                # p_after = p_after + ' [SEP] Related objects: ' + self.raw_test[i] # add raw text
                # breakpoint()
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

    def forward(self, samples):
        ##### Image
        image = samples["image"]
        device = image.device
        # TODO return a dict to contain many terms
        forward_outputs = self.encode_img_woproj(image)  # query_img and query_txt

        raw_image_feature = forward_outputs["raw_image_feature"]
        query_output_img = forward_outputs["query_output_img"]
        query_output_all = query_output_img
        raw_image_feature_dino = forward_outputs["raw_image_feature_dino"]

        # language model loss (caption generation)
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(device)

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
        loss_cap = outputs.loss

        # if self.config.reconstrut_loss == 0 and self.config.diverse_loss == 0:
        #     return {"output": outputs[0], "loss": loss_cap}

        if self.training:
            # reconstruction loss
            if self.config.feat_ds == "linear":
                # breakpoint()
                query_output_img_decoder = query_output_img
                raw_image_feature_decoder = self.feature_downsample(raw_image_feature)
            elif self.config.feat_ds == "pca":
                if self.config.gpu_pca:
                    raw_image_feature_decoder = pca_feature_reduction_gpu(
                        raw_image_feature, 2, 768
                    )
                else:
                    raw_image_feature_decoder = pca_feature_reduction(
                        raw_image_feature, 2, 768
                    )
                query_output_img_decoder = query_output_img
            elif self.config.feat_ds == "none":
                raw_image_feature_decoder = None
                query_output_img_decoder = query_output_img

            if self.config.weight["info"] == 0:
                raw_image_feature_decoder = None
            if self.config.weight["info_dino"] == 0:
                raw_image_feature_dino = None

            if "loss_v" not in self.config:
                reconstruction_losses = compute_compression_loss(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )  # contain info_loss, diversity_loss, total_loss
            elif self.config.loss_v == "v2":
                reconstruction_losses = compute_compression_loss2(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )

            total_loss = (
                loss_cap
                + self.config.weight["info"] * reconstruction_losses["info_loss"]
                + self.config.weight["info_dino"]
                * reconstruction_losses["info_loss_dino"]
                + self.config.weight["div"] * reconstruction_losses["diversity_loss"]
            )

            return {
                "output": outputs[0],
                "loss": total_loss,
                "loss_cap": loss_cap,
                "loss_info": reconstruction_losses["info_loss"],
                "loss_info_dino": reconstruction_losses["info_loss_dino"],
                "loss_div": reconstruction_losses["diversity_loss"],
            }
        return {"output": outputs[0], "loss": loss_cap}

    def encode_img(self, image):
        forward_outputs = self.encode_img_woproj(image)
        query_output_all = forward_outputs["query_output_img"]
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(image.device)
        return qform_all_proj, atts_qform_all_proj

    def encode_img_woproj(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():

            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            if self.dino_model is not None:
                dino_feats = self.dino_model(image)[0].detach()
            else:
                dino_feats = None

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state

        return {
            "raw_image_feature": image_embeds,
            "query_output_img": query_output_img,
            "raw_image_feature_dino": dino_feats,
        }


class FeatAlignCatCap(FeatAlignCap):
    """
    except feature alignment, also add Cat features with most different dino features.
    - diverse: select most different dino features from compressed feature
    - attn: use attention weight to select most different dino features from compressed feature
    - pool: use average pooling
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

        if config.weight["info_dino"] == 0:
            del self.dino_model
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
            for param in self.dino_model.parameters():
                param.requires_grad = False

        if "num_dino_token" not in self.config:
            self.config.num_dino_token = 8

        if "selected_dino_feat" not in self.config:
            self.config.selected_dino_feat = "diverse"

        if self.config.selected_dino_feat == "diverse":
            # pass
            print("Using diverse to downsample dino features")
        elif self.config.selected_dino_feat == "pool":
            print("Using pooling to downsample dino features")
            self.dino_token_downsample = TokenDownsamplerPool(
                768, self.config.num_dino_token
            )
        elif self.config.selected_dino_feat == "attn":
            print("Using attention to downsample dino features")
            self.dino_token_downsample = TokenDownsampler(
                768, self.config.num_dino_token
            )
        else:
            raise ValueError(
                f"Unknown selected_dino_feat: {self.config.selected_dino_feat}"
            )

    def forward(self, samples):
        ##### Image
        image = samples["image"]
        device = image.device
        # TODO return a dict to contain many terms
        forward_outputs = self.encode_img_woproj(image)  # query_img and query_txt

        raw_image_feature = forward_outputs["raw_image_feature"]
        query_output_img = forward_outputs["query_output_img"]
        query_output_all = forward_outputs["query_output_all"]
        raw_image_feature_dino = forward_outputs["raw_image_feature_dino"]

        # language model loss (caption generation)
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(device)

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
        loss_cap = outputs.loss

        # if self.config.reconstrut_loss == 0 and self.config.diverse_loss == 0:
        #     return {"output": outputs[0], "loss": loss_cap}

        if self.training:
            # reconstruction loss
            if self.config.feat_ds == "linear":
                query_output_img_decoder = self.feature_downsample(query_output_img)
                raw_image_feature_decoder = raw_image_feature
            elif self.config.feat_ds == "pca":
                if not self.config.gpu_pca:
                    raw_image_feature_decoder = pca_feature_reduction(
                        raw_image_feature, 2, 768
                    )
                else:
                    raw_image_feature_decoder = pca_feature_reduction_gpu(
                        raw_image_feature, 2, 768
                    )
                    # breakpoint()
                query_output_img_decoder = query_output_img
            elif self.config.feat_ds == "none":
                raw_image_feature_decoder = None
                query_output_img_decoder = query_output_img

            if self.config.weight["info"] == 0:
                raw_image_feature_decoder = None
            if self.config.weight["info_dino"] == 0:
                raw_image_feature_dino = None

            if "loss_v" not in self.config:
                reconstruction_losses = compute_compression_loss(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )  # contain info_loss, diversity_loss, total_loss
            elif self.config.loss_v == "v2":
                reconstruction_losses = compute_compression_loss2(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )
            elif self.config.loss_v == "seperate":  # only used in FeatAlignCatCap model
                reconstruction_losses = compute_compression_loss_seperate(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )

            total_loss = (
                loss_cap
                + self.config.weight["info"] * reconstruction_losses["info_loss"]
                + self.config.weight["info_dino"]
                * reconstruction_losses["info_loss_dino"]
                + self.config.weight["div"] * reconstruction_losses["diversity_loss"]
            )

            return {
                "output": outputs[0],
                "loss": total_loss,
                "loss_cap": loss_cap,
                "loss_info": reconstruction_losses["info_loss"],
                "loss_info_dino": reconstruction_losses["info_loss_dino"],
                "loss_div": reconstruction_losses["diversity_loss"],
            }
        return {"output": outputs[0], "loss": loss_cap}

    def encode_img(self, image):
        forward_outputs = self.encode_img_woproj(image)
        query_output_all = forward_outputs["query_output_all"]
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(image.device)
        return qform_all_proj, atts_qform_all_proj

    def encode_img_woproj(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():

            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            dino_feats = self.dino_model(image)[0].detach()

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state

            if self.config.selected_dino_feat == "diverse":
                dino_selected = select_most_different_features(
                    query_output_img, dino_feats, self.config.num_dino_token
                )
            else:
                dino_selected = self.dino_token_downsample(dino_feats)

            # breakpoint()
            query_output_all = torch.cat([query_output_img, dino_selected], dim=1)

        return {
            "raw_image_feature": image_embeds,
            "query_output_img": query_output_img,
            "query_output_all": query_output_all,
            "raw_image_feature_dino": dino_feats,
        }


class FeatAlignCatCapV1(FeatAlignCatCap):
    """
    except feature alignment, also add Cat features with most different dino features.
    - diverse: select most different dino features from compressed feature
    - attn: use attention weight to select most different dino features from compressed feature
    - pool: use average pooling
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
        """dino feature is always needed as the dino feature is used to optimize the clip feature"""

        if "fusemode" not in self.config: # clip-dino oclip-clip oclip-clip-dino
            self.config.fusemode = "clip-dino"
        self.use_oclip = "oclip" in self.config.fusemode # oclip denote original clip (optimized only by caption loss)       
        
        # default clip-dino is same as FeatAlignCatCapV1
        if "oclip" in self.config.fusemode:
            self.query_tokens_oclip = nn.Parameter(self.query_tokens.clone())


    def encode_img_woproj(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():

            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            dino_feats = self.dino_model(image)[0].detach()

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state


            if self.config.selected_dino_feat == "diverse":
                dino_selected = select_most_different_features(
                    query_output_img, dino_feats, self.config.num_dino_token
                )
            else:
                dino_selected = self.dino_token_downsample(dino_feats)

            # breakpoint()
            if self.use_oclip:
                query_tokens_oclip = self.query_tokens_oclip.expand(image_embeds.shape[0], -1, -1)
                query_outputs_img_oclip = self.Qformer.bert(
                    query_embeds=query_tokens_oclip,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                query_output_img_oclip = query_outputs_img_oclip.last_hidden_state
            
            if self.config.fusemode == "clip-dino":
                query_output_all = torch.cat([query_output_img, dino_selected], dim=1)
            elif self.config.fusemode == "oclip-clip":
                query_output_all = torch.cat([query_output_img_oclip, query_output_img], dim=1)
            elif self.config.fusemode == "oclip-clip-dino":
                query_output_all = torch.cat([query_output_img_oclip, query_output_img, dino_selected], dim=1)
            else:
                raise ValueError(f"the fusemode {self.config.fusemode} is not supported.")

        return {
            "raw_image_feature": image_embeds,
            "query_output_img": query_output_img,
            "query_output_all": query_output_all,
            "raw_image_feature_dino": dino_feats,
        }




class ClipDinoFuseCap(FeatAlignCatCap):
    """
    extract clip and dino features, then fuse them with another Q-Former
    (the performance is pretty bad, guess it is due to the )
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
        self.config.num_query_token = num_query_token
        if "num_visual_token" not in self.config:
            self.config.num_visual_token = 40
        
        if "clip_dino_fuse_mode" not in self.config: # "attn" "attn-cat"
            self.config.clip_dino_fuse_mode = "attn"

        if self.config.weight.get("div_all", None) is None:
            self.config.weight["div_all"] = 0

        self.config.num_dino_token = self.config.num_visual_token - self.config.num_query_token

        self.Qformer_fuse, self.query_token_fuse = self.init_Qformer(
            self.config.num_visual_token, DINO_FEATURE_DIM
        )
        self.Qformer_fuse.cls = None
        self.Qformer_fuse.bert.embeddings.word_embeddings = None
        self.Qformer_fuse.bert.embeddings.position_embeddings = None
        for layer in self.Qformer_fuse.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        for name, param in self.Qformer_fuse.named_parameters():
            param.requires_grad = False
        self.Qformer_fuse = self.Qformer_fuse.eval()
        self.Qformer_fuse.train = disabled_train
        self.query_token_fuse.requires_grad = True
        logging.info("freeze Qformer")


    def forward(self, samples):
        ##### Image
        image = samples["image"]
        device = image.device
        # TODO return a dict to contain many terms
        forward_outputs = self.encode_img_woproj(image)  # query_img and query_txt

        raw_image_feature = forward_outputs["raw_image_feature"]
        query_output_img = forward_outputs["query_output_img"]
        query_output_all = forward_outputs["query_output_all"]
        raw_image_feature_dino = forward_outputs["raw_image_feature_dino"]

        # language model loss (caption generation)
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(device)

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
        loss_cap = outputs.loss

        # if self.config.reconstrut_loss == 0 and self.config.diverse_loss == 0:
        #     return {"output": outputs[0], "loss": loss_cap}

        if self.training:
            # reconstruction loss
            if self.config.feat_ds == "linear":
                query_output_img_decoder = self.feature_downsample(query_output_img)
                raw_image_feature_decoder = raw_image_feature
            elif self.config.feat_ds == "pca":
                if not self.config.gpu_pca:
                    raw_image_feature_decoder = pca_feature_reduction(
                        raw_image_feature, 2, 768
                    )
                else:
                    raw_image_feature_decoder = pca_feature_reduction_gpu(
                        raw_image_feature, 2, 768
                    )
                    # breakpoint()
                query_output_img_decoder = query_output_img
            elif self.config.feat_ds == "none":
                raw_image_feature_decoder = None
                query_output_img_decoder = query_output_img

            if self.config.weight["info"] == 0:
                raw_image_feature_decoder = None
            if self.config.weight["info_dino"] == 0:
                raw_image_feature_dino = None

            if "loss_v" not in self.config:
                reconstruction_losses = compute_compression_loss(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )  # contain info_loss, diversity_loss, total_loss
            elif self.config.loss_v == "v2":
                reconstruction_losses = compute_compression_loss2(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )
            elif self.config.loss_v == "seperate":  # only used in FeatAlignCatCap model
                reconstruction_losses = compute_compression_loss_seperate(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )

            if self.config.weight.get("div_all") != 0:
                loss_div_all = compute_div_loss(query_output_all)
            else:
                loss_div_all = 0

            total_loss = (
                loss_cap
                + self.config.weight["info"] * reconstruction_losses["info_loss"]
                + self.config.weight["info_dino"]
                * reconstruction_losses["info_loss_dino"]
                + self.config.weight["div"] * reconstruction_losses["diversity_loss"]
                + self.config.weight["div_all"] * loss_div_all
            )
            breakpoint()
            
            # if self.config.weight["div_all"] > 0:
            return {
                "output": outputs[0],
                "loss": total_loss,
                "loss_cap": loss_cap,
                "loss_info": reconstruction_losses["info_loss"],
                "loss_info_dino": reconstruction_losses["info_loss_dino"],
                "loss_div": reconstruction_losses["diversity_loss"],
                "loss_div_all": loss_div_all
            }
            # else:
            #     return {
            #         "output": outputs[0],
            #         "loss": total_loss,
            #         "loss_cap": loss_cap,
            #         "loss_info": reconstruction_losses["info_loss"],
            #         "loss_info_dino": reconstruction_losses["info_loss_dino"],
            #         "loss_div": reconstruction_losses["diversity_loss"],
            #     }
        return {"output": outputs[0], "loss": loss_cap}

    def encode_img(self, image):
        forward_outputs = self.encode_img_woproj(image)
        query_output_all = forward_outputs["query_output_all"]
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(image.device)
        return qform_all_proj, atts_qform_all_proj

    def encode_img_woproj(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():

            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            dino_feats = self.dino_model(image)[0].detach()

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state

            """dino feature should be viewed as text like EVCap, see v1 version"""
            clip_dino_visual_feat = torch.cat([query_output_img, dino_feats], dim=1)
            clip_dino_visual_atts = torch.ones(clip_dino_visual_feat.size()[:-1], dtype=torch.long).to(device)
            
            query_tokens_fuse = self.query_token_fuse.expand(image_embeds.shape[0], -1, -1)
            query_outputs_fuse = self.Qformer_fuse.bert(
                query_embeds=query_tokens_fuse,
                encoder_hidden_states=clip_dino_visual_feat,
                encoder_attention_mask=clip_dino_visual_atts,
                return_dict=True,
            )
            query_outputs_fuse_feat = query_outputs_fuse.last_hidden_state  

            # here query_outputs_fuse_feat is the fused feature from clip and dino feature          

            # breakpoint()
            # print("query_outputs_fuse_feat", query_outputs_fuse_feat.shape)
            
            if self.config.clip_dino_fuse_mode == "attn":
                query_output_all = query_outputs_fuse_feat
            elif self.config.clip_dino_fuse_mode == "attnclip":
                query_output_all = torch.cat([query_output_img, query_outputs_fuse_feat], dim=1)
            elif self.config.clip_dino_fuse_mode == "attndino":
                if self.config.selected_dino_feat == "diverse":
                    dino_selected = select_most_different_features(
                        query_output_img, dino_feats, self.config.num_dino_token
                    )
                else:
                    dino_selected = self.dino_token_downsample(dino_feats)
                query_output_all = torch.cat(
                    [query_outputs_fuse_feat, dino_selected], dim=1
                )
            elif self.config.clip_dino_fuse_mode == "attncat":
                """fused clip + original dino"""
                if self.config.selected_dino_feat == "diverse":
                    dino_selected = select_most_different_features(
                        query_output_img, dino_feats, self.config.num_dino_token
                    )
                else:
                    dino_selected = self.dino_token_downsample(dino_feats)
                query_output_all = torch.cat([query_outputs_fuse_feat[:, :self.config.num_query_token], dino_selected], dim=1)
            elif self.config.fuse_mode == "attncatclip":
                """clip + fused dino"""
                if self.config.selected_dino_feat == "diverse":
                    dino_selected = select_most_different_features(
                        query_output_img, dino_feats, self.config.num_dino_token
                    )
                else:
                    dino_selected = self.dino_token_downsample(dino_feats)
                query_output_all = torch.cat([query_output_img, query_outputs_fuse_feat[:, self.config.num_query_token:]], dim=1)
            else:
                print("not supported fuse mode")
                exit(1)
        return {
            "raw_image_feature": image_embeds,
            "query_output_img": query_output_img,
            "query_output_all": query_output_all,
            "raw_image_feature_dino": dino_feats,
        }

class ClipDinoFuseCap_v1(FeatAlignCatCap):
    """
    extract clip and dino features, then fuse them with another Q-Former
    use a customized Q-Former like EVCap, use dino feature as guidance
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
        self.config.num_query_token = num_query_token
        """ num_visual_token: all visual token sends to LLM (use number of dino token directly)
            num_dino_token: the number of query token for DINO"""
        
        if "num_dino_token" not in self.config:
            self.config.num_dino_token = 8
        
        if "clip_dino_fuse_mode" not in self.config: # "attn" "attn-cat"
            self.config.clip_dino_fuse_mode = "attn"

        if self.config.weight.get("div_all", None) is None:
            self.config.weight["div_all"] = 0


        self.Qformer_fuse, self.query_token_fuse = self.init_Qformer(
            self.config.num_dino_token, DINO_FEATURE_DIM
        )
        self.Qformer_fuse.cls = None
        self.Qformer_fuse.bert.embeddings.word_embeddings = None
        self.Qformer_fuse.bert.embeddings.position_embeddings = None
        for layer in self.Qformer_fuse.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        for name, param in self.Qformer_fuse.named_parameters():
            param.requires_grad = False
        self.Qformer_fuse = self.Qformer_fuse.eval()
        self.Qformer_fuse.train = disabled_train
        self.query_token_fuse.requires_grad = True
        logging.info("freeze Qformer")


    def forward(self, samples):
        ##### Image
        image = samples["image"]
        device = image.device
        # TODO return a dict to contain many terms
        forward_outputs = self.encode_img_woproj(image)  # query_img and query_txt

        raw_image_feature = forward_outputs["raw_image_feature"]
        query_output_img = forward_outputs["query_output_img"]
        query_output_all = forward_outputs["query_output_all"]
        raw_image_feature_dino = forward_outputs["raw_image_feature_dino"]

        # language model loss (caption generation)
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(device)

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
        loss_cap = outputs.loss

        # if self.config.reconstrut_loss == 0 and self.config.diverse_loss == 0:
        #     return {"output": outputs[0], "loss": loss_cap}

        if self.training:
            # reconstruction loss
            if self.config.feat_ds == "linear":
                query_output_img_decoder = self.feature_downsample(query_output_img)
                raw_image_feature_decoder = raw_image_feature
            elif self.config.feat_ds == "pca":
                if not self.config.gpu_pca:
                    raw_image_feature_decoder = pca_feature_reduction(
                        raw_image_feature, 2, 768
                    )
                else:
                    raw_image_feature_decoder = pca_feature_reduction_gpu(
                        raw_image_feature, 2, 768
                    )
                    # breakpoint()
                query_output_img_decoder = query_output_img
            elif self.config.feat_ds == "none":
                raw_image_feature_decoder = None
                query_output_img_decoder = query_output_img

            if self.config.weight["info"] == 0:
                raw_image_feature_decoder = None
            if self.config.weight["info_dino"] == 0:
                raw_image_feature_dino = None

            if "loss_v" not in self.config:
                reconstruction_losses = compute_compression_loss(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )  # contain info_loss, diversity_loss, total_loss
            elif self.config.loss_v == "v2":
                reconstruction_losses = compute_compression_loss2(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )
            elif self.config.loss_v == "seperate":  # only used in FeatAlignCatCap model
                reconstruction_losses = compute_compression_loss_seperate(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )

            if self.config.weight.get("div_all") != 0:
                loss_div_all = compute_div_loss(query_output_all)
            else:
                loss_div_all = 0

            total_loss = (
                loss_cap
                + self.config.weight["info"] * reconstruction_losses["info_loss"]
                + self.config.weight["info_dino"]
                * reconstruction_losses["info_loss_dino"]
                + self.config.weight["div"] * reconstruction_losses["diversity_loss"]
                + self.config.weight["div_all"] * loss_div_all
            )
            

            return {
                "output": outputs[0],
                "loss": total_loss,
                "loss_cap": loss_cap,
                "loss_info": reconstruction_losses["info_loss"],
                "loss_info_dino": reconstruction_losses["info_loss_dino"],
                "loss_div": reconstruction_losses["diversity_loss"],
                "loss_div_all": loss_div_all
            }
        return {"output": outputs[0], "loss": loss_cap}

    def encode_img(self, image):
        forward_outputs = self.encode_img_woproj(image)
        query_output_all = forward_outputs["query_output_all"]
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(image.device)
        return qform_all_proj, atts_qform_all_proj

    def encode_img_woproj(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():

            clip_image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            clip_image_atts = torch.ones(clip_image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            dino_feats = self.dino_model(image)[0].detach()
            dino_feats_mask = torch.ones(dino_feats.size()[:-1], dtype=torch.long).to(image.device)


            query_tokens = self.query_tokens.expand(clip_image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=clip_image_embeds,
                encoder_attention_mask=clip_image_atts,
                return_dict=True,
            )
            query_output_clip_img = query_outputs_img.last_hidden_state
            
            query_tokens_fuse = self.query_token_fuse.expand(clip_image_embeds.shape[0], -1, -1)
            query_clip_cat_feature = torch.cat([query_tokens_fuse, query_output_clip_img], dim=1)
            
            query_outputs_fuse = self.Qformer_fuse.bert(
                query_embeds=query_clip_cat_feature,
                encoder_hidden_states=dino_feats,
                encoder_attention_mask=dino_feats_mask,
                return_dict=True,
            )
            query_clip_outputs_fuse_feat = query_outputs_fuse.last_hidden_state  

            fused_query_outputs = query_clip_outputs_fuse_feat[:, :self.config.num_dino_token]
            fused_clip_feat = query_clip_outputs_fuse_feat[:, self.config.num_dino_token:]
            
            if self.config.clip_dino_fuse_mode == "attn":
                query_output_all = query_clip_outputs_fuse_feat # fused query + fused clip
            elif self.config.clip_dino_fuse_mode == "attnclip":
                query_output_all = fused_clip_feat # fused clip only
            elif self.config.clip_dino_fuse_mode == "catclip":
                query_output_all = torch.cat([query_output_clip_img, fused_query_outputs], dim=1) # original clip + fused query (dino)
            elif self.config.clip_dino_fuse_mode == "catdino":
                if self.config.selected_dino_feat == "diverse":
                    dino_selected = select_most_different_features(query_output_clip_img, dino_feats, self.config.num_dino_token
                    )
                else:
                    dino_selected = self.dino_token_downsample(dino_feats)
                query_output_all = torch.cat(
                    [dino_selected, fused_query_outputs], dim=1
                ) # original dino + fused 
            elif self.config.clip_dino_fuse_mode == "catclipdino":
                if self.config.selected_dino_feat == "diverse":
                    dino_selected = select_most_different_features(query_output_clip_img, dino_feats, self.config.num_dino_token
                    )
                else:
                    dino_selected = self.dino_token_downsample(dino_feats)
                query_output_all = torch.cat([query_output_clip_img, dino_selected, fused_query_outputs], dim=1)
            elif self.config.clip_dino_fuse_mode == "catall":
                query_output_all = torch.cat([query_output_clip_img, query_clip_outputs_fuse_feat], dim=1)
            else:
                print("not supported fuse mode")
                exit(1)
        return {
            "raw_image_feature": clip_image_embeds,
            "query_output_img": query_output_clip_img,
            "query_output_all": query_output_all,
            "raw_image_feature_dino": dino_feats,
        }





class DinoCap(Blip2Base):
    """
    Base model with RAG (text fusion or raw text send to LLM)
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

        # # Image
        # print("Loading VIT")
        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     for name, param in self.ln_vision.named_parameters():
        #         param.requires_grad = False
        #     self.ln_vision = self.ln_vision.eval()
        #     self.ln_vision.train = disabled_train
        #     logging.info("freeze vision encoder")
        # print("Loading VIT Done")

        # print("Loading Q-Former")
        # # breakpoint()
        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features
        # )

        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
        # self.load_from_pretrained(url_or_filename=q_former_model)

        # if freeze_qformer:
        #     for name, param in self.Qformer.named_parameters():
        #         param.requires_grad = False
        #     self.Qformer = self.Qformer.eval()
        #     self.Qformer.train = disabled_train
        #     self.query_tokens.requires_grad = True
        #     logging.info("freeze Qformer")
        # print("Loading Q-Former Done")

        # Caption generation
        print("Loading LLAMA")
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            llama_model, use_fast=False
        )
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            # self.llama_model = LlamaForCausalLM.from_pretrained(
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                # device_map="auto"
            )

        # frozen llama model
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print("Loading LLAMA Done")

        ###
        self.llama_proj = nn.Linear(768, self.llama_model.config.hidden_size)

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

        config["num_query_token"] = num_query_token
        config["num_query_token_txt"] = num_query_token_txt

        self.pattern_dictionary = {"None": [""]}
        config.actual_bs = len(self.pattern_dictionary[config.visual_pattern])

        # if config.weight["info_dino"] != 0:
        #     # self.dino_processor = AutoImageProcessor.from_pretrained(
        #     #     "facebook/dinov2-base"
        #     # )
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
        for param in self.dino_model.parameters():
            param.requires_grad = False
        self.dino_tokendownsample = TokenDownsampler(768, num_query_token)

        # if config.weight["info"] != 0 and config.feat_ds == "linear":
        #     self.feature_downsample = nn.Linear(1408, 768)
        # self.feature_downsample = nn.Linear(768, 1408)
        # if "gpu_pca" not in config:
        #     config.gpu_pca = False
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
                # breakpoint()
                p_before, p_after = prompt.split("<ImageHere>", 1)
                # p_after = p_after + ' [SEP] Related objects: ' + self.raw_test[i] # add raw text
                # breakpoint()
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

    def forward(self, samples):
        ##### Image
        image = samples["image"]
        device = image.device
        # TODO return a dict to contain many terms
        forward_outputs = self.encode_img_woproj(image)  # query_img and query_txt

        raw_image_feature = forward_outputs["raw_image_feature"]
        query_output_img = forward_outputs["query_output_img"]
        query_output_all = query_output_img
        raw_image_feature_dino = forward_outputs["raw_image_feature_dino"]

        # language model loss (caption generation)
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(device)

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
        loss_cap = outputs.loss

        # if self.config.reconstrut_loss == 0 and self.config.diverse_loss == 0:
        #     return {"output": outputs[0], "loss": loss_cap}

        if self.training:
            # reconstruction loss
            # if self.config.feat_ds == "linear":
            #     # breakpoint()
            #     query_output_img_decoder = query_output_img
            #     raw_image_feature_decoder = self.feature_downsample(raw_image_feature)
            # elif self.config.feat_ds == "pca":
            #     if self.config.gpu_pca:
            #         raw_image_feature_decoder = pca_feature_reduction_gpu(
            #             raw_image_feature, 2, 768
            #         )
            #     else:
            #         raw_image_feature_decoder = pca_feature_reduction(
            #             raw_image_feature, 2, 768
            #         )
            #     query_output_img_decoder = query_output_img
            # elif self.config.feat_ds == "none":
            #     raw_image_feature_decoder = None
            #     query_output_img_decoder = query_output_img

            # if self.config.weight["info"] == 0:
            #     raw_image_feature_decoder = None
            # if self.config.weight["info_dino"] == 0:
            #     raw_image_feature_dino = None
            raw_image_feature_decoder = raw_image_feature
            query_output_img_decoder = query_output_img

            if "loss_v" not in self.config:
                reconstruction_losses = compute_compression_loss(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )  # contain info_loss, diversity_loss, total_loss
            elif self.config.loss_v == "v2":
                reconstruction_losses = compute_compression_loss2(
                    raw_image_feature_decoder,
                    query_output_img_decoder,
                    raw_image_feature_dino,
                )

            total_loss = (
                loss_cap
                + self.config.weight["info"] * reconstruction_losses["info_loss"]
                + self.config.weight["info_dino"]
                * reconstruction_losses["info_loss_dino"]
                + self.config.weight["div"] * reconstruction_losses["diversity_loss"]
            )

            return {
                "output": outputs[0],
                "loss": total_loss,
                "loss_cap": loss_cap,
                "loss_info": reconstruction_losses["info_loss"],
                "loss_info_dino": reconstruction_losses["info_loss_dino"],
                "loss_div": reconstruction_losses["diversity_loss"],
            }
        return {"output": outputs[0], "loss": loss_cap}

    def encode_img(self, image):
        forward_outputs = self.encode_img_woproj(image)
        query_output_all = forward_outputs["query_output_img"]
        qform_all_proj = self.llama_proj(query_output_all)
        atts_qform_all_proj = torch.ones(
            qform_all_proj.size()[:-1], dtype=torch.long
        ).to(image.device)
        return qform_all_proj, atts_qform_all_proj

    def encode_img_woproj(self, image):
        device = image.device
        # if self.low_resource:
        #     self.vit_to_cpu()
        #     image = image.to("cpu")

        with self.maybe_autocast():

            # image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            #     device
            # )

            # if self.dino_model is not None:
            dino_feats = self.dino_model(image)[0].detach()
            # else:
            #     dino_feats = None

            # breakpoint()
            # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            # query_outputs_img = self.Qformer.bert(
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=dino_feats,
            #     encoder_attention_mask=image_atts,
            #     return_dict=True,
            # )

            # query_outputs_img = self.Qformer.bert(
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=image_embeds,
            #     encoder_attention_mask=image_atts,
            #     return_dict=True,
            # )
            dino_feat_ds = self.dino_tokendownsample(dino_feats)
            # query_output_img = dino_feat_ds

        return {
            "raw_image_feature": dino_feats,
            "query_output_img": dino_feat_ds,
            "raw_image_feature_dino": None,
        }
