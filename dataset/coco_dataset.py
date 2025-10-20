import os
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# from torch.utils.data import Dataset
import numpy as np
import random
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class COCODataset(Dataset):

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_file = f'COCO_train2014_{int(ann["image_id"]):012d}.jpg'
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = ann["caption"]
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

    def __init__(self, data_root):
        ann_path = os.path.join(data_root, "annotations/captions_train2014.json")
        self.vis_root = os.path.join(data_root, "train2014")
        self.annotation = []
        self.annotation.extend(json.load(open(ann_path, "r"))["annotations"])
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )


class COCOWithPredictionsDataset(Dataset):

    def __init__(self, data_root, predicted_captions_file=None):
        # Path to the annotations and image directory
        ann_path = os.path.join(data_root, "annotations/captions_train2014.json")
        self.vis_root = os.path.join(data_root, "train2014")

        # Load the annotations for ground truth captions
        self.annotation = []
        self.annotation.extend(json.load(open(ann_path, "r"))["annotations"])

        # Create a dictionary mapping image_id to index
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        # Load predicted captions if a file is provided
        self.predicted_captions = {}
        if predicted_captions_file:
            with open(predicted_captions_file, "r") as f:
                predicted_data = json.load(f)
                self.predicted_captions = {
                    entry["image_id"]: entry["pred"] for entry in predicted_data
                }

        # Image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self) -> int:
        # The dataset will return 2x the number of samples (for both captions)
        return len(self.annotation) * 2

    def __getitem__(self, index):
        # Determine whether we are returning the ground truth or predicted caption
        ann = self.annotation[
            index // 2
        ]  # Divide by 2 to access the correct annotation
        img_id = ann["image_id"]
        img_file = f"COCO_train2014_{int(img_id):012d}.jpg"
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Ground truth caption
        caption = ann["caption"]

        # # Predicted caption (if exists, otherwise None)
        # predicted_caption = self.predicted_captions.get(ann["image_id"], None)

        # If the index is even, return the ground truth caption
        if index % 2 == 0:
            return {
                "image": image,
                "text_input": caption,
                "image_id": self.img_ids[img_id],
                "initial_caption": "",  # Use empty string instead of None
            }
        else:
            # Predicted caption or empty string if not found
            predicted_caption = self.predicted_captions.get(img_id, "")
            return {
                "image": image,
                "text_input": caption,  # Use predicted or ground truth
                "image_id": self.img_ids[img_id],
                "initial_caption": predicted_caption,  # Return predicted caption or empty string
            }


class COCOOnlyPredictionsDataset(Dataset):

    def __init__(self, data_root, predicted_captions_file=None):
        # Path to the annotations and image directory
        ann_path = os.path.join(data_root, "annotations/captions_train2014.json")
        self.vis_root = os.path.join(data_root, "train2014")

        # Load the annotations for ground truth captions
        self.annotation = []
        self.annotation.extend(json.load(open(ann_path, "r"))["annotations"])

        # Create a dictionary mapping image_id to index
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        # Load predicted captions if a file is provided
        self.predicted_captions = {}
        if predicted_captions_file:
            with open(predicted_captions_file, "r") as f:
                predicted_data = json.load(f)
                self.predicted_captions = {
                    entry["image_id"]: entry["pred"] for entry in predicted_data
                }

        # Image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self) -> int:
        # The dataset will return 2x the number of samples (for both captions)
        return len(self.annotation)

    def __getitem__(self, index):
        # Determine whether we are returning the ground truth or predicted caption
        ann = self.annotation[index]  # Divide by 2 to access the correct annotation
        img_id = ann["image_id"]
        img_file = f"COCO_train2014_{int(img_id):012d}.jpg"
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Ground truth caption
        caption = ann["caption"]

        # # Predicted caption (if exists, otherwise None)
        # predicted_caption = self.predicted_captions.get(ann["image_id"], None)

        # If the index is even, return the ground truth caption

        # Predicted caption or empty string if not found
        predicted_caption = self.predicted_captions.get(img_id, "")
        return {
            "image": image,
            "text_input": caption,  # Use predicted or ground truth
            "image_id": self.img_ids[img_id],
            "initial_caption": predicted_caption,  # Return predicted caption or empty string
        }


# Define the combined dataset (unchanged)
class CombinedCOCODataset(ConcatDataset):
    def __init__(self, data_root, predicted_captions_file):
        # Initialize the original datasets
        coco_dataset = COCODataset(data_root)  # Assume this is defined elsewhere
        coco_with_preds = COCOWithPredictionsDataset(
            data_root, predicted_captions_file
        )  # Assume defined
        # Combine them using ConcatDataset
        super().__init__([coco_dataset, coco_with_preds])


# Modified AlternateBatchSampler for distributed training
class AlternateBatchSampler:
    def __init__(self, distributed_sampler, dataset1_len, dataset2_len, batch_size):
        """
        Initialize the batch sampler with a DistributedSampler.

        Args:
            distributed_sampler (DistributedSampler): Sampler that provides distributed indices.
            dataset1_len (int): Length of the first dataset.
            dataset2_len (int): Length of the second dataset.
            batch_size (int): Size of each batch.
        """
        self.distributed_sampler = distributed_sampler
        self.dataset1_len = dataset1_len
        self.dataset2_len = dataset2_len
        self.batch_size = batch_size

    def set_epoch(self, epoch):
        """
        Set the epoch for the DistributedSampler to ensure different shuffling each epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.distributed_sampler.set_epoch(epoch)

    def __iter__(self):
        """
        Generate batches alternately from each dataset using distributed indices.

        Returns:
            Iterator over shuffled batches of indices.
        """
        # Get the indices assigned to this process by DistributedSampler
        indices = list(self.distributed_sampler)

        # Separate indices into those from dataset1 and dataset2
        indices1 = [idx for idx in indices if idx < self.dataset1_len]
        indices2 = [idx for idx in indices if idx >= self.dataset1_len]

        # Create batches from each dataset's indices
        batches1 = [
            indices1[i : i + self.batch_size]
            for i in range(0, len(indices1), self.batch_size)
        ]
        batches2 = [
            indices2[i : i + self.batch_size]
            for i in range(0, len(indices2), self.batch_size)
        ]

        # Combine all batches and shuffle their order
        all_batches = batches1 + batches2
        random.shuffle(all_batches)

        return iter(all_batches)

    def __len__(self):
        """
        Return the approximate number of batches.

        Returns:
            int: Number of batches.
        """
        total_samples = len(self.distributed_sampler)
        return (
            total_samples + self.batch_size - 1
        ) // self.batch_size  # Ceiling division
