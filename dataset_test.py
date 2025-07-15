import pickle

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Resize

from cosmos_predict2.configs.base.config_video2world import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B,
)
from cosmos_predict2.pipelines.video2world import SimpleVideo2WorldPipeline
from imaginaire.utils.io import save_image_or_video

# class CosmosDatasetWrapper(torch.utils.data.Dataset):
#     def __init__(self, dataset: LeRobotDataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         image = item["image"].permute(1, 0, 2, 3)
#         image = (image * 255).to(torch.uint8)
#         t5_text_embeddings = torch.zeros((512, 1024), dtype=torch.float32)
#         t5_text_mask = torch.zeros(512, dtype=torch.int64)
#         return {
#             "image": image,
#             "t5_text_embeddings": t5_text_embeddings,
#             "t5_text_mask": t5_text_mask,
#             "fps": torch.tensor(5.0, dtype=torch.float32),  # Assuming a fixed FPS of 5
#             "padding_mask": torch.zeros((1, 224, 224), dtype=torch.float32),
#         }


dataset = LeRobotDataset(
    repo_id="ZibinDong/so100_grab_screwdriver",
    delta_timestamps={
        "observation.images.base": [i / 15 for i in range(13)],
    },
)

# dataset = CosmosDatasetWrapper(dataset)
