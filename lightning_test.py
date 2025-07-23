import math
import pickle
from typing import Any, Tuple

import lightning as L
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms.v2 import Resize

from cosmos_predict2.conditioner import DataType, T2VCondition
from cosmos_predict2.configs.base.config_video2world import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_DZB,
)
from cosmos_predict2.models.text2image_dit import Block, FinalLayer
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.utils import log


class WarmupLinearScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        f_start: float,
        f_min: float,
        f_max: float,
        warm_up_steps: int,
        cycle_lengths: int,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ):
        def lr_lambda(n: int) -> float:
            if n < warm_up_steps:
                if warm_up_steps > 0:
                    f = (f_max - f_start) / warm_up_steps * n + f_start
                else:
                    f = f_max
            elif n < cycle_lengths:
                decay_steps = cycle_lengths - warm_up_steps
                if decay_steps > 0:
                    f = f_min + (f_max - f_min) * (cycle_lengths - n) / decay_steps
                else:
                    f = f_min
            else:
                f = f_min
            return f

        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


class Predict2Video2WorldModelLightning(L.LightningModule):
    def __init__(
        self,
        dit_path: str = "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
        train_architecture: str = "lora",
    ):
        super().__init__()
        self.pipe_config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_DZB
        self.precision = torch.bfloat16
        self.tensor_kwargs = {"device": self.device, "dtype": self.precision}
        self.high_sigma_ratio = 0.0
        self.dit_path = dit_path
        self.train_architecture = train_architecture

        self.loss_reduce = "mean"
        self.loss_scale = 10
        self.video_noise_multiplier = math.sqrt(self.pipe_config.state_t)

        self.pipe = None

        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

    def configure_model(self):
        self.pipe = Video2WorldPipeline.from_config(
            config=self.pipe_config,
            dit_path=self.dit_path,
            text_encoder_path="",  # disable
            device=self.device,
        )
        if self.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=128,
                lora_alpha=256,
                lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
                init_lora_weights=True,
            )
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=128,
                    lora_alpha=256,
                    lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
                    init_lora_weights=True,
                )
        else:
            self.pipe.denoising_model().requires_grad_(True)
            self.pipe.tokenizer.requires_grad_(False)

    def add_lora_to_model(
        self,
        model,
        lora_rank=16,
        lora_alpha=16,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,
    ):
        from peft import LoraConfig, inject_adapter_in_model

        self.lora_alpha = lora_alpha

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
            use_rslora=True,
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    def draw_training_sigma_and_epsilon(
        self, x0_size: torch.Size, condition: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device="cuda")
        sigma_B = self.pipe.scheduler.sample_sigma(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(
            sigma_B, "b -> b 1"
        )  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO

        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        if is_video_batch and self.high_sigma_ratio > 0:
            # Implement the high sigma strategy LOGUNIFORM200_100000
            LOG_200 = math.log(200)
            LOG_100000 = math.log(100000)
            mask = (
                torch.rand(sigma_B_1.shape, device=sigma_B_1.device)
                < self.high_sigma_ratio
            )
            log_new_sigma = (
                torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1)
                * (LOG_100000 - LOG_200)
                + LOG_200
            )
            sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
        return sigma_B_1, epsilon

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: T2VCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss givee epsilon and sigma

        This method is responsible for computing loss give epsilon and sigma. It involves:
        1. Adding noise to the input data.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0: image/video latent
            condition: text condition
            epsilon: noise
            sigma: noise level

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = x0_B_C_T_H_W, sigma_B_T
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(
            std_B_T, "b t -> b 1 t 1 1"
        )
        # make prediction
        model_pred = self.pipe.denoise(xt_B_C_T_H_W, sigma_B_T, condition)
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
        # extra loss mask for each sample, for example, human faces, hands
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(
            weights_per_sigma_B_T, "b t -> b 1 t 1 1"
        )
        kendall_loss = edm_loss_B_C_T_H_W
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            "edm_loss": edm_loss_B_C_T_H_W.mean(),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
        }
        output_batch["loss"] = kendall_loss.mean()  # check if this is what we want

        return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (sigma**2 + self.pipe.sigma_data**2) / (
            sigma * self.pipe.sigma_data
        ) ** 2

    def training_step(self, batch, batch_idx):
        batch["num_conditional_frames"] = 1
        # Implement the training step logic here
        _, x0_B_C_T_H_W, condition = self.pipe.get_data_and_condition(batch)
        # Sample pertubation noise levels and N(0, 1) noises
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(
            x0_B_C_T_H_W.size(), condition
        )
        output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )

        if self.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        self.log("loss", kendall_loss.item(), prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"])

        return kendall_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.pipe.parameters(),
            lr=2 ** (-14.5),
            weight_decay=0.01,
            betas=(0.9, 0.99),
        )
        scheduler = WarmupLinearScheduler(
            optimizer=optimizer,
            f_start=1e-6,
            f_min=0.1,
            f_max=0.2,
            warm_up_steps=1000,
            cycle_lengths=100_000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class CosmosDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset
        with open("tasks_with_embeddings.pkl", "rb") as f:
            self.t5_embeddings = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].permute(1, 0, 2, 3)
        # image = item["observation.images.base"].permute(1, 0, 2, 3)
        image = (image * 255).to(torch.uint8)
        t5_text_embeddings = torch.zeros((512, 1024), dtype=torch.float32)
        t5_text_mask = torch.zeros(512, dtype=torch.int64)

        task = item["task"]
        if len(task) > 0:
            task = task[0].upper() + task[1:].lower()
            if not task.endswith("."):
                task += "."
        try:
            emb = self.t5_embeddings[task]
            t5_text_embeddings[: emb.shape[0]] = torch.tensor(emb, dtype=torch.float32)
            t5_text_mask[: emb.shape[0]] = 1
        except KeyError:
            log.warning(
                f"Task '{task}' not found in T5 embeddings. Using zero embeddings."
            )

        return {
            "video": image,
            "t5_text_embeddings": t5_text_embeddings[:32],
            "t5_text_mask": t5_text_mask[:32],
            "fps": torch.tensor(5.0, dtype=torch.float32),  # Assuming a fixed FPS of 5
            "padding_mask": torch.zeros((1, 224, 224), dtype=torch.float32),
        }


model = Predict2Video2WorldModelLightning(
    "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt",
    train_architecture="none",
)

_dataset = LeRobotDataset(
    episodes=[i for i in range(10)],
    repo_id="ZibinDong/bridgedatav2_train",
    root="/openbayes/input/input0",
    delta_timestamps={
        "image": [i / 5 for i in range(21)],
    },
    image_transforms=Resize((480, 480)),
)

dataset = CosmosDatasetWrapper(_dataset)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=11, shuffle=True, persistent_workers=True
)

callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="model-{epoch}-{step}",
    save_top_k=-1,
    every_n_train_steps=10_000,
)

policy = {
    Block,
}
strategy = FSDPStrategy(auto_wrap_policy=policy, sharding_strategy="SHARD_GRAD_OP")

trainer = L.Trainer(
    accelerator="cuda",
    devices=4,
    max_steps=100_000,
    # max_epochs=2,
    enable_progress_bar=True,
    strategy=strategy,
    precision="bf16-mixed",
    callbacks=[callback],
    log_every_n_steps=5,
    # overfit_batches=5,
    accumulate_grad_batches=1,
    # gradient_clip_val=1.0,
)

trainer.fit(model, dataloader)


# import math
# import pickle
# from typing import Any, Tuple

# import lightning as L
# import matplotlib.pyplot as plt
# import torch
# from einops import rearrange
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.strategies import FSDPStrategy
# from torch.optim.lr_scheduler import LambdaLR
# from torchvision.transforms.v2 import Resize

# from cosmos_predict2.conditioner import DataType, T2VCondition
# from cosmos_predict2.configs.base.config_video2world import (
#     PREDICT2_VIDEO2WORLD_PIPELINE_2B_DZB,
# )
# from cosmos_predict2.models.text2image_dit import Block, FinalLayer
# from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
# from imaginaire.utils import log
# from lightning.pytorch import loggers as pl_loggers


# class WarmupLinearScheduler(LambdaLR):
#     def __init__(
#         self,
#         optimizer: torch.optim.Optimizer,
#         f_start: float,
#         f_min: float,
#         f_max: float,
#         warm_up_steps: int,
#         cycle_lengths: int,
#         last_epoch: int = -1,
#         verbose: str = "deprecated",
#     ):
#         def lr_lambda(n: int) -> float:
#             if n < warm_up_steps:
#                 if warm_up_steps > 0:
#                     f = (f_max - f_start) / warm_up_steps * n + f_start
#                 else:
#                     f = f_max
#             elif n < cycle_lengths:
#                 decay_steps = cycle_lengths - warm_up_steps
#                 if decay_steps > 0:
#                     f = f_min + (f_max - f_min) * (cycle_lengths - n) / decay_steps
#                 else:
#                     f = f_min
#             else:
#                 f = f_min
#             return f

#         super().__init__(optimizer, lr_lambda, last_epoch, verbose)


# class Predict2Video2WorldModelLightning(L.LightningModule):
#     def __init__(
#         self,
#         dit_path: str = "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
#         train_architecture: str = "lora",
#     ):
#         super().__init__()
#         self.pipe_config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_DZB
#         self.precision = torch.bfloat16
#         self.tensor_kwargs = {"device": self.device, "dtype": self.precision}
#         self.high_sigma_ratio = 0.0
#         self.dit_path = dit_path
#         self.train_architecture = train_architecture

#         self.loss_reduce = "mean"
#         self.loss_scale = 10
#         self.video_noise_multiplier = math.sqrt(self.pipe_config.state_t)

#         self.pipe = None

#         total_params = sum(p.numel() for p in self.parameters())
#         frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
#         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         # Print the number in billions, or in the format of 1,000,000,000
#         log.info(
#             f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
#         )

#     def configure_model(self):
#         self.pipe = Video2WorldPipeline.from_config(
#             config=self.pipe_config,
#             dit_path=self.dit_path,
#             text_encoder_path="",  # disable
#             device=self.device,
#         )
#         if self.train_architecture == "lora":
#             self.add_lora_to_model(
#                 self.pipe.dit,
#                 lora_rank=128,
#                 lora_alpha=256,
#                 lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
#                 init_lora_weights=True,
#             )
#             if self.pipe.dit_ema:
#                 self.add_lora_to_model(
#                     self.pipe.dit_ema,
#                     lora_rank=128,
#                     lora_alpha=256,
#                     lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
#                     init_lora_weights=True,
#                 )
#         else:
#             self.pipe.denoising_model().requires_grad_(True)
#             self.pipe.tokenizer.requires_grad_(False)

#     def add_lora_to_model(
#         self,
#         model,
#         lora_rank=16,
#         lora_alpha=16,
#         lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
#         init_lora_weights=True,
#     ):
#         from peft import LoraConfig, inject_adapter_in_model

#         self.lora_alpha = lora_alpha

#         lora_config = LoraConfig(
#             r=lora_rank,
#             lora_alpha=lora_alpha,
#             init_lora_weights=init_lora_weights,
#             target_modules=lora_target_modules.split(","),
#             use_rslora=True,
#         )
#         model = inject_adapter_in_model(lora_config, model)
#         for param in model.parameters():
#             # Upcast LoRA parameters into fp32
#             if param.requires_grad:
#                 param.data = param.to(torch.float32)

#     def draw_training_sigma_and_epsilon(
#         self, x0_size: torch.Size, condition: Any
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size = x0_size[0]
#         epsilon = torch.randn(x0_size, device="cuda")
#         sigma_B = self.pipe.scheduler.sample_sigma(batch_size).to(device="cuda")
#         sigma_B_1 = rearrange(
#             sigma_B, "b -> b 1"
#         )  # add a dimension for T, all frames share the same sigma
#         is_video_batch = condition.data_type == DataType.VIDEO

#         multiplier = self.video_noise_multiplier if is_video_batch else 1
#         sigma_B_1 = sigma_B_1 * multiplier
#         if is_video_batch and self.high_sigma_ratio > 0:
#             # Implement the high sigma strategy LOGUNIFORM200_100000
#             LOG_200 = math.log(200)
#             LOG_100000 = math.log(100000)
#             mask = (
#                 torch.rand(sigma_B_1.shape, device=sigma_B_1.device)
#                 < self.high_sigma_ratio
#             )
#             log_new_sigma = (
#                 torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1)
#                 * (LOG_100000 - LOG_200)
#                 + LOG_200
#             )
#             sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
#         return sigma_B_1, epsilon

#     def compute_loss_with_epsilon_and_sigma(
#         self,
#         x0_B_C_T_H_W: torch.Tensor,
#         condition: T2VCondition,
#         epsilon_B_C_T_H_W: torch.Tensor,
#         sigma_B_T: torch.Tensor,
#     ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Compute loss givee epsilon and sigma

#         This method is responsible for computing loss give epsilon and sigma. It involves:
#         1. Adding noise to the input data.
#         2. Passing the noisy data through the network to generate predictions.
#         3. Computing the loss based on the difference between the predictions and the original data, \
#             considering any configured loss weighting.

#         Args:
#             data_batch (dict): raw data batch draw from the training data loader.
#             x0: image/video latent
#             condition: text condition
#             epsilon: noise
#             sigma: noise level

#         Returns:
#             tuple: A tuple containing four elements:
#                 - dict: additional data that used to debug / logging / callbacks
#                 - Tensor 1: kendall loss,
#                 - Tensor 2: MSE loss,
#                 - Tensor 3: EDM loss

#         Raises:
#             AssertionError: If the class is conditional, \
#                 but no number of classes is specified in the network configuration.

#         Notes:
#             - The method handles different types of conditioning
#             - The method also supports Kendall's loss
#         """
#         # Get the mean and stand deviation of the marginal probability distribution.
#         mean_B_C_T_H_W, std_B_T = x0_B_C_T_H_W, sigma_B_T
#         # Generate noisy observations
#         xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(
#             std_B_T, "b t -> b 1 t 1 1"
#         )
#         # make prediction
#         model_pred = self.pipe.denoise(xt_B_C_T_H_W, sigma_B_T, condition)
#         # loss weights for different noise levels
#         weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
#         # extra loss mask for each sample, for example, human faces, hands
#         pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
#         edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(
#             weights_per_sigma_B_T, "b t -> b 1 t 1 1"
#         )
#         kendall_loss = edm_loss_B_C_T_H_W
#         output_batch = {
#             "x0": x0_B_C_T_H_W,
#             "xt": xt_B_C_T_H_W,
#             "sigma": sigma_B_T,
#             "weights_per_sigma": weights_per_sigma_B_T,
#             "condition": condition,
#             "model_pred": model_pred,
#             "mse_loss": pred_mse_B_C_T_H_W.mean(),
#             "edm_loss": edm_loss_B_C_T_H_W.mean(),
#             "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
#         }
#         output_batch["loss"] = kendall_loss.mean()  # check if this is what we want

#         return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

#     def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             sigma (tensor): noise level

#         Returns:
#             loss weights per sigma noise level
#         """
#         return (sigma**2 + self.pipe.sigma_data**2) / (
#             sigma * self.pipe.sigma_data
#         ) ** 2

#     def training_step(self, batch, batch_idx):
#         batch["num_conditional_frames"] = 1
#         # Implement the training step logic here
#         _, x0_B_C_T_H_W, condition = self.pipe.get_data_and_condition(batch)
#         # Sample pertubation noise levels and N(0, 1) noises
#         sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(
#             x0_B_C_T_H_W.size(), condition
#         )
#         output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
#             x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
#         )

#         if self.loss_reduce == "mean":
#             kendall_loss = kendall_loss.mean() * self.loss_scale
#         elif self.loss_reduce == "sum":
#             kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
#         else:
#             raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

#         self.log("loss", kendall_loss.item(), prog_bar=True)
#         self.log("lr", self.optimizers().param_groups[0]["lr"])

#         return kendall_loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.pipe.parameters(),
#             lr=2 ** (-14.5),
#             weight_decay=0.01,
#             betas=(0.9, 0.99),
#         )
#         scheduler = WarmupLinearScheduler(
#             optimizer=optimizer,
#             f_start=1e-6,
#             f_min=0.1,
#             f_max=0.2,
#             warm_up_steps=1000,
#             cycle_lengths=100_000,
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "step",
#                 "frequency": 1,
#             },
#         }


# class CosmosDatasetWrapper(torch.utils.data.Dataset):
#     def __init__(self, dataset: LeRobotDataset):
#         self.dataset = dataset
#         with open("tasks_with_embeddings.pkl", "rb") as f:
#             self.t5_embeddings = pickle.load(f)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         image = item["image"].permute(1, 0, 2, 3)
#         # image = item["observation.images.base"].permute(1, 0, 2, 3)
#         image = (image * 255).to(torch.uint8)
#         t5_text_embeddings = torch.zeros((512, 1024), dtype=torch.float32)
#         t5_text_mask = torch.zeros(512, dtype=torch.int64)

#         task = item["task"]
#         if len(task) > 0:
#             task = task[0].upper() + task[1:].lower()
#             if not task.endswith("."):
#                 task += "."
#         try:
#             emb = self.t5_embeddings[task]
#             t5_text_embeddings[: emb.shape[0]] = torch.tensor(emb, dtype=torch.float32)
#             t5_text_mask[: emb.shape[0]] = 1
#         except KeyError:
#             log.warning(
#                 f"Task '{task}' not found in T5 embeddings. Using zero embeddings."
#             )

#         return {
#             "video": image,
#             "t5_text_embeddings": t5_text_embeddings[:32],
#             "t5_text_mask": t5_text_mask[:32],
#             "fps": torch.tensor(5.0, dtype=torch.float32),  # Assuming a fixed FPS of 5
#             "padding_mask": torch.zeros((1, 224, 224), dtype=torch.float32),
#         }


# model = Predict2Video2WorldModelLightning(
#     "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt",
#     train_architecture="none",
# )

# _dataset = LeRobotDataset(
#     # episodes=[i for i in range(10)],
#     repo_id="ZibinDong/bridgedatav2_train",
#     root="/openbayes/input/input0",
#     delta_timestamps={
#         "image": [i / 5 for i in range(21)],
#     },
#     image_transforms=Resize((480, 480)),
# )

# dataset = CosmosDatasetWrapper(_dataset)
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=2, num_workers=11, shuffle=True, persistent_workers=True
# )

# callback = ModelCheckpoint(
#     dirpath="checkpoints",
#     filename="model-{epoch}-{step}",
#     save_top_k=-1,
#     every_n_train_steps=10_000,
# )

# # policy = {
# #     Block,
# # }
# # strategy = FSDPStrategy(auto_wrap_policy=policy, sharding_strategy="SHARD_GRAD_OP")

# tb_logger = pl_loggers.TensorBoardLogger(save_dir="../tf_dir/")

# trainer = L.Trainer(
#     accelerator="cuda",
#     devices=8,
#     max_steps=100_000,
#     # max_epochs=2,
#     enable_progress_bar=True,
#     strategy="ddp",
#     precision="bf16-mixed",
#     callbacks=[callback],
#     log_every_n_steps=5,
#     # overfit_batches=5,
#     accumulate_grad_batches=1,
#     gradient_clip_val=1.0,
#     logger=tb_logger,
# )

# trainer.fit(model, dataloader)
