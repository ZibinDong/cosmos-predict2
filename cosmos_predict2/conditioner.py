# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import omegaconf
import torch
import torch.nn as nn
from torch.distributed import ProcessGroup

from cosmos_predict2.functional.batch_ops import batch_mul
from cosmos_predict2.utils.context_parallel import broadcast, broadcast_split_tensor
from cosmos_predict2.utils.misc import count_params, disabled_train
from imaginaire.lazy_config import instantiate
from imaginaire.utils import log

T = TypeVar("T", bound="BaseCondition")


class DataType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    MIX = "mix"

    def __str__(self) -> str:
        return self.value


def broadcast_condition(condition: BaseCondition, process_group: Optional[ProcessGroup] = None) -> BaseCondition:
    """
    Broadcast the condition from the minimum rank in the specified group(s).
    """
    if condition.is_broadcasted:
        return condition

    kwargs = condition.to_dict(skip_underscore=False)
    for key, value in kwargs.items():
        if value is not None:
            kwargs[key] = broadcast(value, process_group)
    kwargs["_is_broadcasted"] = True
    return type(condition)(**kwargs)


# AbstractEmbModel and its children
class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._is_trainable = None
        self._dropout_rate = None
        self._input_key = None
        self._return_dict = False

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def dropout_rate(self) -> Union[float, torch.Tensor]:
        return self._dropout_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def is_return_dict(self) -> bool:
        return self._return_dict

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @dropout_rate.setter
    def dropout_rate(self, value: Union[float, torch.Tensor]):
        self._dropout_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_return_dict.setter
    def is_return_dict(self, value: bool):
        self._return_dict = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @dropout_rate.deleter
    def dropout_rate(self):
        del self._dropout_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    @is_return_dict.deleter
    def is_return_dict(self):
        del self._return_dict

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        return batch_mul(
            torch.bernoulli((1.0 - dropout_rate) * torch.ones(in_tensor.shape[0])).type_as(in_tensor),
            in_tensor,
        )

    def details(self) -> str:
        return ""

    def summary(self) -> str:
        input_key = self.input_key if self.input_key is not None else getattr(self, "input_keys", None)
        return (
            f"{self.__class__.__name__} \n\tinput key: {input_key}"
            f"\n\tParam count: {count_params(self, False)} \n\tTrainable: {self.is_trainable}"
            f"\n\tDropout rate: {self.dropout_rate}"
            f"\n\t{self.details()}"
        )


class TextAttr(AbstractEmbModel):
    def __init__(self, input_key: List[str], dropout_rate: Optional[float] = 0.0):
        super().__init__()
        self._input_key = input_key
        self._dropout_rate = dropout_rate

    def forward(self, token: torch.Tensor):
        return {"crossattn_emb": token}

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        if key is not None and "mask" in key:
            return in_tensor
        return super().random_dropout_input(in_tensor, dropout_rate, key)

    def details(self) -> str:
        return "Output key: [crossattn_emb]"


class ReMapkey(AbstractEmbModel):
    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        dropout_rate: Optional[float] = 0.0,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        self.output_key = output_key
        self.dtype = {
            None: None,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "half": torch.float16,
            "float16": torch.float16,
            "int": torch.int32,
            "long": torch.int64,
        }[dtype]
        self._input_key = input_key
        self._output_key = output_key
        self._dropout_rate = dropout_rate

    def forward(self, element: torch.Tensor) -> Dict[str, torch.Tensor]:
        key = self.output_key if self.output_key else self.input_key
        if isinstance(element, torch.Tensor):
            element = element.to(dtype=self.dtype)
        return {key: element}

    def details(self) -> str:
        key = self.output_key if self.output_key else self.input_key
        return f"Output key: {key} \n\tDtype: {self.dtype}"


class BooleanFlag(AbstractEmbModel):
    def __init__(self, input_key: str, output_key: Optional[str] = None, dropout_rate: Optional[float] = 0.0):
        super().__init__()
        self._input_key = input_key
        self._dropout_rate = dropout_rate
        self.output_key = output_key

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        del args, kwargs
        key = self.output_key if self.output_key else self.input_key
        return {key: self.flag}

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        self.flag = torch.bernoulli((1.0 - dropout_rate) * torch.ones(1)).bool().to(device=in_tensor.device)
        return in_tensor

    def details(self) -> str:
        key = self.output_key if self.output_key else self.input_key
        return f"Output key: {key} \n\t This is a boolean flag"


# ------------------- condition classes -------------------


# BaseCondition and its children
@dataclass(frozen=True)
class BaseCondition(ABC):
    """
    Base class for condition data structures that hold conditioning information for generation models.

    Attributes:
        _is_broadcasted: Flag indicating if parallel broadcast splitting
            has been performed. This is an internal implementation detail.
    """

    _is_broadcasted: bool = False

    def to_dict(self, skip_underscore: bool = True) -> Dict[str, Any]:
        """Converts the condition to a dictionary.

        Returns:
            Dictionary containing the condition's fields and values.
        """
        # return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        return {f.name: getattr(self, f.name) for f in fields(self) if not (f.name.startswith("_") and skip_underscore)}

    @property
    def is_broadcasted(self) -> bool:
        return self._is_broadcasted

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> BaseCondition:
        """Broadcasts and splits the condition across the checkpoint parallelism group.
        For most condition, such asTextCondition, we do not need split.

        Args:
            process_group: The process group for broadcast and split

        Returns:
            A new BaseCondition instance with the broadcasted and split condition.
        """
        if self.is_broadcasted:
            return self
        return broadcast_condition(self, process_group)


@dataclass(frozen=True)
class TextCondition(BaseCondition):
    crossattn_emb: Optional[torch.Tensor] = None
    data_type: DataType = DataType.VIDEO
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None

    def edit_data_type(self, data_type: DataType) -> TextCondition:
        """Edit the data type of the condition.

        Args:
            data_type: The new data type.

        Returns:
            A new TextCondition instance with the new data type.
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["data_type"] = data_type
        return type(self)(**kwargs)

    @property
    def is_video(self) -> bool:
        return self.data_type == DataType.VIDEO


@dataclass(frozen=True)
class VideoCondition(TextCondition):
    use_video_condition: bool = False
    # the following two attributes are used to set the video condition; during training, inference
    gt_frames: Optional[torch.Tensor] = None
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
    ) -> VideoCondition:
        """
        Sets the video conditioning frames for video-to-video generation.

        This method creates a conditioning mask for the input video frames that determines
        which frames will be used as context frames for generating new frames. The method
        handles both image batches (T=1) and video batches (T>1) differently.

        Args:
            gt_frames: A tensor of ground truth frames with shape [B, C, T, H, W], where:
                B = batch size
                C = number of channels
                T = number of frames
                H = height
                W = width

            random_min_num_conditional_frames: Minimum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            random_max_num_conditional_frames: Maximum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            num_conditional_frames: Optional; If provided, all examples in the batch will use
                exactly this many frames for conditioning. If None, a random number of frames
                between random_min_num_conditional_frames and random_max_num_conditional_frames
                will be selected for each example in the batch.

        Returns:
            A new VideoCondition object with the gt_frames and conditioning mask set.
            The conditioning mask (condition_video_input_mask_B_C_T_H_W) is a binary tensor
            of shape [B, 1, T, H, W] where 1 indicates frames used for conditioning and 0
            indicates frames to be generated.

        Notes:
            - For image batches (T=1), no conditioning frames are used (num_conditional_frames_B = 0).
            - For video batches:
                - If num_conditional_frames is provided, all examples use that fixed number of frames.
                - Otherwise, each example randomly uses between random_min_num_conditional_frames and
                random_max_num_conditional_frames frames.
            - The mask marks the first N frames as conditioning frames (set to 1) for each example.
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = gt_frames

        # condition_video_input_mask_B_C_T_H_W
        B, _, T, H, W = gt_frames.shape
        condition_video_input_mask_B_C_T_H_W = torch.zeros(
            B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device
        )
        if T == 1:  # handle image batch
            num_conditional_frames_B = torch.zeros(B, dtype=torch.int32)
        else:  # handle video batch
            if num_conditional_frames is not None:
                num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames
            else:
                num_conditional_frames_B = torch.randint(
                    random_min_num_conditional_frames, random_max_num_conditional_frames + 1, size=(B,)
                )
        for idx in range(B):
            condition_video_input_mask_B_C_T_H_W[idx, :, : num_conditional_frames_B[idx], :, :] += 1

        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        return type(self)(**kwargs)

    def edit_for_inference(self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1) -> VideoCondition:
        _condition = self.set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        if not is_cfg_conditional:
            # Do not use classifier free guidance on conditional frames.
            # YB found that it leads to worse results.
            _condition.use_video_condition.fill_(True)
        return _condition

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> VideoCondition:
        if self.is_broadcasted:
            return self
        # extra efforts
        gt_frames = self.gt_frames
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        new_condition = TextCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            if T > 1 and process_group.size() > 1:
                gt_frames = broadcast_split_tensor(gt_frames, seq_dim=2, process_group=process_group)
                condition_video_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_T_H_W, seq_dim=2, process_group=process_group
                )
        kwargs["gt_frames"] = gt_frames
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        return type(self)(**kwargs)


@dataclass(frozen=True)
class GR00TV1VideoCondition(TextCondition):
    gt_first_frame: Optional[torch.Tensor] = None
    use_image_condition: bool = False
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None

    def edit_video_condition(self, x0_B_C_T_H_W, process_group: Optional[ProcessGroup] = None) -> GR00TV1VideoCondition:
        """Edit the video condition to include the video mask information.

        Args:
            x0_B_C_T_H_W: The first frame of the video.

        Returns:
            A new GR00TV1VideoCondition instance with the video mask information.
        """
        pg_size = 1 if process_group is None else process_group.size()
        kwargs = self.to_dict(skip_underscore=False)
        B, _, T, H, W = x0_B_C_T_H_W.shape
        condition_video_input_mask = torch.zeros((B, 1, T, H, W), dtype=x0_B_C_T_H_W.dtype, device=x0_B_C_T_H_W.device)
        if pg_size == 1 or process_group.rank() == 0:
            kwargs["gt_first_frame"] = x0_B_C_T_H_W[:, :, 0].detach()
            condition_video_input_mask[:, :, 0] += 1
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask
        return type(self)(**kwargs)


@dataclass(frozen=True)
class ActionCondition(VideoCondition):
    action: Optional[torch.Tensor] = None


# ------------------- conditioner classes -------------------


# Conditioners
class GeneralConditioner(nn.Module, ABC):
    """
    Base class for processing modules that transform input data into BaseCondition objects using embedders.

    An abstract module designed to handle various embedding models with conditional and unconditional configurations.
    This abstract base class initializes and manages a collection of embedders that can dynamically adjust
    their dropout rates based on conditioning.

    Attributes:
        KEY2DIM (dict): A mapping from output keys to dimensions used for concatenation.
        embedders (nn.ModuleDict): A dictionary containing all embedded models initialized and configured
                                   based on the provided configurations.

    Parameters:
        emb_models (Union[List, Any]): A dictionary where keys are embedder names and values are configurations
                                       for initializing the embedders.
    """

    KEY2DIM = {"crossattn_emb": 1}

    def __init__(self, **emb_models: Union[List, Any]):
        super().__init__()
        self.embedders = nn.ModuleDict()
        for n, (emb_name, emb_config) in enumerate(emb_models.items()):
            embedder = instantiate(emb_config)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = getattr(emb_config, "is_trainable", True)
            embedder.dropout_rate = getattr(emb_config, "dropout_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()

            log.debug(f"Initialized embedder #{n}-{emb_name}: \n {embedder.summary()}")
            self.embedders[emb_name] = embedder

    @abstractmethod
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Any:
        """Should be implemented in subclasses to handle conditon datatype"""
        raise NotImplementedError

    def _forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Processes the input batch through all configured embedders, applying conditional dropout rates if specified.
        Output tensors for each key are concatenated along the dimensions specified in KEY2DIM.

        Parameters:
            batch (Dict): The input data batch to process.
            override_dropout_rate (Optional[Dict[str, float]]): Optional dictionary to override default dropout rates
                                                                per embedder key.

        Returns:
            Dict: A dictionary of output tensors concatenated by specified dimensions.

        Note:
            In case the network code is sensitive to the order of concatenation, you can either control the order via \
            config file or make sure the embedders return a unique key for each output.
        """
        output = defaultdict(list)
        if override_dropout_rate is None:
            override_dropout_rate = {}

        # make sure emb_name in override_dropout_rate is valid
        for emb_name in override_dropout_rate.keys():
            assert emb_name in self.embedders, f"invalid name found {emb_name}"

        for emb_name, embedder in self.embedders.items():
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if isinstance(embedder.input_key, str):
                    emb_out = embedder(
                        embedder.random_dropout_input(
                            batch[embedder.input_key], override_dropout_rate.get(emb_name, None)
                        )
                    )
                elif isinstance(embedder.input_key, (list, omegaconf.listconfig.ListConfig)):
                    emb_out = embedder(
                        *[
                            embedder.random_dropout_input(batch.get(k), override_dropout_rate.get(emb_name, None), k)
                            for k in embedder.input_key
                        ]
                    )
                else:
                    raise KeyError(
                        f"Embedder '{embedder.__class__.__name__}' requires an 'input_key' attribute to be defined as either a string or list of strings"
                    )
            for k, v in emb_out.items():
                output[k].append(v)
        # Concatenate the outputs
        return {k: torch.cat(v, dim=self.KEY2DIM.get(k, -1)) for k, v in output.items()}

    def get_condition_uncondition(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        Processes the provided data batch to generate two sets of outputs: conditioned and unconditioned. This method
        manipulates the dropout rates of embedders to simulate two scenarios — one where all conditions are applied
        (conditioned), and one where they are removed or reduced to the minimum (unconditioned).

        This method first sets the dropout rates to zero for the conditioned scenario to fully apply the embedders' effects.
        For the unconditioned scenario, it sets the dropout rates to 1 (or to 0 if the initial unconditional dropout rate
        is insignificant) to minimize the embedders' influences, simulating an unconditioned generation.

        Parameters:
            data_batch (Dict): The input data batch that contains all necessary information for embedding processing. The
                            data is expected to match the required format and keys expected by the embedders.

        Returns:
            Tuple[Any, Any]: A tuple containing two condition:
                - The first one contains the outputs with all embedders fully applied (conditioned outputs).
                - The second one contains the outputs with embedders minimized or not applied (unconditioned outputs).
        """
        cond_dropout_rates, dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(data_batch, override_dropout_rate=dropout_rates)
        return condition, un_condition

    def get_condition_with_negative_prompt(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        Similar functionality as get_condition_uncondition
        But use negative prompts for unconditon
        """
        cond_dropout_rates, uncond_dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            if isinstance(embedder, TextAttr):
                uncond_dropout_rates[emb_name] = 0.0
            else:
                uncond_dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        data_batch_neg_prompt = copy.deepcopy(data_batch)
        if "neg_t5_text_embeddings" in data_batch_neg_prompt:
            if isinstance(data_batch_neg_prompt["neg_t5_text_embeddings"], torch.Tensor):
                data_batch_neg_prompt["t5_text_embeddings"] = data_batch_neg_prompt["neg_t5_text_embeddings"]

        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(data_batch_neg_prompt, override_dropout_rate=uncond_dropout_rates)

        return condition, un_condition


class TextConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> TextCondition:
        output = super()._forward(batch, override_dropout_rate)
        return TextCondition(**output)


class VideoConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> VideoCondition:
        output = super()._forward(batch, override_dropout_rate)
        return VideoCondition(**output)


class GR00TV1VideoConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> GR00TV1VideoCondition:
        output = super()._forward(batch, override_dropout_rate)
        return GR00TV1VideoCondition(**output)


class ActionConditioner(VideoConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> ActionCondition:
        output = super()._forward(batch, override_dropout_rate)
        assert "action" in batch, "ActionConditioner requires 'action' in batch"
        output["action"] = batch["action"]
        return ActionCondition(**output)
