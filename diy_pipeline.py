import pickle

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from torchvision.transforms.v2 import Resize

from cosmos_predict2.configs.base.config_video2world import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_DZB,
)
from cosmos_predict2.pipelines.video2world import SimpleVideo2WorldPipeline
from imaginaire.utils.io import save_image_or_video

with open("tasks_with_embeddings.pkl", "rb") as f:
    tasks_with_embeddings_pkl = pickle.load(f)

# dit_params = dict()
# tokenizer_params = dict()
# params = torch.load("checkpoints/model-epoch=0-step=100000.ckpt")["state_dict"]
# for k, v in params.items():
#     if k.startswith("pipe.dit."):
#         dit_params[k[9:]] = v
#     elif k.startswith("pipe.tokenizer."):
#         tokenizer_params[k[15:]] = v
#     else:
#         print(k)
# torch.save(dit_params, "checkpoints/dit_test.pt")

pipe = SimpleVideo2WorldPipeline.from_config(
    config=PREDICT2_VIDEO2WORLD_PIPELINE_2B_DZB,
    # dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt",
    dit_path="checkpoints/dit_test.pt",
    # text_encoder_path="checkpoints/google-t5/t5-11b",  # disable
    text_encoder_path=None,
    add_lora=False,
)
task = ""
pipe = pipe.to("cuda:0", dtype=torch.bfloat16)
image_path = "assets/bridge_test.jpg"

_emb = list(tasks_with_embeddings_pkl.values())[4]
_key = list(tasks_with_embeddings_pkl.keys())[4]
print(_key)
_emb = tasks_with_embeddings_pkl["Put the red object into the pot."]
t5_text_embeddings = torch.zeros((1, 32, 1024))
t5_text_embeddings[0, : _emb.shape[0], :] = torch.tensor(_emb)

# Run the video generation pipeline.
pipe.config.state_t = 6
video = pipe(
    input_path=image_path,
    prompt="A high-definition video captures the precision of robotic manipulation in an househouding setting. As the video progresses, the robotic arm maintains its steady position, continuing the manipulation process. The robotics arm grab the screwdriver and move it to the right. Realistic. The background is stable and not moving.",
    t5_text_embeddings=t5_text_embeddings,
    image_sz=(480, 480),
    num_sampling_step=20,
    # guidance=-1.0,
)

save_image_or_video(video, "output/test.mp4", fps=10)


# pipe = SimpleActionConditionedVideo2WorldPipeline.from_config(
#     config=ACTION_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B,
#     dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned/model-480p-4fps.pt",
#     text_encoder_path="",  # disable
# )
# image_path = "assets/bridge_0.jpg"
# image = Image.open(image_path).convert("RGB")
# image = image.resize((224, 224))
# image = np.array(image)
# with open("assets/bridge_0.pkl", "rb") as f:
#     actions = pickle.load(f)

# # t5_text_embeddings = torch.zeros((1, 512, 1024))
# # actions = np.zeros((12, 7))
# # actions[:, 0] = np.arange(12) / 12

# # Run the video generation pipeline.
# video = pipe(
#     first_frame=image,
#     actions=actions,
#     prompt="",
#     guidance=-1.0,
# )

# save_image_or_video(video, "output/test.mp4", fps=16)


# dataset = LeRobotDataset(
#     repo_id="ZibinDong/bridgedatav2_train",
#     root="/mnt/20T/datasets/bridgev2/lerobot/ZibinDong/bridgedatav2_train",
#     # image_transforms=Resize((224, 224)),
# )
# idx = 150
# item = dataset[idx]
# image = Image.fromarray((item['image'].permute(1,2,0)*255).to(torch.uint8).numpy())
# task = item['task']
# if len(task) > 0:
#     task = task[0].upper() + task[1:].lower()
#     if not task.endswith("."):
#         task += "."
# image.save("assets/bridge_test.jpg")
# print(task)