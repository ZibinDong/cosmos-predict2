import json

import h5py
from tqdm import tqdm

from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
import pickle

tasks = dict()
with open(
    "/mnt/20T/datasets/bridgev2/lerobot/ZibinDong/bridgedatav2_val/meta/tasks.jsonl"
) as f:
    for line in tqdm(f):
        info = json.loads(line)
        task_index, task = info["task_index"], info["task"]
        if len(task) > 0:
            task = task[0].upper() + task[1:].lower()
            if not task.endswith("."):
                task += "."
        tasks[task_index] = task

t5_text_encoder = CosmosT5TextEncoder(
    model_name="google-t5/t5-11b",
    device="cuda",
    cache_dir="checkpoints/google-t5/t5-11b",
)


batch_size = 5
tasks_with_embeddings = dict()
languages = list(tasks.values())
for i in tqdm(range(0, len(languages), batch_size)):
    batch_tasks = languages[i : i + batch_size]
    embedding, attn_mask = t5_text_encoder.encode_prompts(batch_tasks, return_mask=True)
    embedding_sz = attn_mask.sum(-1)
    for j, task in enumerate(batch_tasks):
        tasks_with_embeddings[task] = embedding[j, : embedding_sz[j]].cpu().numpy()

with open("so100_tasks_with_embeddings.pkl", "wb") as f:
    pickle.dump(tasks_with_embeddings, f)
