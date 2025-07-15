from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "ZibinDong/bridgedatav2_val",
    root="/mnt/20T/datasets/bridgev2/lerobot/ZibinDong/bridgedatav2_val",
    delta_timestamps={"action": [i/5 for i in range(12)]}
)
