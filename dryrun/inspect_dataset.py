from lerobot.datasets.lerobot_dataset import LeRobotDataset
d = LeRobotDataset("dryrun-mujoco", root="/bulk2/runs/dryrun/m4/lerobot-dataset", download_videos=False)
x = d[0]
print({k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in x.items()})
