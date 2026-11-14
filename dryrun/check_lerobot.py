from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("dryrun-mujoco", root="/bulk2/runs/dryrun/m4/lerobot-dataset", download_videos=False)
print({"frames": len(ds), "episodes": ds.meta.total_episodes, "features": sorted(ds.meta.features)})
