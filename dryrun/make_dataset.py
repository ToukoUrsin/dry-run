"""Create a small, real LeRobot dataset from MuJoCo camera frames."""
from pathlib import Path
import numpy as np
import mujoco
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MODEL = Path(__file__).resolve().parents[1] / "assets/SO101/Simulation/SO101/dryrun_station.xml"
OUT = Path("/bulk2/runs/dryrun/m4/lerobot-dataset")

def main():
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 64, 64)
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (6,)},
        "observation.images.top": {"dtype": "image", "shape": (64, 64, 3)},
    }
    ds = LeRobotDataset.create("dryrun-mujoco", fps=30, features=features,
                               root=OUT, robot_type="so101", use_videos=False,
                               image_writer_processes=0, image_writer_threads=0)
    for episode in range(4):
        for frame in range(60):
            q = np.array([0.12*np.sin(frame/12 + episode), -0.25 + 0.08*np.cos(frame/15),
                          0.2*np.sin(frame/18), 0.1*np.cos(frame/20), 0.05*np.sin(frame/10), 0.4], dtype=np.float32)
            data.qpos[:6] = q; mujoco.mj_forward(model, data)
            renderer.update_scene(data, "top_view")
            image = renderer.render()
            action = np.array([0.12*np.cos(frame/12 + episode), 0.08*np.sin(frame/15),
                               0.2*np.cos(frame/18), -0.1*np.sin(frame/20), 0.05*np.cos(frame/10), 0.4], dtype=np.float32)
            ds.add_frame({"observation.state": q, "action": action,
                          "observation.images.top": image, "task": "sort bricks"})
        ds.save_episode()
    ds.finalize()
    print({"root": str(OUT), "episodes": 4, "frames": 240, "model": str(MODEL)})

if __name__ == "__main__": main()
