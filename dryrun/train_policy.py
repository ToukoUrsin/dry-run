"""CPU imitation training over the real LeRobot MuJoCo dataset."""
from pathlib import Path
import json, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("/bulk2/runs/dryrun/m4")

class VisionStatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6 + 3 * 64 * 64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 6))
    def forward(self, state, image):
        return self.net(torch.cat([state, image.flatten(1)], dim=1))

def main():
    torch.set_num_threads(8)
    ds = LeRobotDataset("dryrun-mujoco", root=ROOT / "lerobot-dataset", download_videos=False)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = VisionStatePolicy()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    logs=[]; started=time.perf_counter()
    for epoch in range(40):
        total=0.0
        for batch in loader:
            state=batch["observation.state"].float(); image=batch["observation.images.top"].float()/255.0; target=batch["action"].float()
            loss=loss_fn(model(state,image),target); opt.zero_grad(); loss.backward(); opt.step(); total += float(loss)*len(state)
        epoch_loss=total/len(ds); logs.append({"epoch":epoch+1,"loss":epoch_loss})
        print(json.dumps(logs[-1]))
    ckpt=ROOT/"policy.pt"; torch.save({"state_dict":model.state_dict(),"architecture":"VisionStatePolicy","input":"MuJoCo rendered image + 6 joint states"},ckpt)
    (ROOT/"training.json").write_text(json.dumps({"epochs":40,"initial_loss":logs[0]["loss"],"final_loss":logs[-1]["loss"],"seconds":time.perf_counter()-started,"checkpoint":str(ckpt),"logs":logs},indent=2))
    print(json.dumps({"checkpoint":str(ckpt),"initial_loss":logs[0]["loss"],"final_loss":logs[-1]["loss"]}))
if __name__ == "__main__": main()
