"""Run the learned image+joint policy in MuJoCo on held-out seeds."""
from pathlib import Path
import argparse, json, time
import numpy as np
import mujoco
import torch
from .train_policy import VisionStatePolicy, ROOT as RUN_ROOT
from .mjcf_station import station_xml, layout_bricks, ROOT

def one(model_policy, seed):
    sentence="good and defective bowls on the sides, inspection camera in front"
    xml=Path('/tmp') / f'dryrun_policy_{seed}.xml'
    xml.write_text(station_xml(seed,sentence,robot_include=str((ROOT/'assets/SO101/Simulation/SO101/so101_dryrun.xml').resolve())))
    model=mujoco.MjModel.from_xml_path(str(xml)); data=mujoco.MjData(model); renderer=mujoco.Renderer(model,64,64)
    image=torch.zeros(1,3,64,64)
    for step in range(1800):
        if step % 10 == 0:
            renderer.update_scene(data,'top_view'); image=torch.from_numpy(renderer.render().transpose(2,0,1)).float().unsqueeze(0)/255
        state=torch.from_numpy(np.asarray(data.qpos[:6],dtype=np.float32)).unsqueeze(0)
        with torch.no_grad(): action=model_policy(state,image).numpy()[0]
        data.ctrl[:6]=np.clip(action,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1]); mujoco.mj_step(model,data)
    specs=layout_bricks(seed,sentence); successes=0
    for spec in specs:
        bid=model.body(f'brick_{spec.index}').id; pos=data.xpos[bid]; target=np.array([.35,-.30 if spec.defect=='none' else .30]); successes += int(np.linalg.norm(pos[:2]-target)<.14 and pos[2]<.10)
    return {'seed':seed,'bricks':len(specs),'successes':successes,'score':successes/len(specs)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=10);p.add_argument('--out',type=Path,default=Path('/bulk2/runs/dryrun/m4/policy-benchmark.json'));a=p.parse_args()
    policy=VisionStatePolicy();ckpt=torch.load(RUN_ROOT/'policy.pt',map_location='cpu',weights_only=False);policy.load_state_dict(ckpt['state_dict']);policy.eval()
    rows=[one(policy,s) for s in range(a.seeds)]; out={'checkpoint':str(RUN_ROOT/'policy.pt'),'seeds':a.seeds,'bricks':sum(x['bricks'] for x in rows),'successes':sum(x['successes'] for x in rows),'score':sum(x['successes'] for x in rows)/sum(x['bricks'] for x in rows),'rows':rows};a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
