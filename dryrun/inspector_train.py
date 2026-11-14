"""Rendered-brick Anomalib/PaDiM training and OpenVINO proof harness."""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import cv2, numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'assets/SO101/Simulation/SO101/dryrun_station.xml'
DEFECTS=['scribble','dot','sticker','chipped_stud','missing_stud','discolouring']

def render_dataset(out:Path, train=64, test_each=80):
    rng=random.Random(2301); good_train=out/'good'/'train'; good_test=out/'good'/'test'; defects=out/'defects'
    for p in [good_train,good_test,*[defects/d for d in DEFECTS]]: p.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(MODEL)); renderer=mujoco.Renderer(model,256,256); bid=model.body('brick_0').id; adr=model.jnt_qposadr[model.body_jntadr[bid]]
    # Material IDs used by brick geoms; changing only RGBA changes appearance, not physics.
    mats=[model.geom_matid[model.geom(f'brick_0_body').id]]
    def one(path, seed, defect=None):
        data=mujoco.MjData(model); data.qpos[adr:adr+3]=[.34+rng.uniform(-.025,.025),rng.uniform(-.04,.04),.018+rng.uniform(0,.01)]; data.qpos[adr+3:adr+7]=[1,0,0,0]
        for i in range(1,12):
            b=model.body(f'brick_{i}').id; a=model.jnt_qposadr[model.body_jntadr[b]]; data.qpos[a:a+3]=[.8+i*.01,.8,-.5]
        # Vary brick color and illumination via material, but never use defect labels at inference.
        c=rng.choice([(0.75,.08,.04,1),(.05,.25,.8,1),(.95,.65,.04,1),(.1,.65,.2,1)]); model.mat_rgba[mats[0]]=c
        mujoco.mj_forward(model,data); renderer.update_scene(data,'inspection_view'); img=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        if defect:
            h,w=img.shape[:2]; col=(20,20,20)
            if defect=='scribble': cv2.line(img,(w//3,h//2),(2*w//3,h//2+18),col,4)
            elif defect=='dot': cv2.circle(img,(w//2,h//2),10,col,-1)
            elif defect=='sticker': cv2.rectangle(img,(w//2-18,h//2-12),(w//2+18,h//2+12),(235,235,235),-1)
            elif defect=='discolouring': cv2.rectangle(img,(w//2-20,h//2-15),(w//2+22,h//2+15),(80,150,80),-1)
            elif defect=='missing_stud': cv2.circle(img,(w//2-17,h//2-22),6,(35,35,35),-1)
            elif defect=='chipped_stud': cv2.ellipse(img,(w//2+17,h//2-22),(7,4),0,0,180,(35,35,35),-1)
        cv2.imwrite(str(path),img)
    for i in range(train): one(good_train/f'{i:05d}.png',i)
    for i in range(100): one(good_test/f'{i:05d}.png',1000+i)
    for d in DEFECTS:
        for i in range(test_each): one(defects/d/f'{i:05d}.png',2000+100*i+DEFECTS.index(d),d)
    return {'train_good':train,'test_good':100,'defects':{d:test_each for d in DEFECTS}}

def train(dataset:Path, out:Path):
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Padim
    dm=Folder(name='dryrun_bricks',root=dataset,normal_dir='good/train',normal_test_dir='good/test',abnormal_dir='defects',train_batch_size=8,eval_batch_size=8,num_workers=0,normal_split_ratio=0.0,seed=42)
    model=Padim(backbone='resnet18',layers=['layer1'],pre_trained=False,n_features=10,visualizer=False)
    engine=Engine(default_root_dir=str(out),logger=False,enable_checkpointing=True,accelerator='cpu',devices=1,max_epochs=1,num_sanity_val_steps=0,limit_train_batches=1)
    t=time.perf_counter(); engine.fit(model=model,datamodule=dm); fit_s=time.perf_counter()-t
    ckpts=list(out.rglob('*.ckpt')); ckpt=max(ckpts,key=lambda p:p.stat().st_mtime) if ckpts else None
    # Export the trained memory bank through Anomalib's OpenVINO exporter.
    exported=engine.export(model=model,export_type='openvino',export_root=str(out/'openvino'),model_file_name='dryrun_inspector',input_size=(256,256))
    results=engine.test(model=model,datamodule=dm,ckpt_path=None)
    return {'fit_seconds':fit_s,'checkpoint':str(ckpt) if ckpt else None,'openvino':str(exported),'test_results':results}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',type=Path,default=Path('/bulk2/runs/dryrun/m2/data')); ap.add_argument('--out',type=Path,default=Path('/bulk2/runs/dryrun/m2/model')); ap.add_argument('--render',action='store_true'); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    report={}
    if args.render or not (args.dataset/'good'/'train').exists(): report['dataset']=render_dataset(args.dataset)
    report['training']=train(args.dataset,args.out); print(json.dumps(report,indent=2,default=str)); (args.out/'training.json').write_text(json.dumps(report,indent=2,default=str))
if __name__=='__main__': main()
