from __future__ import annotations
import json,time
from pathlib import Path
import cv2,numpy as np,openvino as ov

def evaluate(model_path:Path,data:Path,out:Path):
 core=ov.Core(); compiled=core.compile_model(core.read_model(str(model_path)),'CPU'); inp=compiled.input(0); infer=compiled.create_infer_request()
 rows=[]; lat=[]
 groups=[('good',data/'good'/'test',False)]+[(d,data/'defects'/d,True) for d in ['scribble','dot','sticker','chipped_stud','missing_stud','discolouring']]
 for label,folder,is_defect in groups:
  for p in sorted(folder.glob('*.png')):
   b=cv2.imread(str(p)); x=cv2.cvtColor(b,cv2.COLOR_BGR2RGB).transpose(2,0,1)[None].astype(np.float32)/255
   t=time.perf_counter_ns(); o=infer.infer({inp:x}); lat.append((time.perf_counter_ns()-t)/1e6)
   pred=int(np.asarray(o['pred_label']).reshape(-1)[0]); score=float(np.asarray(o['pred_score']).reshape(-1)[0]); rows.append({'path':str(p),'type':label,'actual_defect':is_defect,'pred_defect':bool(pred),'score':score})
 # threshold from model labels; report full table by held-out type.
 table={}
 for typ in sorted(set(r['type'] for r in rows)):
  rs=[r for r in rows if r['type']==typ]; table[typ]={'n':len(rs),'tp':sum(r['actual_defect'] and r['pred_defect'] for r in rs),'tn':sum((not r['actual_defect']) and (not r['pred_defect']) for r in rs),'fp':sum((not r['actual_defect']) and r['pred_defect'] for r in rs),'fn':sum(r['actual_defect'] and (not r['pred_defect']) for r in rs),'mean_score':sum(r['score'] for r in rs)/len(rs)}
 result={'n':len(rows),'table':table,'latency_ms':{'mean':float(np.mean(lat)),'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95))},'model':str(model_path)}; out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': evaluate(Path('/bulk2/runs/dryrun/m2/model/openvino/weights/openvino/dryrun_inspector.xml'),Path('/bulk2/runs/dryrun/m2/data'),Path('/bulk2/runs/dryrun/m2/evaluation.json'))
