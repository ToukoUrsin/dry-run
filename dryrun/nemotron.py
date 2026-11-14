"""Call the real NVIDIA Nemotron failure analyst and save the pair."""
from __future__ import annotations
import argparse, base64, json, os, time
from pathlib import Path
from urllib.request import Request, urlopen

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
SCHEMA = {"reason": "string", "scenes": [{"seed": "integer", "difficulty": "number", "defect": "string"}]}

def call(frame: Path, seed: int, out: Path):
    key=os.environ.get("NVIDIA_API_KEY")
    if not key: raise RuntimeError("NVIDIA_API_KEY is not injected; run through hsec")
    data=base64.b64encode(frame.read_bytes()).decode()
    prompt=("Analyze this failed MuJoCo brick-sorting rollout. Return only JSON matching this schema: "
            +json.dumps(SCHEMA)+". Explain the physical failure and propose harder next scenes.")
    body={"model":MODEL,"temperature":0,"max_tokens":2000,"messages":[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+data}}]}]}
    out.mkdir(parents=True,exist_ok=True); (out/"request.json").write_text(json.dumps({"model":MODEL,"seed":seed,"schema":SCHEMA,"frame":str(frame)},indent=2))
    req=Request("https://integrate.api.nvidia.com/v1/chat/completions",data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    with urlopen(req,timeout=120) as resp: raw=json.loads(resp.read())
    (out/"raw_response.json").write_text(json.dumps(raw,indent=2))
    message=raw["choices"][0]["message"]
    text=message.get("content")
    if isinstance(text,list): text="".join(part.get("text","") for part in text if isinstance(part,dict))
    if not text: text=message.get("reasoning_content", "{}")
    parsed=json.loads(text); result={"model":MODEL,"seed":seed,"analysis":parsed,"raw":raw}
    (out/"response.json").write_text(json.dumps(result,indent=2)); return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--frame",type=Path,required=True);p.add_argument("--seed",type=int,default=7);p.add_argument("--out",type=Path,default=Path("/bulk2/runs/dryrun/m5"));a=p.parse_args(); print(json.dumps(call(a.frame,a.seed,a.out),indent=2))
if __name__=="__main__": main()
