"""Strict Nemotron analyst for 448 px event-aligned failure sheets (prompt 024)."""
from __future__ import annotations
import argparse, base64, json, os, re, time, uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
STAGES = ("no_clear_grasp", "missed_grasp", "slipped", "wrong_verdict", "dropped_outside_bowl", "pushed_out_of_reach", "other")
PANELS = ("before_grasp", "end_closing", "presentation", "end_cycle")
SCHEMA = {
    "type":"object", "additionalProperties":False,
    "required":["failure_stage","visible_support","scene_change","evidence_panel"],
    "properties":{
        "failure_stage":{"type":"string","enum":list(STAGES)},
        "visible_support":{"type":"string","minLength":8,"maxLength":400},
        "scene_change":{"type":"string","minLength":8,"maxLength":400},
        "evidence_panel":{"type":"string","enum":list(PANELS)},
    },
}

def _sentence(x: Any) -> bool:
    return isinstance(x,str) and 8 <= len(x.strip()) <= 400 and "\n" not in x

def validate(value: Any) -> dict[str,Any]:
    if not isinstance(value,Mapping): raise ValueError("response is not an object")
    required=set(SCHEMA["required"]); extra=set(value)-required; missing=required-set(value)
    errors=[]
    if extra: errors.append("unknown fields: "+",".join(sorted(extra)))
    if missing: errors.append("missing fields: "+",".join(sorted(missing)))
    if value.get("failure_stage") not in STAGES: errors.append("invalid failure_stage")
    if not _sentence(value.get("visible_support")): errors.append("visible_support must be one line of 8-400 chars")
    if not _sentence(value.get("scene_change")): errors.append("scene_change must be one line of 8-400 chars")
    if value.get("evidence_panel") not in PANELS: errors.append("invalid evidence_panel")
    if errors: raise ValueError("; ".join(errors))
    return {k:str(value[k]).strip() for k in required}

def _sheet_payload(path: str) -> dict[str,Any]:
    return {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(Path(path).read_bytes()).decode("ascii")}}

def build_request(row: Mapping[str,Any]) -> dict[str,Any]:
    text=("You are judging one failure of a vision-only brick sorting robot. "
          "The single supplied image is a 2x2 sheet of four real rollout frames, "
          "captioned before_grasp, end_closing, presentation, and end_cycle. "
          "Use only visible evidence in the sheet. Return exactly one JSON object "
          "with failure_stage, visible_support, scene_change, and evidence_panel. "
          "failure_stage must be one of: no_clear_grasp, missed_grasp, slipped, "
          "wrong_verdict, dropped_outside_bowl, pushed_out_of_reach, other. "
          "visible_support states what in the image supports the stage. "
          "scene_change states one concrete scene change that would train against it. "
          "evidence_panel names the one captioned panel that best supports the judgment. "
          "Do not infer simulator events, hidden labels, or object metadata.")
    # The analyst needs the answer object, not a hidden chain of thought.  The
    # NVIDIA endpoint supports disabling reasoning for this model; this keeps
    # 448 px requests fast enough for the 300-row evaluation and leaves the
    # four-field response in the visible content channel.
    return {"model":MODEL,"temperature":0,"max_tokens":1200,"reasoning_effort":"none",
            "messages":[{"role":"user","content":[{"type":"text","text":text},_sheet_payload(str(row["sheet"]))]}],
            "response_format":{"type":"json_object"},"metadata":{"analyst_version":"prompt024-448-sheet","rollout_id":str(row["id"]),"schema":SCHEMA}}

def _content(raw: Mapping[str,Any]) -> str:
    m=raw.get("choices",[{}])[0].get("message",{}); c=m.get("content","")
    if isinstance(c,list): c="".join(str(p.get("text","")) for p in c if isinstance(p,Mapping))
    c=str(c or m.get("reasoning_content","")).strip(); c=re.sub(r"^```(?:json)?\s*|\s*```$","",c,flags=re.I).strip()
    # Some NVIDIA responses place the final object at the end of
    # reasoning_content while leaving message.content null.  Recover a
    # complete object with this request's required keys, rather than parsing
    # the first brace from an explanatory trace.
    try:
        value=json.loads(c)
        if isinstance(value,Mapping) and set(SCHEMA["required"]).issubset(value):
            return json.dumps(value,separators=(",",":"))
    except json.JSONDecodeError:
        pass
    decoder=json.JSONDecoder()
    for match in re.finditer(r"\{",c):
        try:
            value,_=decoder.raw_decode(c[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value,Mapping) and set(SCHEMA["required"]).issubset(value):
            return json.dumps(value,separators=(",",":"))
    return c

class Client:
    def __init__(self,out_dir: str|Path, key: str|None=None, interval:float=1.6, retries:int=2):
        self.key=key or os.environ.get("NVIDIA_API_KEY")
        if not self.key: raise RuntimeError("NVIDIA_API_KEY is not injected; use hsec")
        self.out=Path(out_dir); self.out.mkdir(parents=True,exist_ok=True); self.interval=float(interval); self.retries=int(retries); self.last=0.0
    def analyse(self,row: Mapping[str,Any]) -> dict[str,Any]:
        rid=str(row["id"]); folder=self.out/rid; folder.mkdir(parents=True,exist_ok=True); body=build_request(row); (folder/"request.json").write_text(json.dumps(body,indent=2)); last=None
        for attempt in range(self.retries+1):
            delay=self.interval-(time.monotonic()-self.last)
            if delay>0: time.sleep(delay)
            try:
                # Pace every attempt, including retries after malformed or
                # rate-limited responses.  Updating last before decoding is
                # important: a failed response still consumed provider quota.
                req=Request(ENDPOINT,data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"})
                with urlopen(req,timeout=120) as resp:
                    payload=resp.read()
                self.last=time.monotonic()
                try:
                    raw=json.loads(payload)
                except json.JSONDecodeError:
                    (folder/f"raw_response_{attempt}.txt").write_bytes(payload)
                    raise
                (folder/f"raw_response_{attempt}.json").write_text(json.dumps(raw,indent=2))
                diagnosis=validate(json.loads(_content(raw))); result={"id":rid,"model":MODEL,"version":"prompt024-448-sheet","attempt":attempt+1,"diagnosis":diagnosis}
                (folder/"response.json").write_text(json.dumps(result,indent=2)); return result
            except (HTTPError,URLError,TimeoutError,json.JSONDecodeError,ValueError,KeyError) as exc:
                self.last=time.monotonic()
                last=repr(exc)
                if attempt>=self.retries: break
                time.sleep(min(30.0,2**attempt))
        (folder/"error.json").write_text(json.dumps({"error":last},indent=2)); return {"id":rid,"version":"prompt024-448-sheet","error":last}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--interval",type=float,default=1.6); p.add_argument("--retries",type=int,default=2); p.add_argument("--limit",type=int,default=None); a=p.parse_args(); rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]; rows=rows[:a.limit] if a.limit else rows; c=Client(a.out,interval=a.interval,retries=a.retries)
    out=a.out/"results.jsonl"; done={}
    if out.exists():
        for x in out.read_text().splitlines():
            if x.strip():
                r=json.loads(x)
                if "diagnosis" in r: done[str(r["id"])]=1
    with out.open("a") as fh:
        for row in rows:
            if str(row["id"]) in done: continue
            r=c.analyse(row); fh.write(json.dumps(r,separators=(",",":"))+"\n"); fh.flush(); print(json.dumps({"id":row["id"],"ok":"diagnosis" in r}),flush=True)
if __name__=="__main__": main()
