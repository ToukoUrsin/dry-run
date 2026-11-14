"""Export and time the trained policy on OpenVINO CPU."""
from pathlib import Path
import json, time
import numpy as np
import torch
import openvino as ov
from .train_policy import VisionStatePolicy, ROOT

def main():
    model = VisionStatePolicy(); ckpt=torch.load(ROOT/"policy.pt", map_location="cpu", weights_only=False); model.load_state_dict(ckpt["state_dict"]); model.eval()
    onnx=ROOT/"policy.onnx"
    state=torch.zeros(1,6); image=torch.zeros(1,3,64,64)
    torch.onnx.export(model,(state,image),onnx,input_names=["state","image"],output_names=["action"],opset_version=17)
    core=ov.Core(); compiled=core.compile_model(str(onnx),"CPU"); infer=compiled.create_infer_request(); inputs={compiled.inputs[0]:np.zeros((1,6),np.float32),compiled.inputs[1]:np.zeros((1,3,64,64),np.float32)}
    for _ in range(20): infer.infer(inputs)
    t=time.perf_counter()
    for _ in range(200): infer.infer(inputs)
    latency=(time.perf_counter()-t)*1000/200
    result={"onnx":str(onnx),"openvino":ov.__version__,"device":"CPU","iterations":200,"latency_ms":latency}
    (ROOT/"openvino.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__ == "__main__": main()
