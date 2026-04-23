from pathlib import Path

import onnx

root = Path(__file__).resolve().parent
path = root / "moss_tts_local_fixed_sampled_frame.onnx"
m = onnx.load(str(path), load_external_data=False)
for i in m.graph.input:
    print("IN", i.name, onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type))
for o in m.graph.output:
    print("OUT", o.name, onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type))
