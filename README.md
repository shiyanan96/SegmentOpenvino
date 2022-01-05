# SegmentOpenvino
### Build: 

```
cd build/
cmake .
make -j8
```

### Infer with onnx: 

```
./segment infer ../models/model_fixsize_retrain.onnx ../input.jpg ../onnx_res.jpg
```

### Infer with openvinoIR: 

```
./segment infer ../models/model_fixsize_retrain.xml ../input.jpg ../onnx_res.jpg
```

**tools:**
export_onnx.py: pth to onnx
bisenetv2.py: network file