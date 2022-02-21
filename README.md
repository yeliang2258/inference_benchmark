# Inference Benchmark
## Usage Steps
```
1. Put all the models to be tested in the Models folder, each model folder must include one config.yaml file.
2. run `bash run.sh` to get the inference benchmark data in result.txt.
```

We provide three scripts:  
```run.sh```: Start the inference benchmark test, which is divided into two steps: model conversion and benchmark output  

```onnx_convert.sh```: Initiate model conversion and accuracy testing of the converted model.  

```inference_benchmark.sh```: Start the benchmark test of the model and output the test results.

## 1. Model Convert And Diff Check
### Model Convert
python paddle2onnx -h
```
usage: paddle2onnx [-h] [--model_dir MODEL_DIR]
                   [--model_filename MODEL_FILENAME]
                   [--params_filename PARAMS_FILENAME] [--save_file SAVE_FILE]
                   [--opset_version OPSET_VERSION]
                   [--input_shape_dict INPUT_SHAPE_DICT]
                   [--enable_onnx_checker ENABLE_ONNX_CHECKER]
                   [--enable_paddle_fallback ENABLE_PADDLE_FALLBACK]
                   [--version]
```
### Model DIff Check
python model_check.py -h
```
usage: model_check.py [-h] [--batch_size BATCH_SIZE]
                      [--input_shape INPUT_SHAPE] [--cpu_threads CPU_THREADS]
                      [--precision {fp32,fp16}]
                      [--backend_type {paddle,onnxruntime}] [--gpu_id GPU_ID]
                      [--model_dir MODEL_DIR]
                      [--paddle_model_file PADDLE_MODEL_FILE]
                      [--paddle_params_file PADDLE_PARAMS_FILE]
                      [--enable_mkldnn ENABLE_MKLDNN]
                      [--enable_openvino ENABLE_OPENVINO]
                      [--enable_gpu ENABLE_GPU] [--enable_trt ENABLE_TRT]
                      [--enable_profile ENABLE_PROFILE]
                      [--enable_benchmark ENABLE_BENCHMARK]
                      [--return_result RETURN_RESULT] --config_file
                      CONFIG_FILE
```
### Log Info
After you run the test, you can get information about the model transformation in result.txt.  
```
convert success : convert success and no diff.
results has diff : Paddle and ONNX model results has diff.
shape is not equal : The shapes of Paddle and ONNX model results are not equal.
dtype is not equal : The dtypes of Paddle and ONNX model results are not equal.
```
## 2. Benchmark Test
python benchmark.py --help
```
usage: benchmark.py [-h] [--batch_size BATCH_SIZE] [--input_shape INPUT_SHAPE]
                    [--cpu_threads CPU_THREADS] [--precision {fp32,fp16}]
                    [--backend_type {paddle,onnxruntime}] [--gpu_id GPU_ID]
                    [--model_dir MODEL_DIR] [--enable_mkldnn ENABLE_MKLDNN]
                    [--enable_gpu ENABLE_GPU] [--enable_trt ENABLE_TRT]
                    [--enable_profile ENABLE_PROFILE]
                    [--enable_benchmark ENABLE_BENCHMARK] --config_file
                    CONFIG_FILE
```

```
# paddle
python benchmark.py --model_dir=data/faster_rcnn_swin_tiny_fpn_1x_coco/ --config_file=conf/config.yaml --input_shape=2:3,640,640:2 --enable_gpu=true

# paddle gpu+trt
/opt/python/cp37-cp37m/bin/python benchmark.py --model_dir=data/MobileNetV1/ --config_file=conf/config.yaml --input_shape=3,224,224 --enable_gpu=true --gpu_id=2 --enable_trt=true

# onnxruntime gpu+trt
python benchmark.py --model_dir=./ --config_file=conf/config.yaml --input_shape=3,224,224 --enable_gpu=true --gpu_id=2 --backend_type=onnxruntime --enable_trt=true
```

## 3. Benchmark Log

For example, run ```python benchmark.py --model_dir=data/MobileNetV1/ --config_file=conf/config.yaml --input_shape=3,224,224 --enable_gpu=true --gpu_id=2 --enable_trt=true```

```
{
  'detail': {
    'total': 100, 'result': {'50.0': 0.001189, '80.0': 0.001196, '90.0': 0.0012, '95.0': 0.001205, '99.0': 0.00122, '99.9': 0.00122, 'avg_cost': 0.001192}
  },
  'gpu_stat': {
    'index': '2', 'uuid': 'GPU-76134271-1eb0-247b-fcad-06de45a5022f', 'name': 'Tesla T4', 'timestamp': '2022/02/17 07:35:24.467', 'memory.total': '15109', 'memory.free': '13135', 'memory.used': '2096', 'utilization.gpu': '2', 'utilization.memory': '0'
  },
  'backend_type': 'paddle',
  'batch_size': 1,
  'precision': None,
  'enable_gpu': True,
  'enable_trt': True
}
```
