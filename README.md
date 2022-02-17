# usage
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

# benchmark log

For example, run ```python benchmark.py --model_dir=data/MobileNetV1/ --config_file=conf/config.yaml --input_shape=3,224,224 --enable_gpu=true --gpu_id=2 --enable_trt=true```

```
{
  'detail': {
    'total': 100, 'result': {'50.0': 0.001189, '80.0': 0.001196, '90.0': 0.0012, '95.0': 0.001205, '99.0': 0.00122, '99.9': 0.00122, 'avg_cost': 0.001192}
  }, 
  'gpu_stat': {
    'index': '2', 'uuid': 'GPU-76134271-1eb0-247b-fcad-06de45a5022f', 'name': 'Tesla T4', 'timestamp': '2022/02/17 07:35:24.467', 'memory.total': '15109', 'memory.free': '13135', 'memory.used': '2096', 'utilization.gpu': '2', 'utilization.memory': '0'}, 
  'backend_type': 'paddle', 
  'batch_size': 1, 
  'precision': None, 
  'enable_gpu': True, 
  'enable_trt': True
}
```
