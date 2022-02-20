
backend_type=(onnxruntime paddle)
enable_trt=(true false)
enable_gpu=(true false)
gpu_id=0
input_shape=(3,224,224)
batch_size=(1 8 16)
config_file=config.yaml
model_dir=./Models/TestModel2

echo "============ Benchmark result =============" >> result.txt
for backend_type in ${backend_type[@]};do
  if [ "$backend_type" = "onnxruntime" ];then
    if [ ! -f "$model_dir/model.onnx" ];then
      continue
    fi
  fi
  for batch_size in ${batch_size[@]};do
    python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --enable_gpu=true --gpu_id=2 --enable_trt=true --backend_type=${backend_type} --batch_size=${batch_size}
  done
done
