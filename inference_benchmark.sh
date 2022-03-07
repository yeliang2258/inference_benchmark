
backend_type_list=(onnxruntime paddle)
enable_trt_list=(true false)
enable_gpu_list=(true false)
enable_mkldnn_list=(true false)
gpu_id=0
batch_size_list=(1 8 16)
config_file=config.yaml
export BASEPATH=$(cd `dirname $0`; pwd)
export MODELPATH="$BASEPATH/Models"

run_benchmark(){
  for backend_type in ${backend_type[@]};do
    if [ "$backend_type" = "onnxruntime" ];then
      if [ ! -f "$model_dir/model.onnx" ];then
         echo "cont find ONNX model file. "
        continue
      fi
    fi
    model_file=""
    params_file=""
    for file in $(ls $model_dir)
      do
        if [ "${file##*.}"x = "pdmodel"x ];then
          model_file=$file
          echo "find model file: $model_file"
        fi

        if [ "${file##*.}"x = "pdiparams"x ];then
          params_file=$file
          echo "find param file: $params_file"
        fi
    done
    for batch_size in ${batch_size_list[@]};do
      for enable_gpu in ${enable_gpu_list[@]};do
        if [ ${enable_gpu} = "true" ]; then
          for enable_trt in ${enable_trt_list[@]};do
            python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --enable_gpu=${enable_gpu} --gpu_id=${gpu_id} --enable_trt=${enable_trt} --backend_type=${backend_type} --batch_size=${batch_size} --paddle_model_file "$model_file" --paddle_params_file "$params_file"
          done
        elif [ ${enable_gpu} = "false" ]; then
          for enable_mkldnn in ${enable_mkldnn_list[@]};do
            python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --enable_mkldnn=${enable_mkldnn} --enable_gpu=false --gpu_id=0 --enable_trt=false --backend_type=${backend_type} --batch_size=${batch_size} --paddle_model_file "$model_file" --paddle_params_file "$params_file"
          done
        fi
      done
    done
  done
}

echo "============ Benchmark result =============" >> result.txt

for dir in $(ls $MODELPATH);do
  CONVERTPATH=$MODELPATH/$dir
  echo " >>>> Model path: $CONVERTPATH"
  export model_dir=$CONVERTPATH
  run_benchmark
done
