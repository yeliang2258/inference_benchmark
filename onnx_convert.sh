export PYTHON_CMD=python
export BASEPATH=$(cd `dirname $0`; pwd)
export MODELPATH="$BASEPATH/Models"
install_repo(){
  cd "$BASEPATH"
  rm -rf Paddle2ONNX
  git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
  cd Paddle2ONNX
  $PYTHON_CMD setup.py install
}

# install_repo
rm -r result.txt
rm -r *.pdmodel
rm -r *.dot

echo "============ covert and diff check result =============" >> result.txt
for dir in $(ls $MODELPATH)
do
  CONVERTPATH=$MODELPATH/$dir
  model_file=""
  params_file=""
  for file in $(ls $CONVERTPATH)
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
  if [ "$model_file" = "__model__" ];then
      cd $CONVERTPATH
      echo "$model_file"
      paddle2onnx --model_dir ./  --save_file model.onnx --opset_version 13 --enable_onnx_checker True
      cd $BASEPATH
  fi
  if [ "${model_file##*.}"x = "pdmodel"x ];then
    cd $CONVERTPATH
    if [ "$params_file" = "" ];then
      paddle2onnx --model_dir ./  --model_filename "$model_file" --save_file model.onnx --opset_version 13 --enable_onnx_checker True
    else
      paddle2onnx --model_dir ./  --model_filename "$model_file" --params_filename "$params_file" --save_file model.onnx --opset_version 13 --enable_onnx_checker True
    fi  
    cd $BASEPATH
    $PYTHON_CMD model_check.py --config_file config.yaml --model_dir "$CONVERTPATH" --paddle_model_file "$model_file" --paddle_params_file "$params_file"
  fi
done