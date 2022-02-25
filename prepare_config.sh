export BASEPATH=$(cd `dirname $0`; pwd)
export MODELPATH="$BASEPATH/Models"

generate_config(){
  for file in $(ls $model_dir)
    do
      if [ "${file##*.}"x = "txt"x ];then
        config_txt_file=${model_dir}/$file
	yaml_file=${model_dir}/config.yaml
	python generate_yaml.py --input_file ${config_txt_file} --yaml_file ${yaml_file}
        echo "generate config success, model: $model_dir"
      fi
  done
}

echo "============ prepare config =============" >> result.txt

for dir in $(ls $MODELPATH);do
  CONVERTPATH=$MODELPATH/$dir
  echo " >>>> Model path: $CONVERTPATH"
  export model_dir=$CONVERTPATH
  generate_config
done
