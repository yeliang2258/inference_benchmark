echo ">>> Donwnload model link url ..."
wget https://paddle-qa.bj.bcebos.com/fullchain_ce_test/model_download_link/tipc_models_url_03_14.txt
dir=$(pwd)
echo ">>> Donwnload model ..."
cd Models
cat ../tipc_models_url_03_14.txt | while read line
do
    wget -q $line
    tar -xf *tgz
    if [ $? -eq 0 ]; then
        cd norm_train_gpus_0,1_autocast_null_upload
        whole_name=$(ls *txt)
        echo ${whole_name%%_train_infer*}
        model_name=${whole_name%%_train_infer*}
        cd ${dir}/Models
        mv norm_train_gpus_0,1_autocast_null_upload ${model_name}
        rm -rf *tgz
    else
        echo "${$line} decompression failed"
    fi
done
cd ${dir}
echo ">>> Generate yaml configuration file ..."
bash prepare_config.sh

echo ">>> Run onnx_convert and converted model diff checker ..."
bash onnx_convert.sh

echo ">>> Run inference_benchmark ..."
bash inference_benchmark.sh

echo ">>> generate tipc_benchmark_excel.xlsx..."
python result2xlsx.py

echo ">>> Tipc benchmark done"
