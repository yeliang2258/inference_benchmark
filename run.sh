echo ">>> Donwnload model link url ..."
wget https://paddle-qa.bj.bcebos.com/fullchain_ce_test/model_download_link/tipc_models_url.txt

echo ">>> Donwnload model ..."
cat tipc_models_url.txt | while read line
do
    wget -q $line
done

echo ">>> decompression model ..."
mv *tgz Models
cd Models
for tgz_file in ./*tgz
do
    tar -xf ${tgz_file}
done
rm -rf *tgz
cd ..

echo ">>> Generate yaml configuration file ..."
bash prepare_config.sh

echo ">>> Run onnx_convert and converted model diff checker ..."
bash onnx_convert.sh

echo ">>> Run inference_benchmark ..."
bash inference_benchmark.sh

echo ">>> generate tipc_benchmark_excel.xlsx..."
python result2xlsx.py

echo ">>> Tipc benchmark done"
