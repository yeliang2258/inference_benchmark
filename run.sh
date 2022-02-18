echo ">>> Run onnx_convert and converted model diff checker ..."
bash onnx_convert.sh
echo ">>> Run inference_benchmark ..."
bash inference_benchmark.sh