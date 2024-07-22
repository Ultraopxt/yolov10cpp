<h1 align="center">Yolo V10 cpp ONNX</h1>


## Notes

Use the yolov10cpp given in the README(https://github.com/THU-MIG/yolov10) ,will get incorrect results as follows:

<p align="center">
  <img src="figures/ori.png" width=48%>
</p>


The cause of the error is that the pre-processing part does not use letterbox, but only uses a simple resize. Therefore, the corresponding post-processing part is also wrong.

After modification, the correct result is as follows:
<p align="center">
  <img src="figures/fixed.png" width=48%>
</p>


## Prepare the code 


1. Download de model you want 

  
  - yolov10n
  - yolov10s
  - yolov10m
  - yolov10b
  - yolov10l
  - yolov10x


```bash
    python download_model.py  --model {MODEL_SELECTED}
```

## Install packages

```
    conda create -n yolov10 python=3.9
    conda activate yolov10

    git clone https://github.com/THU-MIG/yolov10
    cd yolov10

    pip install -r requirements.txt
    pip install -e .

    cd ..
```

## Convert model

```
    yolo export model=yolov10n.pt format=onnx
```
## Dependencies 

1. ffmpeg
2. Opnecv
3. onnxruntime


- MacOs
```
    brew install ffmpeg 
    brew install opencv
    brew install onnxruntime
```

- Ubuntu: Unfortunately, onnx runtime is no available using native apt-get

You can use python
```
sudo apt-get update
sudo apt-get install python3-pip
pip3 install onnxruntime
```

dotnet 
```
dotnet add package Microsoft.ML.OnnxRuntime

```


## How to run this code 


1. Using Cmake, Recommended

```
    mkdir build
    cd build
    cmake ..
    make
```


2. Run the following command 

> static images

```
    ./yolov10_cpp [MODEL_PATH] [IMAGE_PATH]
```

> realtime 

```
    ./yolov10_cpp_video [MODEL_PATH] [SOURCE]
```

## Future plans

1. Modularize the components.
2. Make a example to video real time.
3. Support Cuda. ?


## Reference 

[1] Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., & Ding, G. (2024). YOLOv10: Real-Time End-to-End Object Detection. arXiv [Cs.CV]. Retrieved from http://arxiv.org/abs/2405.14458

[2] yolov10cpp Jose Sarmiento  https://github.com/DanielSarmiento04/yolov10cpp