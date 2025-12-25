# README
gpuの推論/学習性能を測定するため環境整備用

## 1. 準備
### 1.1. nvidia / CPUでの計測
projectのクローン
```
git clone https://github.com/CameIIian/gpu_benchmark_4llm
```

package/Cpythonのインストール
```
cd gpu_benchmark_4llm/bench_nvidia/
uv sync
```

実行
```
uv run main.py
```

### 1.2. radeonでの計測
#### docker+uv version
ROCmが利用可能なdockerバージョンを実行
```
docker pull rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1
docker run -it     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     --device=/dev/kfd     --device=/dev/dri     --group-add video     --ipc=host     --shm-size 8G     rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1
```
参考: https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html#using-wheels-package

以降は **1.1.** と基本的に同様

**エラーへの対処**
1. パッケージが取得できない場合\
 1つずつadd していってください\
  参考: https://qiita.com/tsuchm/items/d0cff0c53f3ffb690901

2. /opt/venv/lib/python3.10/site-packages/unsloth/device_type.py \
 NameError: name 'inspect' is not defined \
 -> device_type.pyにimport文を追加すれば動作する

3. module torch has no attribute no_gad\
 修正方法が基本的にtorchのアップデートなので、利用しないようにする

4. Exception: operator torchvision::nms does not exist\
 不要なので全てコメントアウトしていく

#### uv_only version
wip

### 1.2. VRAM使用量の確認方法
GPUメモリの使用量はsmiツール等で確認\
Nvidia
```
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Radeon
```
rocm-smi --showmemuse | grep "VRAM"
```

## 2. 結果
### 2.1. 実験環境
| name | cpu | mem | gpu |
| --- | --- | --- | --- |
| R1 | AMD Ryzen Threadripper 7960X | 128GB | Radeon AI PRO 9700 32GB |
| N1 | AMD Ryzen Threadripper 7960X | 128GB | NVIDIA RTX 5090 32GB |
| N2 | Intel(R) Xeon(R) W-1290P | 64GB | NVIDIA RTX A6000 48GB |

Model: gpt-oss:20b (4bit量子化ON)

### 2.2. 実験結果
#### 推論
TokenPerSeconds(TPS): 5種類の推論を3セット行ったときの平均
mem(preload): 読込時のVRAMの最大使用量
mem(infer): 推論時のVRAM最大使用量

| name | TPS | mem(preload) | mem(infer) | 
| --- | --- | --- | --- |
| R1 | 7.16 | 16GB | 16GB |
| N1 | - | -GB | -GB |
| N2 | 19.29 | 20GB | 12.5GB |

#### 学習

