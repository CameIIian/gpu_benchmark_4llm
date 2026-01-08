# README
gpuの推論/学習性能を測定するため環境整備用

## 1. 推論
### 1.1. nvidia / CPUでの計測
projectのクローン
```
git clone https://github.com/CameIIian/gpu_benchmark_4llm
```

package/Cpythonのインストール
```
cd gpu_benchmark_4llm/infer/bench_nvidia/
uv sync
```

実行
```
uv run main.py
```

### 1.2. radeonでの計測
#### docker+uv version
ROCmが利用可能なdockerバージョンを実行\
参考: https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html#using-wheels-package
```
docker pull rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1
docker run -it     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     --device=/dev/kfd     --device=/dev/dri     --group-add video     --ipc=host     --shm-size 8G     rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1
```

projectのクローン
```
git clone https://github.com/CameIIian/gpu_benchmark_4llm
```

package/Cpythonのインストール
```
cd gpu_benchmark_4llm/infer/bench_radeon/
uv sync
```

実行
```
uv run main.py
```

**エラーへの対処**
1. パッケージが取得できない場合\
pyproject.tomlを以下のように書き換える\
先に``pytorch-triton-rocm``をaddする\
``torch unsloth transformers``をaddする\
  参考: https://qiita.com/tsuchm/items/d0cff0c53f3ffb690901
```
[project]
name = "samplecodes"
version = "0.1.0"
description = "Sample codes to use ROCm"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
]

[[tool.uv.index]]
name = "pytorch-src"
url = "https://download.pytorch.org/whl/rocm6.4"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-src" }
pytorch-triton-rocm = { index = "pytorch-src" }
```

2. Exception: operator torchvision::nms does not exist\
 不要なので全てコメントアウトしていく

<details>
<summary>ネイティブのpython(3.10)を使うと起きるエラーへの対処</summary>

1. /opt/venv/lib/python3.10/site-packages/unsloth/device_type.py \
 NameError: name 'inspect' is not defined \
 -> device_type.pyにimport文を追加すれば動作する

2. module torch has no attribute no_gad\
 修正方法が基本的にtorchのアップデートなので、利用しないようにする
</details>


#### uv_only version
wip

## 2. 学習
利用したデータセット
> https://github.com/shi3z/alpaca_ja/blob/main/alpaca_cleaned_ja.json

### 2.1. nvidia / CPUでの計測
projectのクローン
```
git clone https://github.com/CameIIian/gpu_benchmark_4llm
```

package/Cpythonのインストール
```
cd gpu_benchmark_4llm/learning/bench_nvidia/
uv sync
```

実行
```
uv run LoRA.py
```
```
uv run Full-FT.py
```


### 2.2. radeonでの計測
#### docker+uv version
ROCmが利用可能なdockerバージョンを実行\
参考: https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html#using-wheels-package
```
docker pull rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1
docker run -it     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     --device=/dev/kfd     --device=/dev/dri     --group-add video     --ipc=host     --shm-size 8G     rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1
```

projectのクローン
```
git clone https://github.com/CameIIian/gpu_benchmark_4llm
```

package/Cpythonのインストール
```
cd gpu_benchmark_4llm/learning/bench_radeon/
uv sync
```

実行
```
uv run LoRA.py
```
```
uv run Full-FT.py
```

**エラーへの対処**
1. パッケージが取得できない場合\
pyproject.tomlを以下のように書き換える\
先に``pytorch-triton-rocm``をaddする\
``torch psutil unsloth datasets trl transformers setuptools``をaddする\
  参考: https://qiita.com/tsuchm/items/d0cff0c53f3ffb690901
```
[project]
name = "samplecodes"
version = "0.1.0"
description = "Sample codes to use ROCm"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
]

[[tool.uv.index]]
name = "pytorch-src"
url = "https://download.pytorch.org/whl/rocm6.4"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-src" }
pytorch-triton-rocm = { index = "pytorch-src" }
```

2. Exception: operator torchvision::nms does not exist\
 不要なので全てコメントアウトしていく

#### uv_only version
wip

## 3. 結果
### 3.1. 実験環境
| name | cpu | mem | gpu |
| --- | --- | --- | --- |
| R1 | AMD Ryzen Threadripper 7960X | 128GB | Radeon AI PRO 9700 32GB |
| N1 | AMD Ryzen Threadripper 7960X | 128GB | NVIDIA RTX 5090 32GB |
| N2 | Intel(R) Xeon(R) W-1290P | 64GB | NVIDIA RTX A6000 48GB |

<br/>

Model1: gpt-oss:20b (4bit量子化ON)
> https://huggingface.co/unsloth/gpt-oss-20b

Model2: Ministral-3-3B (予定)
> https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF

### 3.2. VRAM使用量の確認方法
GPUメモリの使用量はsmiツール等で確認\
Nvidia
```
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Radeon
```
rocm-smi --showmemuse | grep "VRAM"
```

### 3.3. 実験結果
#### 推論
TokenPerSeconds(TPS): 5種類の推論を3セット行ったときの平均\
mem(preload): 読込時のVRAMの最大使用量\
mem(infer): 推論時のVRAM最大使用量

| name | TPS | mem(preload) | mem(infer) | 
| --- | --- | --- | --- |
| R1 | 7.09 | 16GB | 16GB |
| N1 | - | -GB | -GB |
| N2 | 19.29 | 20GB | 12.5GB |

#### 学習
time: 学習終了までの時間\
mem: 学習時のVRAM最大使用量

##### gpt-oss:20b (LoRA, epoc = 60)
| name | time | mem | 
| --- | --- | --- |
| R1 | 1244.50s | 30GB |
| N1 | -s | -GB | 
| N2 |  800.41s | 21GB | 

##### Ministral-3-3B (FullFT, epoc = 1)
| name | time | mem | 
| --- | --- | --- |
| R1 | -s | -GB |
| N1 | -s | -GB | 
| N2 | -s | -GB | 
