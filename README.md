# Mamba-UIQM

基于 Mamba/Restormer 的水下图像质量评价 (UIQM) 训练与测试代码。当前仓库包含了训练脚本、评测脚本、数据读取与部分模型实现（含本地的 `timm` 与 `basicsr` 代码）。

## 目录结构

```
Mamba-UIQM/
├─ train.py                 # 训练入口
├─ test.py                  # 测试/评估入口
├─ config.py                # 配置读取工具
├─ data/                    # 数据列表与数据集类
│  └─ UWIQA/                 # UWIQA 列表文件
├─ basicsr/                 # 模型与基础组件 
├─ timm/                    # 本地 timm 依赖
├─ checkpoint/              # 默认模型权重输出目录
└─ output/                  # 推理/评估输出
```

## 环境准备

建议环境：Python 3.8+、CUDA GPU。脚本中大量使用 `.cuda()`，默认不支持纯 CPU 运行。

关键依赖（需自行安装）：
- torch / torchvision
- numpy
- opencv-python
- scipy
- tqdm
- tensorboard
- einops
- kornia
- mamba-ssm（及其 triton 依赖）


## 数据准备

以 UWIQA 为例：

1) 准备图像目录（与 `train.py` 中的 `train_dis_path` / `val_dis_path` 对应）。

```
IQA-dataset/
└─ UWIQA/
   ├─ 0001.png
   ├─ 0002.png
   └─ ...
```

2) 准备标签列表（仓库已提供示例）：

`data/UWIQA/uwiqa_train.txt` / `data/UWIQA/uwiqa_test.txt`

格式要求：每行是 `文件名, 分数`，文件名后有逗号。

```
0001.png, 0.7000
0002.png, 0.5000
```

注意：`data/UWIQA/uwiqa.py` 会执行 `dis = dis[:-1]` 去掉末尾逗号。如果你使用的是不带逗号的列表，请删除该行或改写解析逻辑。

## 训练

1) 打开 `train.py`，修改配置字典中的关键字段：
- `train_dis_path` / `val_dis_path`：数据路径
- `uiqa_train_label` / `uiqa_val_txt_label`：训练与验证列表
- `batch_size` / `n_epoch` / `learning_rate` 等训练参数
- `model_name` / `ckpt_path` / `log_path`：模型与日志输出路径

2) 启动训练：

```
python train.py
```

训练日志会写入 `checkpoint/log/`，模型权重默认保存到 `checkpoint/<type_name>/<model_name>/`。

## 测试与评估

1) 打开 `test.py`，修改配置字典中的关键字段：
- `test_dis_path`：测试图像路径
- `uwiqa_train_label`：测试列表（脚本中沿用该字段名）
- `model_save_path`：待评估权重路径
- `valid_path`：评估输出目录

2) 启动评估：

```
python test.py
```

控制台会输出 SRCC / PLCC / KRCC / RMSE 指标。