# FedOSP：多中心糖网分级的序数-风格原型联邦学习

> **Ordinal-Style Prototype Federated Learning for Multi-Center Diabetic Retinopathy Grading**
>
> 冻结的 RETFound ViT-L 骨干 + LoRA，每轮每个 client 只上传 **2.31 MB**（实测），
> 相比全量同步降低约 **520 倍**通信量。

[![CI](https://github.com/USERNAME/fedosp/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/fedosp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 最快上手：三条命令

```bash
# 1) 装环境
pip install -r requirements.txt

# 2) 冒烟测试：不需要任何真实数据，5 分钟确认代码是通的
bash scripts/run_all.sh --smoke

# 3) 数据齐了之后，一条命令跑完全部实验 + 出表 + 出图
bash scripts/run_all.sh --gpus 0,1,2,3
```

第 3 步会依次做完：环境自检 → 建 manifest → 预处理 → 文献锚点核对 →
44 个配置 × 多 seed → 生成论文表格 T1–T6 与插图 F2–F7。
**中断了直接重跑同一条命令**，已完成的配置会自动跳过。

> ⚠️ **但在这之前，请先花 5 分钟办两件有等待期的事**，详见下一节。

---

## 第 0 天：先把两张申请表填掉

整套流程里只有两个环节需要人工审批，而它们都在关键路径上。**今天就去办，拖一天整个项目晚一天。**

| 要办的事 | 在哪办 | 等多久 |
|---|---|---|
| **Messidor-2 图像**（未见中心外测用） | <https://www.adcis.net/en/third-party/messidor2/> 填表 | **数天–数周** |
| **RETFound 权重**（gated model） | <https://huggingface.co/YukunZhou/RETFound_mae_natureCFP> 点 Agree | **1–3 天** |

其余四个数据集（EyePACS / APTOS / DDR / IDRiD）都能立刻下载。
等审批期间就用 `--smoke` 把代码跑通、把四个数据集下好、把 manifest 建起来。

完整的数据获取步骤见 **[DATA.md](DATA.md)**。

---

## 这个方法在解决什么问题

多中心糖网分级的联邦学习里有三个已有工作没有正面处理的问题：

| | 问题 | 现有做法为什么不够 | FedOSP 怎么做 |
|---|---|---|---|
| **P1** | 相机与光照造成的**风格差异在浅层累积**，聚合本身消不掉 | FedBN 只换 LayerNorm 统计量，没动特征本身 | **FSR**：浅层做频域幅度归一化、相位保留 |
| **P2** | DR 严重度**本质是有序的**（0<1<2<3<4），标准联邦目标当成无序类别 | 原型联邦只做类内聚拢，不约束类间顺序 | **双层原型 + 序数间隔**：让原型排成有序流形 |
| **P3** | **一个大中心主导聚合**，最小中心反而变差 | 按样本数加权时 EyePACS 占 84%，IDRiD 只占 1.5% | **client 等权原型聚合 + sqrt 本地步数** |

四个训练 client 的规模差是 **64:1**（EyePACS 24,588 vs IDRiD 372），
这个悬殊比例正是 P3 存在的根源。

---

## 方法结构

```
输入眼底图 (224x224)
        │
   ┌────▼─────────────────────────────────┐
   │ RETFound ViT-L/16 骨干（全程冻结）      │
   │  ├── block 0..5                       │
   │  │                                    │
   │  ├──► FSR 频域风格校准 ◄── 浅层风格原型   │  ← 解决 P1
   │  │    幅度归一化 / 相位保留               │
   │  │                                    │
   │  └── block 6..23（后 12 层注入 LoRA r=8）│
   └────┬─────────────────────────────────┘
        │
   深层特征 ──► 等级原型（序数间隔约束）  ← 解决 P2
        │
   分类头 ──► 5 类 DR 等级

上传给服务器（2.31 MB）：LoRA A/B + 分类头 + FSR 门控 + 双层原型
留在本地（不上传）：LayerNorm 仿射参数
服务器聚合：参数按 sqrt(n_k) 加权，**原型按 client 等权**  ← 解决 P3
```

实测参数量（timm 1.0.27，ViT-L/16 @224，LoRA r=8 打后 12 个 block）：

| 项 | 数值 |
|---|---|
| 总参数 | 303.90 M |
| 可训练参数 | **0.70 M（0.23%）** |
| 单轮上传 | **2.31 MB / client** |
| 100 轮 × 4 client 累计 | **约 0.9 GB**（全量微调是 480 GB） |
| 显存（batch 32） | 10–14 GB |

---

## 五步复现

### 第 1 步：环境

```bash
git clone https://github.com/USERNAME/fedosp.git && cd fedosp
pip install -r requirements.txt

bash scripts/run_all.sh --stage check    # 逐项核对依赖、GPU、数据、权重
```

### 第 2 步：数据

```bash
bash scripts/download_data.sh --all       # 自动下 Kaggle 的三项
bash scripts/download_data.sh --ddr --idrid   # 打印手动下载指引
bash scripts/download_data.sh --retfound  # 权重获批后再跑这条
bash scripts/download_data.sh --verify    # 校验完整性
```

### 第 3 步：manifest 与预处理（一次性，约 2–4 小时）

```bash
bash scripts/run_all.sh --stage data
```

这一步会强制执行四项数据完整性检查，任一不过直接报错退出：

- **EyePACS 患者级划分**：`10_left.jpeg` 与 `10_right.jpeg` 必须同 split（有反向测试验证守卫有效）
- **DDR ungradable 剔除**：标签 5 是"无法评级"不是第 6 个等级，约 1,151 张
- **IDRiD 官方划分**：直接读官方 csv，不做二次随机划分
- **Messidor-2 隔离**：split 全标 test，且不在训练 client 列表里

### 第 4 步：文献锚点核对（**这是关卡，别跳过**）

```bash
bash scripts/run_all.sh --stage sanity
```

单中心训练的结果要能对上文献量级（APTOS QWK ≈ 0.90、EyePACS ≈ 0.80）。
**对不上就说明数据管线有问题，此时往下跑联邦实验是在浪费几十个 GPU 小时。**
`run_all.sh` 在这里设了硬关卡，不过就直接停。

### 第 5 步：跑全套并出表出图

```bash
bash scripts/run_all.sh --gpus 0,1,2,3           # 全部阶段
bash scripts/run_all.sh --stage main --gpus 0,1  # 或分阶段跑
bash scripts/run_all.sh --stage analyze          # 只重新出表出图
```

产物：

- `tables/` —— T1–T6，每张表三种格式（`.md` 看、`.csv` 加工、`.tex` 直接贴论文）
- `figures/` —— F2–F7，pdf 矢量 + png 位图，字体已嵌入（投稿系统要求）
- `runs/<exp_id>_seed<n>/` —— 每次实验的 `result.json`、`best.pt`、`predictions.npz`
- `logs/` —— 每个任务一个日志；失败清单在 `logs/failed.txt`

---

## 一次跑完所有实验：调度器怎么工作

`run_all.sh` 内部调用 `scripts/scheduler.py`，它解决手工并行的三个痛点：

| 痛点 | 解法 |
|---|---|
| 断了不知道断在哪 | 每个任务一个日志，失败清单单独写 `logs/failed.txt` |
| 重跑会把已完成的再跑一遍 | 有 `result.json` 就跳过（`--force` 可覆盖） |
| 一个配置崩了后面全不跑 | 自动重试 `--retries` 次，仍失败则记录并继续 |

其他实用开关：

```bash
# 80GB 卡可以单卡塞两个实验
bash scripts/run_all.sh --gpus 0,1 --jobs-per-gpu 2

# 只跑 1 个 seed 快速看趋势
bash scripts/run_all.sh --stage main --seeds 0

# 只打印命令不执行，确认矩阵展开对不对
bash scripts/run_all.sh --dry-run
```

**加新实验不用改代码**：在 `configs/experiment_matrix.csv` 里加一行，
把开关写进 `extra_args` 列即可，调度器会自动展开成命令。

---

## 代码结构

```
fedosp/
├── fedosp/
│   ├── data/
│   │   ├── build_manifest.py    五个数据集 → 统一 CSV，含四项完整性检查
│   │   ├── preprocess.py        圆形裁剪 + 短边 512 缓存（不做跨中心风格归一化！）
│   │   └── dataset.py           数据集、增强、标签预算、本地步数与聚合权重
│   ├── models/
│   │   ├── backbones.py         骨干适配层：统一 ViT / SwinV2 / ResNet 的 stage 接口
│   │   ├── retfound_lora.py     主模型，含参数三分（上传 / 本地 / 冻结）
│   │   ├── fsr.py               频域风格校准，支持 token / NHWC / NCHW 三种排布
│   │   ├── lora.py              LoRALinear + LoRAConv2d（CNN 骨干对照要用）
│   │   └── prototypes.py        原型库（EMA 更新）与服务器端聚合
│   ├── fed/
│   │   ├── strategies.py        8 个联邦策略
│   │   ├── client.py            本地训练、原型更新、SCAFFOLD control variate
│   │   └── flower_adapter.py    可选：接 Flower 跑真实多进程
│   ├── losses.py                6 项损失 + evidential（FedUAA 复现用）
│   ├── metrics.py               QWK / worst-client / macro-over-client / ECE / 锚点核对
│   ├── stats.py                 DeLong / Wilcoxon / 配对 bootstrap / Holm-Bonferroni
│   ├── run_fed.py               联邦实验入口
│   └── run_central.py           集中式基线与锚点核对
├── scripts/
│   ├── run_all.sh               ★ 一键入口
│   ├── scheduler.py             多 GPU 任务池
│   ├── download_data.sh         数据获取助手
│   ├── aggregate_results.py     生成 T1–T6
│   └── make_figures.py          生成 F2–F7
├── configs/
│   ├── default.yaml             默认超参
│   └── experiment_matrix.csv    ★ 44 个配置，加实验只改这里
├── tests/test_pipeline.py       关键正确性测试
├── DATA.md                      ★ 数据获取详细步骤
└── README.md
```

---

## 组件与开关对照

每个创新点都有对应的开关，用来做消融：

| 组件 | 论文位置 | 关掉它 | 对应消融 |
|---|---|---|---|
| FSR 频域风格校准 | P1 | `--no-fsr` | `abl_nofsr` |
| 浅层风格原型 | P1 | `--no-shallow-proto` | `abl_noshallow` |
| 深层等级原型 | P2 | `--no-deep-proto` | `abl_nodeep` |
| 序数间隔约束 | P2 | `--ordinal-margin 0` | `abl_nomargin` |
| EMD 序数损失 | P2 | `--lambda-ord 0` | `abl_noord` |
| client 等权原型聚合 | P3 | `--proto-agg sample` | `abl_proto_sample` |
| 本地步数均衡 | P3 | `--steps-rule equal` | `abl_equalsteps` |
| 本地 LayerNorm | 4.7 | `--no-personal-ln` | `abl_globalln` |

```bash
# 单独跑一个消融
python -m fedosp.run_fed --strategy fedosp --no-fsr \
    --exp-id abl_nofsr --seed 0 --out runs/abl_nofsr_seed0
```

---

## 结果出来之后先看这三处

跑完 `--stage analyze` 后，**优先核对这三个数字**，它们决定文章的走向：

1. **`tables/t3.md` 里 `abl_nofsr` 的 ΔMacro** —— 若 > −0.5，说明 FSR 收益不足（P1 站不住）
2. **`tables/t3.md` 里 `abl_proto_sample` 的 ΔMacro** —— 这决定 P3 卖点是否成立
3. **`figures/F4` 的标题** —— 若显示 `NO ordinal structure`，说明序数间隔损失没起作用（P2 有问题）

前两项任一不成立时的退路已经提前写好了，见
`../眼底图像_高影响因子论文_2026-09-02/补充论文_问题与Idea_2026-09-07/11_可行性评估与风险清单.md`
第 5 节的三档方案。**最坏情况下这套代码和 44 个实验原样就是一篇 benchmark 论文**，
不需要额外实验 —— 这是整个方案最重要的抗风险设计。

---

## 关于统计检验的一个重要提醒

只有 4 个 client，**跨 client 的 Wilcoxon 符号秩检验双侧 p 值下界是 0.125，
永远不可能小于 0.05**，经 Holm 校正后更不可能。这不是 bug，是样本量的硬限制。

所以本项目的检验策略是：

- **主检验放在样本级**：每个 client 的测试样本上做配对 bootstrap（QWK）与 DeLong 检验（AUROC），
  n 是几百到几千，能达到显著
- **client 级只报方向一致性**（例如"4/4 个 client 全部提升"）作为辅助证据
- 全部比较统一做 **Holm-Bonferroni** 校正 —— 一次比 8 个基线，
  不校正的话至少一次假阳性的概率是 34%

这要求每次实验保存 per-sample 预测，`run_fed.py` 会自动写 `predictions.npz`。
`scripts/aggregate_results.py` 的 T6 就是基于它生成的。

---

## 常见坑

<details>
<summary><b>1. 不要做跨中心风格归一化</b></summary>

预处理里**故意不做** CLAHE、直方图匹配之类的跨中心归一化。
本文的方法就是在学怎么处理风格差异，先把差异抹平等于把要研究的现象删掉了，
FSR 也就无从验证。`preprocess.py` 里对此有明确注释。
</details>

<details>
<summary><b>2. 主表不能用样本加权平均</b></summary>

EyePACS 占了 84% 的样本，样本加权平均基本等于只看 EyePACS，
小中心的改善会被完全淹没。本项目所有汇总都用 **macro-over-client**（各 client 等权平均）
和 **worst-client**，早停也用 macro。样本加权的数字只放附录。
</details>

<details>
<summary><b>3. RETFound 权重加载失败会直接报错，不会静默继续</b></summary>

静默加载失败等于拿随机初始化的骨干跑完整套实验，那是最坏的结果。
`load_retfound_weights` 会检查 `patch_embed` / `blocks` 是否齐全，缺了就抛异常。
如果看到 `骨干是随机初始化的！` 这条警告，说明忘了传 `--pretrained`。
</details>

<details>
<summary><b>4. IDRiD 只有 372 张训练图，方差天然就大</b></summary>

3 个 seed 之间波动几个点是正常的，不要以为是 bug。
建议把这个高方差本身当作"小中心问题真实存在"的证据来写，配 F7 的 per-client 方差图。
</details>

<details>
<summary><b>5. FedUAA-style 是复现版，论文里必须声明</b></summary>

为了公平比较，B9 的骨干统一换成了 RETFound-LoRA，与原文不同。
论文里要明确写"统一骨干下的复现版"，**不要声称等同原文数字**。
代码里 `FedUAAStyle` 的 docstring 也写了这一点。
</details>

<details>
<summary><b>6. SwinV2 的 qkv 不能包 LoRA</b></summary>

timm 的 SwinV2 attention 里是 `F.linear(x, weight=self.qkv.weight, ...)` 直接读 `.weight`，
包成 `LoRALinear` 后会报 `'LoRALinear' object has no attribute 'weight'`。
`backbones.py` 里 SwinV2 改打 `attn.proj` 与 MLP 的 `fc1`/`fc2`，
可训练参数量与 ViT 方案同量级（1.14M vs 0.70M），A12 对照仍然公平。
</details>

<details>
<summary><b>7. ResNet 骨干的 FSR 频率分辨率高 16 倍</b></summary>

ViT@224 的 patch 网格是 14×14 = 196 个频率 bin，
ResNet50 在同样切点是 56×56 = **3136** 个 bin。
这不是 bug，反而是个有用的实验：如果 FSR 在 ResNet 上收益明显大于 ViT，
就直接证明了"频率分辨率是 FSR 有效性的关键因素"，这本身是一个可发表的发现。
</details>

---

## 引用

```bibtex
@article{fedosp2026,
  title   = {Ordinal-Style Prototype Federated Learning for Multi-Center
             Diabetic Retinopathy Grading},
  author  = {FIXME},
  journal = {FIXME},
  year    = {2026}
}
```

本项目基于 RETFound，请同时引用：

```bibtex
@article{zhou2023retfound,
  title   = {A foundation model for generalizable disease detection from retinal images},
  author  = {Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and others},
  journal = {Nature}, volume = {622}, number = {7981}, pages = {156--163}, year = {2023}
}
```

数据集引用见 [DATA.md](DATA.md) 第 9 节。

---

## 许可

代码是 [MIT](LICENSE)。**但数据集与预训练权重各有自己更严格的许可**
（RETFound 是 CC BY-NC 4.0 非商业，Messidor-2 禁止再分发），
MIT 不覆盖它们。本仓库不包含任何数据集图像或模型权重。
