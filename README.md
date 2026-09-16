# FedOSP：多中心糖网分级的序数-风格原型联邦学习

> **Ordinal-Style Prototype Federated Learning for Multi-Center Diabetic Retinopathy Grading**
>
> 冻结的 RETFound ViT-L 骨干 + LoRA，每轮每个 client 只上传 **2.31 MB**（实测），
> 相比全量同步降低约 **520 倍**通信量。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **CI 还没启用**：配置在 [`ci/github-workflow-ci.yml`](ci/github-workflow-ci.yml)，
> 挪到 `.github/workflows/ci.yml` 即生效。一步操作，见 [`ci/README.md`](ci/README.md)。
> （GitHub 不允许缺 `workflow` scope 的 token 写 `.github/workflows/`，所以首次上传放在了 `ci/`。）

---

## 最快上手：四条命令

**这四条按顺序跑，前三条都不需要真实数据、不需要 GPU。**

```bash
# 1) 装环境
pip install -r requirements.txt

# 2) 理论自检：5 秒，纯 CPU，验证两个创新点的数学命题是否成立
python scripts/verify_theory.py

# 3) 冒烟测试：不需要任何真实数据，5 分钟确认代码是通的
bash scripts/run_all.sh --smoke

# 3b) ★ 强烈建议：用仿真数据跑通**真实数据路径**（--smoke 走的是合成数据，
#     绕过了 FundusDataset、路径拼接、目录布局解析和未见中心评估）
python scripts/make_fixture.py --out data/fixture
python -m fedosp.data.build_manifest --data-root data/fixture --out data/fixture/manifest.csv
python -m fedosp.data.preprocess --manifest data/fixture/manifest.csv \
    --cache-dir data/fixture/cache --out data/fixture/manifest_cached.csv --short-side 64
python -m fedosp.run_fed --manifest data/fixture/manifest_cached.csv \
    --strategy fedosp --backbone debug_vit --img-size 64 --rounds 2 \
    --batch-size 4 --min-steps 2 --max-steps 3 --out runs/fixture

# 4) 数据齐了之后，一条命令跑完全部实验 + 出表 + 出图
bash scripts/run_all.sh --gpus 0,1,2,3
```

> 第 2 步值得先跑：它在不花任何 GPU 机时、不等数据审批的前提下，检验"精度加权
> 聚合"和"序数原型几何"这两个命题在理想条件下是否成立，并输出真实实验的**定量
> 预测**作为验收标准。见下节。

> **第 3b 步为什么不能省。** `--dry-run` / `--smoke` 用的是内存里造的合成数据集，
> **完全不经过 `FundusDataset`**。本项目真的因此踩过坑：真实数据路径上有两个必崩的
> bug（`mp.Value` 和 DataLoader 迭代器都无法 `deepcopy`），导致
> **Messidor-2 未见中心评估从未被执行过** —— 而它是论文的第二个主指标。
> 当时 55 个测试全绿。
>
> `scripts/make_fixture.py` 造的是**目录布局、标签文件格式、文件名规则全部与真实
> 数据一致**的迷你数据集（五个数据集共 196 张假图，画了圆形视野以走真实裁剪分支）。
> 它能在你花几小时下载 35 GB 之前，就把目录名猜错、路径拼接错这类问题暴露出来。

### 第 4 条就是"一键跑完所有实验"

`scripts/run_all.sh` 是唯一的入口，它把整条流水线串起来，依次完成：

```
环境自检 → 建 manifest → 图像预处理 → 文献锚点核对（关卡）
  → 74 个配置 × 多 seed（12 基线 + B17 交叉组 + 主方法 + 消融 + 标签效率 + 鲁棒 + 骨干）
  → 汇总论文表格 T1–T7 + 插图 F2–F7
```

| 你的处境 | 用哪条命令 | 耗时 |
|---|---|---|
| 有 4 张 A100 | `bash scripts/run_all.sh --gpus 0,1,2,3` | 2–2.5 天 |
| 有 1 张 A100 | `bash scripts/run_all.sh --gpus 0 --jobs-per-gpu 2` | 8–10 天 |
| **只有一台 Mac / 没有 GPU** | `bash scripts/run_all.sh --pilot` | **约 2 天**（降规模，见下） |
| 只想确认代码是通的 | `bash scripts/run_all.sh --smoke` | 5 分钟 |
| 只想看命令展开对不对 | `bash scripts/run_all.sh --dry-run` | 几秒 |

三个让它好用的性质：

- **断点续跑**：中断后重跑**同一条命令**即可，已有 `result.json` 的配置自动跳过（`--force` 可强制覆盖）。
- **单个配置崩了不影响全局**：自动重试 `--retries` 次，仍失败则记入 `logs/failed.txt` 并继续跑后面的。
- **加新实验不用改代码**：往 `configs/experiment_matrix.csv` 加一行，开关写进 `extra_args` 列即可。

只跑其中一段用 `--stage`（可选值见 `bash scripts/run_all.sh --help`）：

```bash
bash scripts/run_all.sh --stage sanity     # 只跑文献锚点核对
bash scripts/run_all.sh --stage main       # 只跑主对比
bash scripts/run_all.sh --stage analyze    # 只重新出表出图（不重跑实验）
```

> ⚠️ **但在第 4 步之前，请先花 5 分钟办两件有等待期的事**，详见下一节。

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
git clone https://github.com/Eternity1212/flosp.git && cd flosp
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

- `tables/` —— T1–T7，每张表三种格式（`.md` 看、`.csv` 加工、`.tex` 直接贴论文）
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

## 没有 GPU？本机 pilot（`configs/pilot_local.csv`）

**先说清楚能做什么、不能做什么。** 本机实测（Apple Silicon，12 核 / 36 GB，MPS）：

| 设备 / 骨干 | 吞吐（fwd+bwd, 224） |
|---|---:|
| MPS, ViT-Large | **11.4 图/s**（比 A100 慢约 11×） |
| CPU, ViT-Large | 3.5 图/s |
| MPS, ViT-Base | 32.1 图/s |
| MPS, ViT-Small | 84.0 图/s |

显存不是瓶颈（ViT-L @ batch 32 不 OOM），**时间才是**：

| 范围 | 本机墙钟 |
|---|---:|
| 单次正式 run（ViT-L / 100 轮 / 全量） | 36 小时 |
| 全部 74 行正式矩阵（169 run） | **256 天** |
| 仅 P0 且单 seed（42 run） | 64 天 |
| **pilot 矩阵（13 行 / 20 run）** | **约 2.1 天** |

pilot 回答的是**方向性**问题（C1、C2 的符号与量级），**不产出可发表数字**。

```bash
# 零成本前置检查：只看 n_eff 是否≈1.90（2 轮，0.2 h）。★ 先跑这个
python -m fedosp.run_fed --strategy fedavg --backbone vit_base_patch16_224 \
    --imagenet-pretrained --train-fraction 0.2 --rounds 2 --out runs/p_neff_check

# 跑整个 pilot 矩阵
bash scripts/run_all.sh --matrix configs/pilot_local.csv --seeds 0
```

### ★ pilot 降规模必须按比例，不能用统一上限

C2（精度加权聚合）的收益**完全来自客户端规模不平衡**。用统一上限降规模会把联邦
变成近似等规模，从而**人为消掉 C2 的全部改进空间** —— pilot 于是得出"C2 没用"，
但这个结论只是降规模方式的产物：

| 降规模方式 | 客户端规模 | $n_\text{eff}$ | 后果 |
|---|---|---:|---|
| 全量（正式） | 24600/6260/2560/372 | **1.75** | — |
| ❌ `--max-train-per-client 2000` | 2000/2000/2000/372 | **3.34** | C2 改进空间被消掉 |
| ✅ `--train-fraction 0.2` | 4920/1253/512/372 | **1.90** | 结论可迁移 |

所以用 `--train-fraction 0.2 --min-train-per-client 500`：按同一比例抽样，
且 ≤500 张的 client（IDRiD 只有 372 张）整体保全 —— 按比例抽会把它的 grade-1
抽没（全院仅 20 张，20% 只剩 4 张），而它只占 6% 机时，削它没有收益。
`val`/`test` 一律不降规模，保证指标与正式实验可比。
这条纪律有测试守护（含统一截断的反例对照），不靠记忆维护。

### `provenance.tier`：防止 pilot 结果被当成正式结果引用

pilot 的 `result.json` 在**格式上与正式结果完全一样**。`--stage main` 那道闸门只拦
"随机骨干"，拦不住"ViT-Base 跑 30 轮"。所以每个 run 都会自判等级：

```bash
python -c "import json;p=json.load(open('runs/xxx/result.json'))['provenance'];\
print(p['tier']);[print(' -',v) for v in p['tier_violations']]"
# main  → 可引用
# pilot → 不可引用，并逐条列出差在哪（骨干/分辨率/轮数/权重来源/是否降规模）
```

`tier="main"` 要求全部满足：ViT-Large、224、≥100 轮、权重来源 `retfound:`、
`stage=main`、未降规模、非 dry-run。任何一项不满足即 `pilot`，并打 WARNING。

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
│   │   └── prototypes.py        原型库（EMA 更新 + 抽样方差追踪）与服务器端聚合
│   ├── fed/
│   │   ├── strategies.py        13 个联邦策略（B3–B16 + FedOSP）
│   │   ├── client.py            本地训练、原型更新、SCAFFOLD control variate
│   │   ├── diagnostics.py       ★ 随机效应方差分解（DerSimonian-Laird）与有效客户端数
│   │   └── flower_adapter.py    可选：接 Flower 跑真实多进程
│   ├── losses.py                6 项损失 + 4 种序数范式 + evidential（FedUAA 复现用）
│   ├── metrics.py               QWK / worst-client / macro-over-client / ECE / 锚点核对
│   ├── stats.py                 DeLong / Wilcoxon / 配对 bootstrap / Holm-Bonferroni
│   ├── run_fed.py               联邦实验入口
│   └── run_central.py           集中式基线与锚点核对
├── scripts/
│   ├── run_all.sh               ★ 一键入口
│   ├── scheduler.py             多 GPU 任务池
│   ├── download_data.sh         数据获取助手
│   ├── verify_theory.py         ★ 合成验证两个理论命题（纯 CPU，5 秒）
│   ├── aggregate_results.py     生成 T1–T7
│   └── make_figures.py          生成 F2–F7
├── configs/
│   ├── default.yaml             默认超参
│   └── experiment_matrix.csv    ★ 74 个配置，加实验只改这里
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
| **精度加权原型聚合** | **P3** | `--proto-agg sample` / `client_equal` | `abl_proto_sample` |
| 本地步数均衡 | P3 | `--steps-rule equal` | `abl_equalsteps` |
| 本地 LayerNorm | 4.7 | `--no-personal-ln` | `abl_globalln` |

```bash
# 单独跑一个消融
python -m fedosp.run_fed --strategy fedosp --no-fsr \
    --exp-id abl_nofsr --seed 0 --out runs/abl_nofsr_seed0
```

---

## 基线：13 个联邦方法 + 4 种序数范式

每个基线都跑在**同一个 RETFound-LoRA 骨干、同一份 manifest、同一套本地步数规则**上，
所以主表的差异只能归因到机制本身。`--aux-reg` 对全部 13 个策略同等生效。

| 策略 | 年份 | 改哪一环 | 为什么必须有它 | 额外开销 |
|---|---|---|---|---|
| `fedavg` `fedprox` `fedbn` `fedper` `scaffold` | 17–21 | 经典 | 通用参照 | — |
| `fedproto` | 2022 | 聚合 | 原型联邦、按样本加权，C2 的直接对手 | — |
| `feduaa` | 2023 | 聚合权重 | 眼科联邦的领域内方法 | — |
| **`moon`** | 2021 | 本地损失 | C1 的同族最强对手（通用表征对齐） | **计算 1.7x**（每步 3 次前向） |
| **`fedala`** | 2023 | **下发** | C2 的同族对手：启发式元素级 vs 闭式最优 | 每轮 +5 次迭代学 W |
| **`qfedavg`** | 2020 | 聚合公式 | `n_eff` 的公平性叙事必须有公平性基线 | — |
| **`ditto`** | 2021 | 个性化 | 个性化这条线的标准做法 | **计算 2x**（w 与 v 各训一遍） |
| **`feddg`** | 2021 | 数据 | FSR 的直接对手 | **外传 23 MB 幅度谱** |

```bash
python -m fedosp.run_fed --strategy moon --moon-mu 1.0      # B12
python -m fedosp.run_fed --strategy qfedavg --q 1.0         # B14
python -m fedosp.run_fed --strategy ditto --ditto-lambda 0.1  # B15
```

### 序数范式（`--ord-type`，B17 交叉组）

`binomial` 与 `ordinal_encoding` **正是 Corbetta MIDL'25 用的两种**，实现它们同时
建立了与 MIDL'25 的可比性。`coral` / `ordinal_encoding` 会自动把输出头换成 K−1 维，
并在边界处用 `losses.ordinal_logits_to_probs` 转回 5 类概率 —— 下游所有指标
（QWK / ECE / DeLong / T6 / T7）完全复用，不需要任何特殊处理。

| `--ord-type` | 输出头 | 机制 | 注意 |
|---|---|---|---|
| `none` | K | 纯 CB-CE | 交叉表的左上角基准 |
| `emd` | K | 累积分布的平方距离 | 本文默认 |
| `binomial` | K | 以真值为中心的二项软标签 | ⚠ 见下 |
| `ordinal_encoding` | K−1 | K−1 个**独立**阈值 | 阈值可能自相矛盾 |
| `coral` | K−1 | K−1 个阈值**共享权重** | 秩单调性由构造保证 |

两条实现上必须知道的性质：

1. **`binomial` 内在压制置信度**。软标签 CE 的最优点在 $\hat p=q$ 而非 one-hot
   （$y{=}2$ 时最优损失 $=H(q)=1.4075$，而"完美自信"的 peak@2 损失是 6.25）。
   后果是它在 **ECE 上天然占便宜**，拿它跟普通 CE 比校准是不公平的，必须同时看
   QWK/MAE 才能判断是真校准好还是只是不自信。argmax 不受影响，QWK/accuracy 照常可比。
2. **阈值式范式下 CB-CE 关闭**。K−1 列上算 K 类 CE 是错的（不会报错，但类别语义
   完全错位），所以 `coral`/`ordinal_encoding` 的分类损失全部由阈值 BCE 承担，
   类别不平衡改由阈值重要性权重承担。

---

## 理论自检：`scripts/verify_theory.py`

两个创新点都是**数学命题**，不是"跑跑看"的经验技巧。既然是数学命题，就该在花掉
任何 GPU 机时之前先在完全可控的合成数据上检验 —— 不成立就立刻改设计，成立则拿到
一条定量预测当验收标准（真实结果偏离预测时，能区分"理论错了"和"实现有 bug"）。

```bash
python scripts/verify_theory.py                # 打印结论，退出码 0 = 两个命题都通过
python scripts/verify_theory.py --figures out/ # 额外出 5 联图
```

**结论（2026-09-15）：**

| 命题 | 结论 | 依据 |
|---|---|---|
| **精度加权聚合最优** | **无条件通过** | 全 $\tau^2$ 范围内是 MSE 下包络；两个极限特例精确成立（误差 1e−18 / 1e−14）；实测与理论式吻合 <5%；DL 估计相对误差 <15% |
| 序数几何优于均匀几何 | **有条件通过** | 远端误判必降；但净收益需"基线远端误判率 > ~4.5%"，低噪声区是净亏；效应量 QWK +0.02~0.04 |

**精度加权为什么有意思**：把原型聚合写成随机效应模型 $p_k=\mu+b_k+e_k$ 后，最优权重
有闭式解 $w_k^\star\propto 1/(\tau^2+v_k)$，而**两种现有做法恰好是它的极限特例**：

| 条件 | 退化为 | 对应方法 |
|---|---|---|
| $\tau^2=0$（无域偏移） | $w_k\propto n_k$ | FedProto（按样本量） |
| $\tau^2\gg v_k$（域偏移主导） | $w_k\to 1/K$ | client 等权 |

于是 A6 消融从"三个拍脑袋选项的横向比较"变成"沿 $\tau^2$ 一条理论曲线的扫描"：

```bash
# 用 DL 自动估 tau^2（默认）
python -m fedosp.run_fed --strategy fedosp --proto-agg precision
# 固定 tau^2 做扫描，验证退化行为
python -m fedosp.run_fed --strategy fedosp --proto-agg precision --tau2-override 1e-4
```

**有效客户端数 $n_{\text{eff}}=1/\sum_k w_k^2$** 逐轮写进 `result.json` 的 `history`，
对**所有策略**都记录（不只 FedOSP），因此是个跨方法诊断量：

| 配置 | $n_{\text{eff}}$(权重) | $n_{\text{eff}}$(复合，含本地步数) |
|---|---|---|
| 按样本量 + 按 epoch（朴素 FedAvg 默认） | 1.75 | **1.15** |
| sqrt 权重 + sqrt 步数 | 2.77 | 1.76 |
| client 等权 + 等步数（上界） | 4.00 | 4.00 |

> 4 家医院的联邦，用标准配方训练，统计上的有效客户端数只有 **1.15/4** —— 几乎
> 等于只训了最大的那家。这是"为什么必须重新设计聚合权重"最直接的证据。

**C1 的四条验收标准**（缺一条就不能声称序数几何生效）：

| | 预测 | 若不满足说明什么 |
|---|---|---|
| P1 | 远端误判率(T7)下降约 9% | 序数几何没真正生效，原型可能退化回单形 |
| P2 | accuracy **略降**（约 −0.012） | 若大涨，收益来自别处，不能归因给 C1 |
| P3 | QWK 改善约 +0.024 | — |
| P4 | T6 ρ 从 ~0 跃升到 >0.9 | 几何没被改造 |

> ⚠ **T6 高不等于 C1 奏效**：合成实验显示序数结构在预算占比 20% 时 T6 ρ 就已经
> 到 0.974，而 QWK/accuracy 的权衡还在继续。T6 只证明几何被改造了，不证明改造有
> 收益。所以上面四条必须一起看。

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
- 全部比较统一做 **Holm-Bonferroni** 校正 —— 一次比 12 个基线，
  不校正的话至少一次假阳性的概率是 46%

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
