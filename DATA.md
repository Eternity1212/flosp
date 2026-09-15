# 数据获取指南

五个公开眼底数据集 + RETFound 预训练权重的逐个获取步骤。

> ## ⚠️ 先读这一段：有两件事今天就要办
>
> 下面 7 项里有 **2 项需要人工审批、有等待期**，别的都能立刻下。
> 这两项拖一天，整个项目就晚一天：
>
> 1. **Messidor-2 图像** —— 去 [ADCIS 官网](https://www.adcis.net/en/third-party/messidor2/) 填表申请（**数天到数周**）
> 2. **RETFound 权重** —— 去 [HuggingFace](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP) 申请访问（**1–3 天**）
>
> 先把这两张表填掉，再回来下其他数据。等待期间可以用 `--smoke` 把代码跑通。

---

## 0. 总览

| # | 资源 | 大小 | 获取方式 | 等待期 | 用途 |
|---|---|---|---|---|---|
| 1 | EyePACS (Kaggle DR 2015) | ~35 GB | Kaggle CLI | 即时 | 训练 client（最大） |
| 2 | APTOS 2019 | ~10 GB | Kaggle CLI | 即时 | 训练 client |
| 3 | DDR | ~8 GB | GitHub / 网盘 | 即时 | 训练 client |
| 4 | IDRiD | ~500 MB | IEEE DataPort | 即时 | 训练 client（最小） |
| 5 | Messidor-2 图像 | ~3 GB | **ADCIS 填表申请** | **数天–数周** | **未见中心外测** |
| 6 | Messidor-2 DR 标签 | ~50 KB | Kaggle | 即时 | 配 #5 使用 |
| 7 | RETFound 权重 | ~1.2 GB | **HF gated 申请** | **1–3 天** | 骨干预训练 |

最终目录结构（`scripts/run_all.sh --stage check` 会逐项核对）：

```
data/
├── raw/
│   ├── eyepacs/         train/ test/ trainLabels.csv testLabels.csv
│   ├── aptos/           train_images/ train.csv
│   ├── ddr/             DR_grading/ (train/ valid/ test/ + 三个 .txt)
│   ├── idrid/           B_Disease_Grading/
│   └── messidor2/       IMAGES/ + messidor_data.csv
├── manifest.csv         ← build_manifest.py 生成
└── cache/               ← preprocess.py 生成（短边 512 的 jpg）
weights/
└── RETFound_mae_natureCFP.pth
```

---

## 1. 准备 Kaggle CLI（#1 #2 #6 都要用）

```bash
pip install kaggle

# 去 https://www.kaggle.com/settings/account → "Create New API Token"
# 会下载一个 kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json     # 权限不对 CLI 会拒绝启动
```

**竞赛数据必须先在网页上接受规则**，否则 CLI 会报 403。分别去这两个页面点一下
"Late Submission" 或 "Rules → I Understand and Accept"：

- <https://www.kaggle.com/c/diabetic-retinopathy-detection/rules>
- <https://www.kaggle.com/c/aptos2019-blindness-detection/rules>

---

## 2. EyePACS（35 GB，最大的一个）

```bash
cd data/raw && mkdir -p eyepacs && cd eyepacs
kaggle competitions download -c diabetic-retinopathy-detection

# 图像分成 5 个卷，需要先拼接再解压
cat train.zip.00* > train.zip && unzip -q train.zip && rm train.zip train.zip.00*
cat test.zip.00*  > test.zip  && unzip -q test.zip  && rm test.zip  test.zip.00*
unzip -q trainLabels.csv.zip && unzip -q retinopathy_solution.csv.zip
```

校验：`train/` 应有 **35,126** 张，`test/` 应有 **53,576** 张。

> **患者级划分的关键**：文件名形如 `10_left.jpeg` / `10_right.jpeg`，
> 前面的数字是患者 ID。**同一患者的左右眼必须落在同一个 split**，
> 否则会有数据泄漏、QWK 虚高。`build_manifest.py` 已经按 `_` 前的 ID 做患者级划分，
> 并且有测试用例故意注入泄漏来验证守卫有效。

---

## 3. APTOS 2019

```bash
cd data/raw && mkdir -p aptos && cd aptos
kaggle competitions download -c aptos2019-blindness-detection
unzip -q aptos2019-blindness-detection.zip && rm aptos2019-blindness-detection.zip
```

校验：`train_images/` 应有 **3,662** 张，`train.csv` 3,662 行。
测试集标签未公开，所以本文只用它的训练集（内部再划分 70/10/20）。

---

## 4. DDR

官方仓库：<https://github.com/nkicsl/DDR-dataset>

作者把数据放在 Google Drive / 百度网盘，**没有直接的 wget 链接**，需要手动下。
下载后按下面处理：

```bash
cd data/raw/ddr
# 若是分卷 zip（DDR-dataset.zip.001 ...），先合并
cat DDR-dataset.zip.0* > DDR-dataset.zip
unzip -q DDR-dataset.zip
# 本文只用 DR 分级子集
ls DR_grading/    # 应有 train/ valid/ test/ 与 train.txt valid.txt test.txt
```

校验：DR grading 子集共 **13,673** 张。

> **必须过滤 ungradable**：DDR 的标签里 **5 = ungradable（无法评级）**，不是第 6 个严重度等级。
> 把它当成一类会污染 QWK（QWK 对类别顺序敏感，凭空多一档会压低所有方法的分数）。
> 这类样本约 **1,151 张**。`build_manifest.py` 会自动剔除并在日志里报数量，
> 测试用例也会检查剔除是否生效。

---

## 5. IDRiD

<https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>

注册 IEEE 账号后即可下载（免费）。只需要 **B. Disease Grading** 部分。

```bash
cd data/raw/idrid && unzip -q B_Disease_Grading.zip
```

校验：train **413** 张、test **103** 张。

> **必须用官方划分**。IDRiD 是 grand challenge 数据集，文献报的数字都基于官方
> train/test 划分。自己重新随机划分会导致跟文献锚点对不上，
> 也无法与他人比较。`build_manifest.py` 直接读官方 csv，不做二次划分。

---

## 6. Messidor-2（未见中心外测，**图像与标签要分两处拿**）

### 6a. 图像：ADCIS 填表申请（有等待期）

1. 打开 <https://www.adcis.net/en/third-party/messidor2/>
2. 填写个人信息表单（姓名、单位、邮箱、用途），验证邮箱
3. 等审批邮件，拿到下载链接后取回图像（**共 1,748 张**）

```bash
cd data/raw/messidor2 && unzip -q 'Messidor-2*.zip' -d IMAGES/
```

> 许可：仅限研究与教育用途，禁止商业使用与再分发。
> 论文致谢里必须写：*Kindly provided by the Messidor program partners
> (see https://www.adcis.net/en/third-party/messidor/)*。

### 6b. DR 等级标签：Kaggle（即时）

**Messidor-2 原始发布不含 5 级 DR 标签**，通用的是 Google Brain 团队的裁定标签：

```bash
cd data/raw/messidor2
kaggle datasets download -d google-brain/messidor2-dr-grades
unzip -q messidor2-dr-grades.zip && rm messidor2-dr-grades.zip
# 得到 messidor_data.csv，含 image_id / adjudicated_dr_grade
```

校验：csv 里应有 **1,744** 条有效等级（少数图像未裁定）。

> **这个数据集在本项目里全程隔离**：不参与训练、不参与验证、不参与早停、不参与调参，
> 只在最终评估时用一次。`build_manifest.py` 会把它的 split 全部标成 `test`，
> 且 `run_fed.py` 的 client 列表里没有它。这比 FedUAA 的协议更严格。

### 6c. 万一批不下来怎么办

如果 ADCIS 超过 3 周没回，切换到 **leave-one-client-out** 协议：
轮流拿一个训练 client 当未见中心（4 折），完全用手头数据。
这个协议其实**更严格**，可能反而是加分项。见可行性文档 5.2 节。

---

## 7. RETFound 权重（HF gated，有等待期）

```bash
pip install -U huggingface_hub

# 1) 去 https://huggingface.co/YukunZhou/RETFound_mae_natureCFP 点 "Agree and access"
#    （需要登录 HF 账号，填一个简短表单，等 1-3 天审批）
# 2) 去 https://huggingface.co/settings/tokens 建一个 read token
hf auth login                  # 粘贴 token

# 3) 下载
mkdir -p weights
hf download YukunZhou/RETFound_mae_natureCFP --local-dir weights/

export RETFOUND_CKPT="$PWD/weights/RETFound_mae_natureCFP.pth"
echo "export RETFOUND_CKPT=$RETFOUND_CKPT" >> ~/.zshrc   # bash 用户改 ~/.bashrc
```

> **命令改过名。** `huggingface_hub` 1.x 起 CLI 从 `huggingface-cli` 改为 **`hf`**，
> `login` 变成 `hf auth login`，并且移除了 `--local-dir-use-symlinks`。
> 装了却报"找不到命令"，通常是脚本目录不在 PATH：
> ```bash
> export PATH="$HOME/Library/Python/3.9/bin:$HOME/.local/bin:$PATH"
> ```
> `scripts/download_data.sh --retfound` 两种命令名都兼容，会自动选可用的那个。
>
> 没获批时它会明确报 `Access denied. This repository requires approval.`
> —— 这说明申请还没过，不是脚本坏了。

校验：约 **1.2 GB**，ViT-Large/16（~303 M 参数）。
加载时 `retfound_lora.py` 会检查 `patch_embed` / `blocks` 是否齐全，
不齐就直接报错而不是静默继续 —— 静默加载失败等于拿随机初始化的骨干跑完整套实验，
那是最坏的情况。

许可为 **CC BY-NC 4.0**（非商业）。论文需引用 Zhou et al., *Nature* 2023。

### 拿不到权重的临时替代

```bash
# 用 ImageNet 预训练的同尺寸 ViT-L 先把管线跑通
python -m fedosp.run_fed --strategy fedosp --rounds 5 --imagenet-pretrained
```

方法本身不依赖特定预训练（四个组件都是骨干无关的），但会丢掉"眼科基础模型"这个卖点，
文献锚点也对不上。只作过渡使用。

---

## 8. 一键校验

全部下完后：

```bash
bash scripts/download_data.sh --verify      # 逐项核对文件数与关键字段
bash scripts/run_all.sh --stage data        # 建 manifest + 预处理
```

`build_manifest.py` 会强制执行四项检查，任一不过直接报错退出：

| 检查 | 内容 |
|---|---|
| 患者级无泄漏 | EyePACS 同一患者 ID 不跨 split |
| DDR ungradable | 标签 5 已剔除，日志报出剔除数量 |
| IDRiD 官方划分 | 与官方 csv 完全一致，未做二次划分 |
| Messidor-2 隔离 | split 全为 test，且不在训练 client 列表里 |

---

## 9. 引用

用了这些数据就必须在论文里引用：

```bibtex
@misc{kaggle-dr-2015,
  title = {Diabetic Retinopathy Detection},
  howpublished = {Kaggle competition, EyePACS},
  year = {2015},
  url = {https://www.kaggle.com/c/diabetic-retinopathy-detection}
}
@misc{aptos2019,
  title = {APTOS 2019 Blindness Detection},
  howpublished = {Kaggle competition},
  year = {2019},
  url = {https://www.kaggle.com/c/aptos2019-blindness-detection}
}
@article{li2019ddr,
  title = {Diagnostic assessment of deep learning algorithms for diabetic
           retinopathy screening},
  author = {Li, Tao and Gao, Yingqi and Wang, Kai and Guo, Song and
            Liu, Hanruo and Kang, Hong},
  journal = {Information Sciences}, volume = {501}, pages = {511--522}, year = {2019}
}
@article{porwal2018idrid,
  title = {Indian Diabetic Retinopathy Image Dataset (IDRiD)},
  author = {Porwal, Prasanna and Pachade, Samiksha and Kamble, Ravi and others},
  journal = {Data}, volume = {3}, number = {3}, pages = {25}, year = {2018}
}
@article{decenciere2014messidor,
  title = {Feedback on a publicly distributed image database: the Messidor database},
  author = {Decenci{\`e}re, Etienne and Zhang, Xiwei and Cazuguel, Guy and others},
  journal = {Image Analysis \& Stereology}, volume = {33}, number = {3},
  pages = {231--234}, year = {2014}
}
@article{krause2018grader,
  title = {Grader variability and the importance of reference standards for
           evaluating machine learning models for diabetic retinopathy},
  author = {Krause, Jonathan and Gulshan, Varun and Rahimy, Ehsan and others},
  journal = {Ophthalmology}, volume = {125}, number = {8}, pages = {1264--1272},
  year = {2018}, note = {Messidor-2 的裁定 DR 等级来源}
}
@article{zhou2023retfound,
  title = {A foundation model for generalizable disease detection from
           retinal images},
  author = {Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and others},
  journal = {Nature}, volume = {622}, number = {7981}, pages = {156--163}, year = {2023}
}
```
