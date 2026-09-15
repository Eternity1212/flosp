# 启用 CI（一步）

这个目录里的 `github-workflow-ci.yml` 就是本项目的 GitHub Actions 配置。
它本该放在 `.github/workflows/ci.yml`，但**首次上传时放在了这里**，原因如下。

## 为什么不直接放在 .github/workflows/

GitHub 拒绝通过缺少 `workflow` scope 的 OAuth token 创建或修改
`.github/workflows/` 下的任何文件，会报：

```
! [remote rejected] main -> main
  (refusing to allow an OAuth App to create or update workflow
   `.github/workflows/ci.yml` without `workflow` scope)
```

这是 GitHub 的安全设计：workflow 文件能在仓库里执行任意代码，
所以改动它需要比普通推送更高的权限。为了不动你的认证凭证，
首次上传把它放在 `ci/` 下，内容完整无损。

## 怎么启用

**方式一：命令行（推荐）**

```bash
# 1) 给本地 gh 补上 workflow 权限（会打开浏览器，一次性）
gh auth refresh -s workflow

# 2) 把文件挪回标准位置并推送
mkdir -p .github/workflows
git mv ci/github-workflow-ci.yml .github/workflows/ci.yml
git commit -m "启用 CI"
git push
```

**方式二：网页（不用配权限）**

网页编辑器用的是你的登录态而不是 OAuth token，所以不受这个限制：

1. 打开仓库 → **Add file** → **Create new file**
2. 文件名填 `.github/workflows/ci.yml`
3. 把 `ci/github-workflow-ci.yml` 的内容整段粘贴进去
4. Commit，然后删掉 `ci/github-workflow-ci.yml`

## 这套 CI 检查什么

**不碰真实数据**（数据集有许可限制且几十 GB），全部跑在合成数据 + 微型 ViT 上，
几分钟出结果。Python 3.9 与 3.11 两个版本并行：

| 检查项 | 拦什么 |
|---|---|
| `pytest tests/` | 55 个单测 |
| 理论自检 | 两个命题的合成验证 |
| 骨干兼容性 | ViT / SwinV2 / ResNet 三条结构路径；**本地 norm 不能泄漏进上传集合** |
| **13 个策略端到端** | 每个联邦策略都能跑完 |
| **无漏测策略断言** | 新增策略后忘记加进上面的循环 → 直接红（否则它永远不被覆盖，而 CI 依然是绿的） |
| 5 种序数范式 | none / emd / binomial / coral / ordinal_encoding |
| 联邦回路真的在学 | 合成数据 40 轮后 macro QWK 必须 >0.5，否则训练循环坏了 |
| 统计检验 | DeLong 的 AUC 必须与 sklearn 完全一致（1e-9） |
| **两个矩阵都能展开** | 正式矩阵 + pilot 矩阵 |
| Shell 脚本语法 | `bash -n` |
