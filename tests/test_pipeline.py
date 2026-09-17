"""骨架自检：不需要真实数据和 GPU，几十秒跑完。

每次改代码之后先跑这个::

    cd fedosp && python -m pytest tests -q
    # 没装 pytest 也可以：python tests/test_pipeline.py

覆盖的是最容易出错、一旦错了整篇论文作废的几件事：

1. EyePACS 患者级划分不能泄漏
2. 上传的参数里不能混进 LayerNorm
3. 原型 client 等权聚合确实没被大 client 主导
4. QWK 的方向和边界值正确
5. 六项损失都能回传梯度
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedosp.data.build_manifest import (  # noqa: E402
    MANIFEST_COLUMNS,
    stratified_group_split,
    verify_manifest,
)
from fedosp.data.dataset import aggregation_weights, apply_label_budget, local_steps  # noqa: E402
from fedosp.losses import (  # noqa: E402
    FedOSPLoss,
    LossWeights,
    edl_uncertainty,
    effective_number_weights,
    ordinal_prototype_loss,
    squared_emd_loss,
)
from fedosp.metrics import aggregate_over_clients, quadratic_weighted_kappa  # noqa: E402
from fedosp.models.prototypes import aggregate_prototypes  # noqa: E402
from fedosp.models.retfound_lora import build_model  # noqa: E402


def _fake_eyepacs(n_patients: int = 300) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    rows = []
    for p in range(n_patients):
        g = int(rng.choice(5, p=[0.735, 0.07, 0.15, 0.025, 0.02]))
        for side in ("left", "right"):
            rows.append(
                dict(image_id=f"{p}_{side}", client="eyepacs", dr_grade=g,
                     patient_id=str(p), source_split="official_train", path="x")
            )
    df = pd.DataFrame(rows)
    df["split"] = stratified_group_split(df, (0.7, 0.1, 0.2), 0)
    return df[MANIFEST_COLUMNS]


def test_patient_level_split_has_no_leakage():
    df = _fake_eyepacs()
    groups = {s: set(g["patient_id"]) for s, g in df.groupby("split")}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    # 同一患者的两只眼必须在同一个 split
    assert (df.groupby("patient_id")["split"].nunique() == 1).all()


def test_verify_manifest_catches_messidor_leak():
    df = _fake_eyepacs().copy()
    df["client"] = "messidor2"
    problems = verify_manifest(df)
    assert any("messidor2 全部为 test" in p for p in problems)


def test_verify_manifest_catches_injected_patient_leak():
    """故意制造泄漏，确认守卫真的会报警（守卫本身也要被测）。"""
    df = _fake_eyepacs().copy()
    assert not [p for p in verify_manifest(df) if "patient 交集" in p]

    leaked = df.copy()
    victim = leaked.loc[leaked["split"] == "train", "patient_id"].iloc[0]
    # 把某个患者的一只眼挪到 test：这正是按 image 随机划分会犯的错
    idx = leaked.index[(leaked["patient_id"] == victim)][0]
    leaked.loc[idx, "split"] = "test"
    assert [p for p in verify_manifest(leaked) if "patient 交集" in p]


def test_verify_manifest_catches_ddr_ungradable():
    df = _fake_eyepacs().copy()
    df["client"] = "ddr"
    df.loc[df.index[0], "dr_grade"] = 5   # 残留的 ungradable
    problems = verify_manifest(df)
    assert any("0-4" in p or "ungradable" in p for p in problems)


def test_local_steps_and_weights():
    n = dict(eyepacs=24600, ddr=6260, aptos=2560, idrid=372)
    steps = {k: local_steps(v) for k, v in n.items()}
    assert steps == {"eyepacs": 200, "ddr": 101, "aptos": 65, "idrid": 25}
    # sqrt 规则把 epoch-based 的 64x 不平衡压到 8x
    assert steps["eyepacs"] / steps["idrid"] == 8.0

    w_sample = aggregation_weights(list(n.values()), "sample")
    w_sqrt = aggregation_weights(list(n.values()), "sqrt")
    # 按样本数时 IDRiD 权重不到 2%，sqrt 之后应显著抬升
    assert w_sample[-1] < 0.02 < w_sqrt[-1]


def test_client_equal_prototype_aggregation():
    d = 8
    protos = {"eyepacs": torch.zeros(5, d), "idrid": torch.zeros(5, d)}
    protos["eyepacs"][1, 0] = 1.0
    protos["idrid"][1, 1] = 1.0
    seen = {k: torch.tensor([False, True, False, False, False]) for k in protos}
    sizes = {"eyepacs": 24600, "idrid": 372}

    eq = aggregate_prototypes(protos, seen, sizes, "client_equal")[1]
    sm = aggregate_prototypes(protos, seen, sizes, "sample")[1]
    # 等权时两个方向各占一半；按样本数时 IDRiD 基本被淹没
    assert abs(float(eq[0]) - float(eq[1])) < 1e-5
    assert float(sm[1]) < 0.05
    # 没有任何 client 拥有的类别保持 0 向量
    assert float(aggregate_prototypes(protos, seen, sizes, "client_equal")[0].abs().sum()) == 0.0


def test_qwk_bounds_and_direction():
    y = np.array([0, 1, 2, 3, 4] * 20)
    assert quadratic_weighted_kappa(y, y) == 1.0
    # 差一格 应该远好于 差四格
    near = quadratic_weighted_kappa(y, np.clip(y + 1, 0, 4))
    far = quadratic_weighted_kappa(y, 4 - y)
    assert near > far


def test_macro_and_worst_differ_from_weighted():
    per = {
        "eyepacs": {"qwk": 0.80, "n": 7000},
        "idrid": {"qwk": 0.40, "n": 103},
    }
    s = aggregate_over_clients(per, "qwk")
    assert abs(s["macro_qwk"] - 0.60) < 1e-6
    assert abs(s["worst_qwk"] - 0.40) < 1e-6
    # 样本加权几乎等于只看 EyePACS —— 这正是主表不能用它的原因
    assert s["weighted_qwk"] > 0.79


def test_ordinal_losses_respect_grade_distance():
    logits_near = torch.tensor([[0.0, 0.0, 0.0, 1.0, 5.0]])   # 真值 3，预测 4
    logits_far = torch.tensor([[5.0, 0.0, 0.0, 1.0, 0.0]])    # 真值 3，预测 0
    y = torch.tensor([3])
    assert squared_emd_loss(logits_near, y) < squared_emd_loss(logits_far, y)

    protos = torch.eye(5, 8)
    feat = protos[3:4].clone()
    loss = ordinal_prototype_loss(feat, y, protos, margin=0.5)
    assert torch.isfinite(loss) and float(loss) >= 0.0


def test_effective_number_weights_are_clipped():
    w = effective_number_weights([25810, 2443, 5292, 873, 708])
    assert w.min() >= 0.1 and w.max() <= 10.0
    assert w[0] < w[4]                       # 多数类权重更小
    w0 = effective_number_weights([100, 0, 50, 10, 5])
    assert float(w0[1]) == 0.0               # 本地没有的类别不产生梯度


def test_forward_backward_and_upload_split():
    torch.manual_seed(0)
    m = build_model(backbone="debug_vit", img_size=64, lora_last_n_blocks=4, lora_rank=4)
    x = torch.randn(4, 3, 64, 64)
    y = torch.tensor([0, 2, 4, 1])

    out = m(x)
    assert out.logits.shape == (4, 5)
    m.update_prototypes(out, y)

    crit = FedOSPLoss([100, 20, 50, 10, 5], LossWeights())
    loss, parts = crit(out, y, deep_protos=m.deep_proto.proto,
                       shallow_protos=m.shallow_proto.proto, out_aug=m(x))
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) >= {"cbce", "ord", "proto_grade", "proto_style", "style", "cons"}

    trainable_with_grad = [
        n for n, p in m.named_parameters() if p.requires_grad and p.grad is not None
    ]
    assert any("lora_" in n for n in trainable_with_grad)
    assert any("gate_logit" in n for n in trainable_with_grad)

    shared = m.shared_state_dict()
    personal = m.personal_state_dict()
    # LayerNorm 绝不能出现在上传内容里
    assert not [k for k in shared if "norm" in k]
    assert personal and all("norm" in k for k in personal)
    assert m.communication_mb() > 0


def test_fsr_preserves_shape_and_is_learnable():
    m = build_model(backbone="debug_vit", img_size=64, lora_last_n_blocks=2)
    m_off = build_model(backbone="debug_vit", img_size=64, lora_last_n_blocks=2, use_fsr=False)
    x = torch.randn(2, 3, 64, 64)
    assert m(x).shallow_feat.shape == m_off(x).shallow_feat.shape
    assert any("gate_logit" in n for n, p in m.named_parameters() if p.requires_grad)
    assert not any("gate_logit" in n for n, _ in m_off.named_parameters())


def test_edl_uncertainty_in_range():
    u = edl_uncertainty(torch.randn(16, 5))
    assert bool(((u > 0) & (u <= 5)).all())


def test_label_budget_keeps_all_grades():
    df = _fake_eyepacs()
    train = df[df["split"] == "train"]
    small = apply_label_budget(train, "100", 0)
    assert len(small) <= len(train)
    assert small["dr_grade"].nunique() == train["dr_grade"].nunique()


# --------------------------------------------------------------------------- #
# 骨干适配层：A12 消融要换骨干，这三个必须都能跑且参数量在同一量级
# --------------------------------------------------------------------------- #
def test_backbone_adapters_all_work():
    """ViT / SwinV2 / ResNet 三条路径都能前向反向，且本地 norm 不上传。

    这条测试的由来：早前 ``SwinV2`` 报 ``no attribute 'blocks'``、
    ``ResNet50`` 报 ``unexpected keyword 'img_size'``，A12 骨干消融根本跑不了。
    """
    import pytest

    pytest.importorskip("timm")
    from fedosp.models.retfound_lora import build_model

    # 用小号模型省测试时间，结构路径与 ViT-L 完全一致
    for backbone, size in [
        ("vit_small_patch16_224", 224),
        ("swinv2_tiny_window8_256", 256),
        ("resnet50", 224),
    ]:
        m = build_model(backbone=backbone, img_size=size, lora_last_n_blocks=6)
        out = m(torch.randn(2, 3, size, size))
        out.logits.sum().backward()

        assert out.logits.shape == (2, 5), backbone
        shared = set(m.shared_state_dict())
        personal = set(m.personal_state_dict())
        assert not (shared & personal), f"{backbone}: 本地 norm 泄漏进上传集合"
        assert personal, f"{backbone}: 一个本地 norm 参数都没识别出来"
        # 必须真的注入了 LoRA，否则退化成「只训分类头」，与其它骨干不可比
        assert any("lora_" in k for k in shared), f"{backbone}: 没有注入 LoRA"


def test_fsr_reports_frequency_resolution():
    """FSR 在三种排布下都要保持形状不变。

    ResNet 的频率 bin 数应远多于 ViT（56x56 vs 14x14），
    这是「频率分辨率是否决定 FSR 收益」这一假设的实验基础。
    """
    from fedosp.models.fsr import FrequencyStyleRecalibration
    from fedosp.models.backbones import LAYOUT_NCHW, LAYOUT_NHWC, LAYOUT_TOKENS

    cases = [
        (LAYOUT_TOKENS, torch.randn(2, 197, 64), 1),
        (LAYOUT_NHWC, torch.randn(2, 16, 16, 64), 0),
        (LAYOUT_NCHW, torch.randn(2, 64, 28, 28), 0),
    ]
    for layout, x, prefix in cases:
        fsr = FrequencyStyleRecalibration(64, layout=layout, num_prefix_tokens=prefix)
        y = fsr(x)
        assert y.shape == x.shape, layout
        y.sum().backward()
        assert fsr.gate_logit.grad is not None, f"{layout}: 门控没有梯度"


# --------------------------------------------------------------------------- #
# FSR 的数值正确性。
#
# 上面两条只验了形状和梯度 —— 一个什么都不做的模块也能通过。FSR 的全部立论是
# 「幅度承载风格、相位承载语义，所以只归一化幅度、原样保留相位」，这三条测试
# 直接检验这个数学声明本身。
# --------------------------------------------------------------------------- #
def _open_gate(fsr):
    """把门控推到全开（g≈1），此时 FSR 应等价于纯幅度归一化。"""
    with torch.no_grad():
        fsr.gate_logit.fill_(20.0)   # sigmoid(20) ≈ 1
    return fsr


def test_fsr_preserves_phase_spectrum():
    """★ FSR 的核心声明：相位谱必须逐 bin 不变，无论门控开多大。

    相位被改动就意味着病灶的空间结构被破坏，整个方法的立论就不成立了。
    """
    from fedosp.models.fsr import FrequencyStyleRecalibration
    from fedosp.models.backbones import LAYOUT_NCHW

    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, 16)
    fsr = _open_gate(FrequencyStyleRecalibration(8, layout=LAYOUT_NCHW))

    with torch.no_grad():
        spec_before = torch.fft.fft2(x.double(), norm="ortho")
        spec_after = torch.fft.fft2(fsr(x).double(), norm="ortho")

    # 相位是角度，要在圆周上比较：wrap 到 (-pi, pi] 之后再看差
    d_pha = spec_after.angle() - spec_before.angle()
    d = torch.atan2(torch.sin(d_pha), torch.cos(d_pha))
    # 相位只在幅度非零处有定义。输入端幅度近 0，或输出端被 clamp_min(0) 归零的 bin，
    # 其 angle() 是任意值，必须排除后再比较。
    mask = (spec_before.abs() > 1e-3) & (spec_after.abs() > 1e-3)
    assert mask.any(), "掩码把所有 bin 都排除了，测试本身失效"
    assert d[mask].abs().max() < 1e-4, f"相位被改动了，最大偏差 {d[mask].abs().max():.2e}"

    # 被 clamp 归零的 bin 只能是极少数：如果占比很高，说明幅度归一化把频谱打穿了，
    # 「保留相位」就只是名义上成立。实测各种分辨率下都在 0.2% 以内。
    zeroed = float((~mask & (spec_before.abs() > 1e-3)).float().mean())
    assert zeroed < 0.01, f"有 {zeroed:.2%} 的有效 bin 被归零，幅度归一化过强"


def test_fsr_suppresses_amplitude_style_shift():
    """★ FSR 必须真的压掉风格差异。

    用「同内容 + 不同逐通道幅度增益」模拟两个相机的风格差（这正是风格差在幅度谱上
    的表现形式），检验 FSR 之后两者的距离显著变小。
    """
    from fedosp.models.fsr import FrequencyStyleRecalibration
    from fedosp.models.backbones import LAYOUT_NCHW

    torch.manual_seed(0)
    base = torch.randn(4, 8, 16, 16)
    # 客户端 B：同一内容，但**逐通道**幅度增益不同（相机/光照差异在幅度谱上就是这个形态）。
    # 增益的几何均值归一到 1，这样整体亮度一致、差异纯粹落在通道间比例上，
    # 正好对应 FSR 声称要压掉的那一部分。
    gain = torch.exp(torch.randn(1, 8, 1, 1) * 0.5)
    gain = gain / gain.log().mean().exp()
    shifted = base * gain

    fsr = _open_gate(FrequencyStyleRecalibration(8, layout=LAYOUT_NCHW))
    with torch.no_grad():
        d_before = (base - shifted).pow(2).mean().item()
        d_after = (fsr(base) - fsr(shifted)).pow(2).mean().item()

    assert d_after < d_before, (
        f"FSR 没有压掉风格差：处理前距离 {d_before:.4f} → 处理后 {d_after:.4f}"
    )
    # 只是「略微变小」不够，方法要有意义就得有量级上的压缩
    assert d_after < 0.5 * d_before, (
        f"风格压制幅度不足：{d_before:.4f} → {d_after:.4f}（期望至少减半）"
    )


def test_fsr_gate_interpolates_between_identity_and_normalization():
    """★ 门控语义必须正确：g→0 等于什么都不做，g→1 等于完全归一化。

    门控接反或被量纲修正（``amp_norm * sd.detach() + mu.detach()``）抵消掉的话，
    A1 消融就失去意义 —— 开关 FSR 会看不出差别。
    """
    from fedosp.models.fsr import FrequencyStyleRecalibration
    from fedosp.models.backbones import LAYOUT_NCHW

    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 16) * 3.0 + 1.0

    closed = FrequencyStyleRecalibration(8, layout=LAYOUT_NCHW)
    with torch.no_grad():
        closed.gate_logit.fill_(-20.0)      # sigmoid(-20) ≈ 0
        y_closed = closed(x)
    assert torch.allclose(y_closed, x, atol=1e-4), "g→0 时 FSR 应当是恒等映射"

    opened = _open_gate(FrequencyStyleRecalibration(8, layout=LAYOUT_NCHW))
    with torch.no_grad():
        y_open = opened(x)
    assert not torch.allclose(y_open, x, atol=1e-3), "g→1 时 FSR 必须真的改变了特征"

    # g→1 的定义性效果：各通道的幅度谱统计量被拉平到同一水平。
    # 相机风格差异正是体现在「通道之间的幅度比例」上，所以这个跨通道离散度
    # 才是 FSR 要压掉的量（而不是单通道内的频率离散度）。
    with torch.no_grad():
        amp_in = torch.fft.fft2(x.double(), norm="ortho").abs()
        amp_out = torch.fft.fft2(y_open.double(), norm="ortho").abs()
        # 每个通道的幅度均值 → 再看这些均值在通道维上的离散程度
        spread_in = amp_in.mean(dim=(-2, -1)).std(dim=1).mean()
        spread_out = amp_out.mean(dim=(-2, -1)).std(dim=1).mean()
    assert spread_out < 0.5 * spread_in, (
        f"跨通道幅度离散度没有被压平：{spread_in:.4f} → {spread_out:.4f}（期望至少减半）"
    )


def test_scaffold_control_variate_is_not_silently_empty():
    """SCAFFOLD 必须真的产出 control variate。

    之前的实现只在 ``optimizer.step()`` **之后**记了个梯度滑动平均，
    从头到尾没修正过任何一次更新 —— 等于静默退化成 FedAvg，
    当基线用会得出错误结论。这条测试就是防止再次退化。
    """
    from fedosp.fed.strategies import ClientUpdate, Scaffold, ServerState

    strat = Scaffold()
    assert getattr(strat, "needs_control", False), (
        "Scaffold 必须声明 needs_control，否则 run_fed 会把 state.control 初始化成 None，"
        "client 侧就不会记录本轮起点"
    )

    # 构造两个 client 的上传包，各带一个非零的 control delta
    shared = {"head.weight": torch.randn(5, 8), "head.bias": torch.zeros(5)}
    updates = [
        ClientUpdate(
            client=f"c{i}", n=100 * (i + 1),
            shared={k: v.clone() for k, v in shared.items()},
            control_delta={"head.weight": torch.full((5, 8), 0.1 * (i + 1))},
        )
        for i in range(2)
    ]
    state = ServerState(shared=shared, control={})
    new_state = strat.aggregate(updates, state)

    assert new_state.control, "聚合后 control 仍为空 —— control variate 没有被累积"
    c = new_state.control["head.weight"]
    # c <- c + mean(dc_i) = 0 + mean(0.1, 0.2) = 0.15
    assert torch.allclose(c, torch.full((5, 8), 0.15), atol=1e-6), (
        f"control 累积值不对，期望 0.15，实际 {c.mean():.4f}"
    )

    # 再聚合一轮，control 应继续累加而不是被覆盖
    state2 = strat.aggregate(updates, new_state)
    assert torch.allclose(state2.control["head.weight"], torch.full((5, 8), 0.30), atol=1e-6), (
        "第二轮 control 应累加到 0.30，说明它被覆盖而非累积"
    )


def test_delong_matches_sklearn_auc():
    """DeLong 检验算出的 AUC 必须与 sklearn 完全一致。"""
    import pytest

    pytest.importorskip("sklearn")
    from sklearn.metrics import roc_auc_score

    from fedosp.stats import delong_test

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    a = y * 1.2 + rng.normal(0, 1, 400)
    b = y * 0.5 + rng.normal(0, 1, 400)

    r = delong_test(y, a, b)
    expected = roc_auc_score(y, a) - roc_auc_score(y, b)
    assert abs(r.effect - expected) < 1e-9, f"AUC 差值 {r.effect} != sklearn 的 {expected}"
    assert r.p_value < 0.01, "明显差异没被检出"
    # 自己和自己比必须不显著
    assert delong_test(y, a, a).p_value > 0.99


def test_holm_bonferroni_is_monotone():
    """Holm 校正后的 p 值必须单调不减，且不小于原始 p 值。"""
    from fedosp.stats import TestResult, holm_bonferroni

    raw = [0.001, 0.01, 0.03, 0.04, 0.2]
    results = [TestResult(f"c{i}", 0.0, p, 0.1) for i, p in enumerate(raw)]
    holm_bonferroni(results)

    corrected = [r.p_corrected for r in results]
    assert all(c >= p for c, p in zip(corrected, raw)), "校正后 p 值不应变小"
    assert corrected == sorted(corrected), f"校正后 p 值不单调：{corrected}"


# =========================================================================== #
# C2：精度加权聚合与 n_eff 诊断
# =========================================================================== #
def test_precision_weighting_degenerates_to_sample_weighted():
    """tau^2 = 0（无域偏移）时，精度加权必须精确退化为按样本量加权。

    这是 C2 理论的第一个支点：它证明 FedProto 的做法是我们的一个特例，
    而不是一个平行的竞争选项。
    """
    from fedosp.fed.diagnostics import precision_weights

    n = np.array([24600.0, 6260.0, 2560.0, 372.0])
    protos = np.random.RandomState(0).randn(4, 64) * 0.1
    w, _ = precision_weights(protos, sampling_vars=1.0 / n, tau2=0.0)
    assert np.allclose(w, n / n.sum()), f"tau^2=0 应给出按样本量权重，实得 {w}"


def test_precision_weighting_degenerates_to_client_equal():
    """tau^2 >> v_k（域偏移主导）时，精度加权必须退化为 client 等权。

    这是第二个支点：方案 v1.0 的「client 等权」也是特例。于是 A6 消融从
    「三个拍脑袋选项」变成「沿 tau^2 一条理论曲线的扫描」。
    """
    from fedosp.fed.diagnostics import precision_weights

    n = np.array([24600.0, 6260.0, 2560.0, 372.0])
    protos = np.random.RandomState(0).randn(4, 64) * 0.1
    w, _ = precision_weights(protos, sampling_vars=1.0 / n, tau2=1e10)
    assert np.allclose(w, 0.25, atol=1e-6), f"tau^2->inf 应给出等权，实得 {w}"


def test_precision_weighting_is_monotone_in_tau2():
    """随 tau^2 增大，最大 client 的权重必须单调下降、最小 client 单调上升。"""
    from fedosp.fed.diagnostics import precision_weights

    n = np.array([24600.0, 6260.0, 2560.0, 372.0])
    protos = np.random.RandomState(0).randn(4, 64) * 0.1
    big, small = [], []
    for tau2 in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        w, _ = precision_weights(protos, 1.0 / n, tau2=tau2)
        big.append(w[0])
        small.append(w[-1])
    assert all(x >= y - 1e-12 for x, y in zip(big, big[1:])), f"最大 client 权重非单调：{big}"
    assert all(x <= y + 1e-12 for x, y in zip(small, small[1:])), f"最小 client 权重非单调：{small}"


def test_dersimonian_laird_recovers_known_tau2():
    """DL 估计量在已知 tau^2 的合成数据上必须近似无偏（多次重复取均值）。"""
    from fedosp.fed.diagnostics import dersimonian_laird_tau2

    rng = np.random.RandomState(0)
    k, d = 12, 64
    n = np.full(k, 500.0)
    v = 1.0 / n
    for true_tau2 in [0.0, 0.01, 0.1]:
        ests = []
        for _ in range(200):
            mu = rng.randn(d)
            b = rng.randn(k, d) * np.sqrt(true_tau2)      # 域偏置，各向同性
            e = rng.randn(k, d) * np.sqrt(v[0])           # 有限样本噪声
            ests.append(dersimonian_laird_tau2(mu + b + e, v))
        got = float(np.mean(ests))
        # tau^2=0 时 DL 因截断在 0 而有正偏，只检查它很小
        if true_tau2 == 0.0:
            assert got < 0.005, f"tau^2=0 时估计值应接近 0，实得 {got:.4f}"
        else:
            assert abs(got - true_tau2) / true_tau2 < 0.15, (
                f"真值 {true_tau2} 估计 {got:.4f}，相对误差超过 15%"
            )


def test_dersimonian_laird_edge_cases():
    """单 client、零方差等边界不能抛异常或返回 nan。"""
    from fedosp.fed.diagnostics import dersimonian_laird_tau2, precision_weights

    assert dersimonian_laird_tau2(np.zeros((1, 8)), np.array([0.1])) == 0.0
    t = dersimonian_laird_tau2(np.zeros((4, 8)), np.array([0.1] * 4))
    assert t == 0.0, f"完全一致的原型应给出 tau^2=0，实得 {t}"
    w, _ = precision_weights(np.random.randn(3, 8), np.array([0.0, 0.0, 0.0]))
    assert np.isfinite(w).all() and abs(w.sum() - 1) < 1e-9


def test_effective_client_count_matches_hand_calculation():
    """n_eff 的数值必须与设计文档 §2 表格中的手算值一致。"""
    from fedosp.fed.diagnostics import (
        compound_effective_client_count,
        effective_client_count,
    )

    n = [24600, 6260, 2560, 372]
    assert abs(effective_client_count([1] * 4) - 4.0) < 1e-9, "等权必须给出 n_eff = K"
    assert abs(effective_client_count(n) - 1.754) < 5e-3
    assert abs(effective_client_count(np.sqrt(n)) - 2.768) < 5e-3
    # 复合 n_eff：聚合权重 x 本地步数
    assert abs(compound_effective_client_count(n, [769, 196, 80, 12]) - 1.153) < 5e-3
    assert abs(compound_effective_client_count([1] * 4, [200, 101, 65, 25]) - 2.777) < 5e-3
    assert abs(compound_effective_client_count([1] * 4, [100] * 4) - 4.0) < 1e-9


def test_effective_client_count_bounded_by_k():
    """Cauchy-Schwarz：n_eff <= K 对任意权重成立，等号仅在均匀权重处取到。"""
    from fedosp.fed.diagnostics import effective_client_count

    rng = np.random.RandomState(0)
    for _ in range(200):
        k = rng.randint(2, 12)
        w = rng.dirichlet(np.ones(k) * rng.choice([0.1, 1.0, 10.0]))
        assert effective_client_count(w) <= k + 1e-9


def test_prototype_bank_tracks_sampling_variance():
    """原型库上报的抽样方差必须随样本量下降，且与实测原型方差同阶。

    这是精度加权的输入。如果它算错了（比如拿累计样本数当 n 而忽略 EMA 的有效窗口），
    权重就会系统性偏向大 client —— 那 C2 就等于什么都没做。
    """
    from fedosp.models.prototypes import PrototypeBank

    torch.manual_seed(0)
    dim, n_rep = 16, 300
    var_by_batch = {}
    for b in (4, 32):
        spreads = []
        for r in range(n_rep):
            bank = PrototypeBank(num_classes=2, dim=dim, momentum=0.9, normalize=False)
            for _ in range(20):
                feat = torch.randn(b, dim)
                bank.update(feat, torch.zeros(b, dtype=torch.long))
            spreads.append(bank.proto[0].clone())
        emp = torch.stack(spreads).var(dim=0, unbiased=True).mean().item()
        var_by_batch[b] = (bank.sampling_variance()[0].item(), emp)

    for b, (reported, emp) in var_by_batch.items():
        ratio = reported / emp
        assert 0.5 < ratio < 2.0, (
            f"batch={b}: 上报抽样方差 {reported:.5f} 与实测 {emp:.5f} 差了 {ratio:.2f} 倍"
        )
    assert var_by_batch[32][0] < var_by_batch[4][0], "样本量更大时抽样方差必须更小"


def test_precision_aggregation_end_to_end():
    """aggregate_prototypes(mode='precision') 能跑通并回填诊断量。"""
    from fedosp.models.prototypes import aggregate_prototypes

    rng = np.random.RandomState(0)
    names = ["eyepacs", "ddr", "aptos", "idrid"]
    n = dict(zip(names, [24600, 6260, 2560, 372]))
    protos = {k: torch.from_numpy(rng.randn(5, 32)).float() for k in names}
    seens = {k: torch.ones(5, dtype=torch.bool) for k in names}
    seens["idrid"][1] = False                     # IDRiD 没有 grade 1
    svars = {k: torch.full((5,), 1.0 / n[k]) for k in names}
    svars["idrid"][1] = float("inf")

    diag = {}
    out = aggregate_prototypes(protos, seens, n, mode="precision",
                               sampling_vars=svars, diagnostics=diag)
    assert out.shape == (5, 32)
    assert torch.isfinite(out).all()
    assert set(diag) >= {"proto_weights", "tau2_per_class"}
    # grade 1 只有 3 个 owner，其余类 4 个
    assert len(diag["proto_weights"][1]) == 3
    assert len(diag["proto_weights"][0]) == 4
    for cls, w in diag["proto_weights"].items():
        assert abs(sum(w) - 1.0) < 1e-6, f"类 {cls} 的权重没归一化：{sum(w)}"

    with assert_raises(ValueError):
        aggregate_prototypes(protos, seens, n, mode="precision")   # 缺 sampling_vars
    with assert_raises(ValueError):
        aggregate_prototypes(protos, seens, n, mode="不存在的模式")


def assert_raises(exc):
    """极简 raises 上下文管理器，避免为此依赖 pytest（本文件要能裸跑）。"""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        try:
            yield
        except exc:
            return
        raise AssertionError(f"期望抛出 {exc.__name__}，但没有")

    return _cm()


# =========================================================================== #
# C1：序数几何检验指标 T6 / T7
# =========================================================================== #
def test_t6_separates_ordinal_from_uniform_geometry():
    """T6 必须把「序数几何」和「均匀几何」区分开 —— 否则它无法作为 C1 的证据。

    均匀几何（正交单形，标准 CE 的典型解）是 T6 的零假设，必须得 0 分；
    共线等距的序数几何必须得满分。
    """
    from fedosp.metrics import prototype_geometry_metrics

    ordinal = np.zeros((5, 32))
    ordinal[:, 0] = np.arange(5)                      # 沿一条轴等距排列
    uniform = np.zeros((5, 32))
    uniform[np.arange(5), np.arange(5)] = 1.0         # 正交：所有成对距离相等

    m_ord = prototype_geometry_metrics(ordinal, valid_mask=[True] * 5)
    m_uni = prototype_geometry_metrics(uniform, valid_mask=[True] * 5)

    assert m_ord["spearman_rho"] > 0.99, m_ord
    assert m_ord["linear_r2"] > 0.99, m_ord
    assert m_ord["adjacent_violations"] < 1e-9, m_ord
    assert abs(m_ord["dist_ratio_far_near"] - 4.0) < 1e-6, m_ord

    # 退化几何必须给出有限的零假设基准，不能是 nan
    assert abs(m_uni["spearman_rho"]) < 1e-9, m_uni
    assert abs(m_uni["linear_r2"]) < 1e-9, m_uni
    assert abs(m_uni["adjacent_violations"] - 0.5) < 1e-9, m_uni
    assert abs(m_uni["dist_ratio_far_near"] - 1.0) < 1e-9, m_uni
    assert all(np.isfinite(v) for v in m_uni.values()), f"退化几何返回了 nan：{m_uni}"


def test_t6_handles_missing_classes_by_gap_not_index():
    """某个 client 没见过的等级被剔除后，T6 仍要按**真实等级差**而非数组下标算。

    IDRiD 只有 372 张、经常缺 grade 1，如果按下标算会把 |0-2| 误当成 |0-1|，
    单调性结论就全错了。
    """
    from fedosp.metrics import prototype_geometry_metrics

    ordinal = np.zeros((5, 32))
    ordinal[:, 0] = np.arange(5)
    m = prototype_geometry_metrics(ordinal, valid_mask=[True, False, True, True, True])
    assert m["n_valid_classes"] == 4
    assert m["spearman_rho"] > 0.99 and m["linear_r2"] > 0.99, m
    assert abs(m["dist_ratio_far_near"] - 4.0) < 1e-6, m


def test_t7_distinguishes_error_structure_at_equal_accuracy():
    """T7 必须在**准确率完全相同**时区分出误判结构 —— 这正是它存在的理由。"""
    from fedosp.metrics import far_error_metrics

    y = np.repeat(np.arange(5), 100)
    near = y.copy()
    near[::5] = np.clip(y[::5] + 1, 0, 4)        # 错到隔壁
    far = y.copy()
    far[::5] = 4 - y[::5]                        # 错到对端

    assert (y == near).mean() == (y == far).mean(), "两种误判的准确率必须相同"
    m_near, m_far = far_error_metrics(y, near), far_error_metrics(y, far)
    assert m_near["far_error_rate"] < 1e-9 < m_far["far_error_rate"], (m_near, m_far)
    assert m_far["far_error_share"] > 0.99
    assert m_far["mean_error_gap"] > m_near["mean_error_gap"]

    perfect = far_error_metrics(y, y)
    assert perfect["far_error_rate"] == 0.0 and perfect["far_error_share"] == 0.0


# --------------------------------------------------------------------------- #
# B12–B17 新基线：每一条都在验证「机制真的改变了行为」，而不只是「能跑完」
#
# 这一组测试的存在理由：5 个新策略全都能无报错跑完 3 轮并产出格式正常的
# result.json —— 但「能跑完」和「实现对了」是两件事。一个静默退化成 FedAvg 的
# 基线会让主表上多出一行看起来合理、实际毫无意义的数字，而且事后完全无法分辨。
# SCAFFOLD 就真的这样退化过一次（见 test_scaffold_control_variate_is_not_silently_empty）。
# --------------------------------------------------------------------------- #
def _tiny_client(name: str = "c0", n: int = 24, style_aug: bool = False):
    """造一个能跑真前向的最小 client（debug_vit + 合成数据）。"""
    from torch.utils.data import DataLoader, Dataset

    from fedosp.fed.client import LocalClient
    from fedosp.losses import LossWeights
    from fedosp.models.retfound_lora import FedOSPConfig, build_model

    class _DS(Dataset):
        def __init__(self) -> None:
            g = torch.Generator().manual_seed(abs(hash(name)) % 2**31)
            self.x = torch.randn(n, 3, 32, 32, generator=g)
            self.y = torch.arange(n) % 5

        def __len__(self) -> int:
            return n

        def __getitem__(self, i):
            if style_aug:
                return self.x[i], self.x[i] + 0.05, int(self.y[i])
            return self.x[i], int(self.y[i])

        def class_counts(self):
            return np.bincount(self.y.numpy(), minlength=5).astype(float)

    ds = _DS()
    loaders = {s: DataLoader(ds, batch_size=8) for s in ("train", "val", "test")}
    model = build_model(FedOSPConfig(backbone="debug_vit", img_size=32, stage="smoke"))
    return LocalClient(name, model, loaders, loss_weights=LossWeights(), lr=1e-3)


def test_moon_contrastive_is_zero_in_round1_and_active_afterwards():
    """MOON 的对比项必须第 1 轮为 0、第 2 轮起非 0，且梯度能传回去。

    两个失效模式都要防：

    1. 第 1 轮没有 ``prev`` 却硬拿全局模型顶上 —— 损失恒为 ``log 2``、梯度恒为 0，
       白烧两次前向。
    2. 第 2 轮起 ``prev`` 没被正确保存 —— 对比项恒为 0，MOON 静默退化成 FedAvg。
    """
    from fedosp.fed.strategies import ServerState

    c = _tiny_client()
    cfg = {"moon": True, "moon_mu": 1.0, "moon_tau": 0.5}
    state = ServerState(shared=c.model.shared_state_dict())

    # 第 1 轮：没有上一轮本地模型
    c.load_from_server(state, cfg)
    assert c._moon_prev is None, "第 1 轮不应该有 prev 模型"
    x = torch.randn(4, 3, 32, 32)
    assert c._moon_reference_feats(x) is None, "第 1 轮应跳过两次冻结前向"
    c.local_train(3, cfg)

    # 第 2 轮：prev 必须是上一轮训完的本地模型，且与下发的全局参数不同
    state2 = ServerState(shared={k: v + 0.2 for k, v in state.shared.items()})
    c.load_from_server(state2, cfg)
    assert c._moon_prev is not None, "第 2 轮必须有 prev，否则 MOON 退化成 FedAvg"
    drift = max(
        float((c._moon_prev[k] - c._moon_global[k]).abs().max())
        for k in c._moon_global
    )
    assert drift > 1e-4, f"prev 与 global 几乎相同（{drift:.2e}），对比项不会有信号"

    refs = c._moon_reference_feats(x)
    assert refs is not None and len(refs) == 2

    # 对比项非 0 且可导
    feat = c.model(x).deep_feat
    l_con = c._moon_loss(feat, refs, 0.5)
    assert float(l_con.detach()) > 0, "对比损失必须为正（softplus 恒正）"
    l_con.backward()
    assert c.model.head.weight.grad is not None, "对比项的梯度没有传回可训练参数"
    assert float(c.model.head.weight.grad.abs().sum()) > 0

    # 换参数不能破坏模型状态：跑完之后 shared 必须还是当前这份
    after = c.model.shared_state_dict()
    for k, v in after.items():
        assert not torch.isnan(v).any(), f"{k} 在换参数后出现 NaN"


def test_moon_loss_direction_rewards_closeness_to_global():
    """对比损失必须「离全局越近、离上一轮自己越远」时更小。方向错了 MOON 会反向优化。"""
    c = _tiny_client()
    z_g = F.normalize(torch.randn(6, 16), dim=-1)
    z_p = F.normalize(torch.randn(6, 16), dim=-1)

    good = c._moon_loss(z_g.clone(), (z_g, z_p), 0.5)      # 与全局完全对齐
    bad = c._moon_loss(z_p.clone(), (z_g, z_p), 0.5)       # 与上一轮自己完全对齐
    assert float(good) < float(bad), (
        f"与全局对齐的损失 {float(good):.4f} 应小于与旧本地模型对齐的 {float(bad):.4f}；"
        "方向反了说明 s_g / s_p 写颠倒了"
    )


def test_qfedavg_with_q0_exactly_reproduces_fedavg():
    """q=0 时 q-FedAvg 必须**精确**退化成 FedAvg —— 这是它唯一的闭式可验证点。

    q=0 时 ``F^q = 1``、``h_k = L``，于是
    ``w - sum(L(w-w_k)) / (K L) = w - mean(w - w_k) = mean(w_k)``。
    对不上就说明 Delta / h 的公式抄错了。
    """
    from fedosp.fed.strategies import ClientUpdate, FedAvg, QFedAvg, ServerState

    torch.manual_seed(0)
    w_global = {"head.weight": torch.randn(5, 8), "head.bias": torch.randn(5)}
    updates = [
        ClientUpdate(
            client=f"c{i}", n=100,                      # 等样本量，让 FedAvg 也取均值
            shared={k: v + torch.randn_like(v) * 0.1 for k, v in w_global.items()},
            loss_at_global=float(l),
            metrics={"steps": 10},
        )
        for i, l in enumerate([0.5, 1.5, 3.0])
    ]

    q0 = QFedAvg(q=0.0, lr=1e-3).aggregate(
        updates, ServerState(shared={k: v.clone() for k, v in w_global.items()})
    )
    avg = FedAvg().aggregate(
        updates, ServerState(shared={k: v.clone() for k, v in w_global.items()})
    )
    for k in w_global:
        assert torch.allclose(q0.shared[k], avg.shared[k], atol=1e-4), (
            f"q=0 时 {k} 与 FedAvg 不一致：最大偏差 "
            f"{float((q0.shared[k] - avg.shared[k]).abs().max()):.2e}"
        )


def test_qfedavg_shifts_toward_high_loss_client():
    """q>0 必须把更新**方向**偏向损失高的 client —— 这就是它的公平性机制。

    注意断言的是**方向**而不是步长。q-FedAvg 的步长对 q **不单调**：分母里的
    ``q F^{q-1} ||Δw||²`` 含 ``L²`` 因子，比分子的 ``F^q`` 增长更快，所以 q 越大
    有效步长反而越小（q=1 时 3.0e-4，q=5 时只剩 1.3e-4）。这是原公式的性质、
    不是 bug，但很容易被误读成"q 调大了却没反应"。见 QFedAvg 文档串。
    """
    from fedosp.fed.strategies import ClientUpdate, QFedAvg, ServerState

    w = {"head.bias": torch.zeros(3)}
    # 低损失 client 想往 -1 走，高损失 client 想往 +1 走
    updates = [
        ClientUpdate(client="easy", n=100, shared={"head.bias": torch.full((3,), -1.0)},
                     loss_at_global=0.2, metrics={"steps": 10}),
        ClientUpdate(client="hard", n=100, shared={"head.bias": torch.full((3,), 1.0)},
                     loss_at_global=2.0, metrics={"steps": 10}),
    ]
    for q in (0.5, 1.0, 3.0):
        strat = QFedAvg(q=q, lr=1e-3)
        out = strat.aggregate(
            updates, ServerState(shared={k: v.clone() for k, v in w.items()})
        )
        val = float(out.shared["head.bias"].mean())
        assert val > 0, (
            f"q={q} 时更新方向应偏向高损失 client（>0），实际 {val:.3e}"
        )

    # 真正随 q 单调的是**权重份额**：hard client 在分子里的占比
    shares = []
    for q in (0.0, 0.5, 1.0, 2.0, 3.0):
        f_easy, f_hard = 0.2**q, 2.0**q
        shares.append(f_hard / (f_easy + f_hard))
    assert shares[0] == 0.5, "q=0 时两个 client 必须等权（等价于 FedAvg）"
    assert all(b > a for a, b in zip(shares, shares[1:])), (
        f"高损失 client 的权重份额必须随 q 单调上升，实际 {[round(s, 3) for s in shares]}"
    )

    # 诊断里记录的等效权重必须与上面一致，否则公平性对比图会画错
    strat = QFedAvg(q=1.0, lr=1e-3)
    strat.aggregate(updates, ServerState(shared={k: v.clone() for k, v in w.items()}))
    assert abs(strat.history[-1]["param_weights"][1] - 2.0 / 2.2) < 1e-6, (
        "记入诊断的等效权重与 F_k^q 的归一化值不一致"
    )


def test_qfedavg_fails_loudly_without_loss_at_global():
    """client 没上报 F_k(w^t) 时必须报错，不能拿训练平均损失凑。"""
    from fedosp.fed.strategies import ClientUpdate, QFedAvg, ServerState

    updates = [
        ClientUpdate(client="c0", n=10, shared={"head.bias": torch.zeros(3)},
                     metrics={"steps": 5}),        # 故意不给 loss_at_global
    ]
    with assert_raises(RuntimeError):
        QFedAvg(q=1.0, lr=1e-3).aggregate(
            updates, ServerState(shared={"head.bias": torch.zeros(3)})
        )


def test_ditto_personal_model_diverges_and_is_used_for_eval():
    """Ditto 的个人模型必须(1)真的被训练、(2)与全局模型不同、(3)被评估实际采用。

    第 (3) 条是最关键的：如果评估仍然用 ``w``，Ditto 精确退化成 FedAvg，
    而主表只会显示成「Ditto 在本任务上没效果」—— 一个从结果里看不出来的假结论。
    """
    from fedosp.fed.strategies import Ditto, ServerState

    assert Ditto.eval_mode == "personal", "Ditto 必须声明用个人模型评估"

    c = _tiny_client()
    cfg = {"ditto": True, "ditto_lambda": 0.1}
    state = ServerState(shared=c.model.shared_state_dict())
    c.load_from_server(state, cfg)
    upd = c.local_train(4, cfg)

    assert upd.metrics["ditto_steps"] > 0, "个人模型一步都没训"
    assert upd.compute_steps >= 2 * upd.metrics["steps"], (
        "Ditto 的本地计算量应约为 2x（w 和 v 各训一遍），"
        f"实际 {upd.compute_steps} vs steps {upd.metrics['steps']}"
    )

    # v 必须与上传的 w 不同
    diff = max(float((c._ditto_v[k] - upd.shared[k]).abs().max()) for k in upd.shared)
    assert diff > 1e-6, f"个人模型与全局模型几乎相同（{diff:.2e}），prox 太强或没训"

    # personal_model 上下文必须真的换进 v，并且退出后换回 w
    w_before = {k: v.clone() for k, v in c.model.shared_state_dict().items()}
    with c.personal_model() as switched:
        assert switched is True
        inside = c.model.shared_state_dict()
        assert max(float((inside[k] - c._ditto_v[k]).abs().max()) for k in inside) < 1e-6, (
            "上下文里装的不是个人模型 v"
        )
    w_after = c.model.shared_state_dict()
    for k in w_before:
        assert torch.allclose(w_before[k], w_after[k], atol=1e-6), (
            f"退出上下文后 {k} 没有换回全局模型 —— 会污染下一轮训练"
        )


def test_ditto_context_is_noop_without_personal_model():
    """非 Ditto 策略下 personal_model() 必须是 no-op，而不是抛异常或改参数。"""
    c = _tiny_client()
    before = {k: v.clone() for k, v in c.model.shared_state_dict().items()}
    with c.personal_model() as switched:
        assert switched is False
    for k, v in c.model.shared_state_dict().items():
        assert torch.allclose(before[k], v)


def test_fedala_learns_nonuniform_weights_and_persists_them():
    """FedALA 必须(1)把 W 学离全 1、(2)跨轮保留 W、(3)低层直接覆盖。

    ``W`` 全程停在 1 就等于普通下发，FedALA 静默退化成 FedAvg。
    """
    from fedosp.fed.strategies import FedALA, ServerState

    assert FedALA.eval_mode == "local", (
        "FedALA 的个性化模型存在本地，评估前重新下发全局参数会把它冲掉"
    )

    c = _tiny_client()
    cfg = {"ala": True, "ala_lr": 0.5, "ala_iters": 8, "ala_last_n": 4, "ala_round": 0}
    # 让全局参数明显不同于本地，插值才有可观测效果
    state = ServerState(
        shared={k: v + 0.5 for k, v in c.model.shared_state_dict().items()}
    )
    c.load_from_server(state, cfg)

    assert c._ala_W, "没有建立任何插值权重 W"
    all_w = torch.cat([v.flatten() for v in c._ala_W.values()])
    assert float(all_w.min()) >= 0.0 and float(all_w.max()) <= 1.0, "W 必须被 clamp 在 [0,1]"
    assert float((all_w - 1.0).abs().max()) > 1e-4, (
        f"W 全程停在 1（最大偏离 {float((all_w - 1.0).abs().max()):.2e}），"
        "等于普通下发，FedALA 没有生效"
    )

    # 低层必须被完全覆盖成全局值（等价于 W=1）
    cur = c.model.shared_state_dict()
    low = [k for k in state.shared
           if not k.startswith("head.") and k not in c._ala_W]
    for k in low:
        assert torch.allclose(cur[k], state.shared[k].float(), atol=1e-5), (
            f"低层 {k} 应被全局参数直接覆盖"
        )

    # W 跨轮持久：第二轮的起点必须是上一轮的 W，不能重置回全 1
    snapshot = {k: v.clone() for k, v in c._ala_W.items()}
    c.local_train(3, cfg)
    cfg2 = dict(cfg, ala_round=1)
    c.load_from_server(ServerState(shared=state.shared), cfg2)
    moved = max(float((c._ala_W[k] - snapshot[k]).abs().max()) for k in snapshot)
    reset = max(float((c._ala_W[k] - 1.0).abs().max()) for k in snapshot)
    assert reset > 1e-4, "W 被重置回全 1，跨轮持久化失效"
    assert moved >= 0.0


def test_feddg_amplitude_swap_changes_style_but_keeps_phase():
    """ELCFS 的频域增广必须换掉幅度、保留相位；lam=0 必须是恒等变换。"""
    from fedosp.data.freq_aug import AmplitudeBank, amplitude_mix, amplitude_of

    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3)).astype(np.float32)
    other = rng.random((32, 32, 3)).astype(np.float32)
    amp_other = amplitude_of(other)

    assert np.abs(amplitude_mix(img, amp_other, 0.0) - img).max() < 1e-5, (
        "lam=0 必须恒等，否则增广强度的下界就不是「无增广」"
    )

    mixed = amplitude_mix(img, amp_other, 1.0, ratio=0.1)
    f_in = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
    f_out = np.fft.fftshift(np.fft.fft2(mixed, axes=(0, 1)), axes=(0, 1))
    dphase = np.abs(np.angle(f_out) - np.angle(f_in))
    dphase = np.minimum(dphase, 2 * np.pi - dphase)
    damp = np.abs(np.abs(f_out) - np.abs(f_in))
    assert np.median(dphase) < 0.05, (
        f"相位中位偏差 {np.median(dphase):.4f} rad 过大 —— 内容被改了，不只是风格"
    )
    assert damp.mean() > 10 * np.median(dphase), "幅度几乎没变，增广实际没生效"

    # bank 必须排除自己家的谱：拿自己的幅度做增广等于没增广
    bank = AmplitudeBank(
        np.stack([amplitude_of(rng.random((32, 32, 3)).astype(np.float32)) for _ in range(6)]),
        ["a", "a", "b", "b", "c", "c"],
    )
    assert set(bank.others("a").tolist()) == {2, 3, 4, 5}
    assert bank.payload_mb() > 0, "外传体积必须可计算 —— T5 与隐私讨论要引用它"


def test_style_aug_and_freq_bank_cannot_both_be_on():
    """两者都产出第二视图，同时开会让其中一个被静默丢弃，必须直接报错。"""
    from fedosp.data.dataset import FundusDataset
    from fedosp.data.freq_aug import AmplitudeBank, amplitude_of

    frame = pd.DataFrame({"path": ["/nonexistent.png"], "dr_grade": [0]})
    bank = AmplitudeBank(
        np.stack([amplitude_of(np.zeros((32, 32, 3), dtype=np.float32))]), ["a"]
    )
    with assert_raises(ValueError):
        FundusDataset(frame, img_size=32, train=True, style_aug=True, freq_bank=bank)


def test_all_strategies_expose_consistent_eval_mode():
    """每个策略都必须声明一个合法的 eval_mode，且个性化方法不能声明成 global。"""
    from fedosp.fed.strategies import STRATEGIES

    for name, cls in STRATEGIES.items():
        assert cls.eval_mode in ("global", "local", "personal"), (
            f"{name} 的 eval_mode={cls.eval_mode!r} 非法"
        )
    assert STRATEGIES["ditto"].eval_mode == "personal"
    assert STRATEGIES["fedala"].eval_mode == "local"
    # FedPer/FedBN 走 global 是对的：个性化部分本来就不在 state.shared 里
    assert STRATEGIES["fedper"].eval_mode == "global"
    assert STRATEGIES["fedper"].local_only_prefixes, (
        "FedPer 必须靠 local_only_prefixes 把 head 留在本地"
    )


def test_aux_reg_applies_to_every_strategy_including_new_ones():
    """``--aux-reg`` 必须对全部 13 个策略同等生效，新加的 5 个不能漏。"""
    from fedosp.fed.strategies import STRATEGIES, build_strategy

    for name in STRATEGIES:
        kw = {"lr": 1e-3}
        assert build_strategy(name, aux_reg=True, **kw).client_config(1)["style_aug"] is True, (
            f"{name} 在 --aux-reg 下没拿到 style_aug —— 回到了不公平比较"
        )
        assert build_strategy(name, aux_reg=False, **kw).client_config(1)["style_aug"] is False, (
            f"{name} 在关闭 --aux-reg 时仍然吃到了辅助正则"
        )


# --------------------------------------------------------------------------- #
# B17：序数范式交叉组
# --------------------------------------------------------------------------- #
def test_all_ordinal_paradigms_produce_valid_k_class_probs():
    """四种序数范式都必须能训、且在边界处给出合法的 K 类概率分布。

    阈值式范式（coral / ordinal_encoding）输出只有 K-1 列，若下游忘了转换而直接
    softmax，会得到一个 4 维的、含义完全错误的「概率」—— 而且 QWK 照样能算出一个
    像样的数字，不会报错。所以这里逐个范式检查 ``class_probs`` 的形状与归一性。
    """
    from fedosp.losses import (
        ORDINAL_HEAD_LOSSES, ORDINAL_LOSSES, FedOSPLoss, LossWeights,
    )
    from fedosp.models.retfound_lora import FedOSPConfig, build_model

    for ord_type in ORDINAL_LOSSES:
        head = ord_type if ord_type in ORDINAL_HEAD_LOSSES else "none"
        net = build_model(FedOSPConfig(
            backbone="debug_vit", img_size=32, ordinal_head=head, stage="smoke",
        ))
        crit = FedOSPLoss([100, 20, 50, 30, 10], LossWeights(ord_type=ord_type))
        x, y = torch.randn(6, 3, 32, 32), torch.randint(0, 5, (6,))
        out = net(x)

        expect_cols = 4 if head != "none" else 5
        assert out.logits.shape == (6, expect_cols), (
            f"{ord_type}: logits 应是 (6,{expect_cols})，实际 {tuple(out.logits.shape)}"
        )

        probs = net.class_probs(out.logits)
        assert probs.shape == (6, 5), f"{ord_type}: class_probs 必须是 K=5 列"
        assert torch.allclose(probs.sum(-1), torch.ones(6), atol=1e-5), (
            f"{ord_type}: 概率没有归一化"
        )
        assert (probs >= 0).all(), f"{ord_type}: 出现负概率"

        loss, parts = crit(out, y)
        assert torch.isfinite(loss), f"{ord_type}: loss 非有限"
        loss.backward()
        assert net.head.weight.grad is not None and float(net.head.weight.grad.abs().sum()) > 0, (
            f"{ord_type}: 梯度没有传到输出头"
        )
        if ord_type in ORDINAL_HEAD_LOSSES:
            assert float(parts["cbce"]) == 0.0, (
                f"{ord_type}: 阈值式范式下 CB-CE 必须关闭（K-1 列上算 K 类 CE 是错的）"
            )
            assert float(parts["ord"]) > 0, f"{ord_type}: 阈值 BCE 为 0，分类损失丢了"


def test_coral_head_is_rank_monotone_by_construction():
    """CORAL 的秩单调性必须由**结构**保证，与样本无关 —— 这是它相对独立头的唯一卖点。"""
    import torch.nn as nn

    from fedosp.models.retfound_lora import CoralHead

    torch.manual_seed(0)
    h = CoralHead(32, 5)
    with torch.no_grad():
        h.bias.copy_(torch.tensor([2.0, 1.0, -0.5, -2.0]))     # 递减偏置
    z = h(torch.randn(512, 32))
    assert int((z[:, :-1] < z[:, 1:]).sum()) == 0, (
        "递减偏置下不允许有任何单调性违反；有违反说明权重没有真正共享"
    )

    # 关键判别：违反率只能是 0 或 1（顺序由 bias 决定、与样本无关）
    with torch.no_grad():
        h.bias.copy_(torch.tensor([1.0, 2.0, -0.5, -2.0]))     # 故意把前两个调乱
    rates = (h(torch.randn(512, 32))[:, :-1] < h(torch.randn(512, 32))[:, 1:]).float().mean(0)
    # 用同一批输入重算，避免两次随机输入带来的噪声
    zz = h(torch.randn(512, 32))
    rates = (zz[:, :-1] < zz[:, 1:]).float().mean(0)
    assert all(r in (0.0, 1.0) for r in rates.tolist()), (
        f"违反率 {rates.tolist()} 不是纯 0/1，说明顺序依赖样本 —— 秩一致性没了"
    )

    # 对照：独立线性头会逐样本出现自相矛盾
    ind = nn.Linear(32, 4)
    zi = ind(torch.randn(512, 32))
    share = float((zi[:, :-1] < zi[:, 1:]).any(1).float().mean())
    assert share > 0.3, (
        f"独立头本应经常自相矛盾（实测 {share:.2f}），若接近 0 则这条对照无意义"
    )


def test_binomial_soft_target_is_unimodal_and_distance_aware():
    """Binomial 范式：损失必须随预测偏离真值单调增加，且最优点在 p=q 而非 one-hot。"""
    from fedosp.losses import binomial_unimodal_ce

    y = torch.tensor([2])
    losses = []
    for peak in range(5):
        lg = torch.zeros(1, 5)
        lg[0, peak] = 3.0
        losses.append(float(binomial_unimodal_ce(lg, y)))
    assert losses[2] < losses[1] < losses[0], f"损失应随远离真值而增加，实际 {losses}"
    assert losses[2] < losses[3] < losses[4], f"另一侧同理，实际 {losses}"
    assert abs(losses[1] - losses[3]) < 1e-4, "等距误判应有相同损失（对称性）"

    # 最优点在 p=q，损失等于 H(q)。这是该范式内在压制置信度的原因，
    # 也是它在 ECE 上天然占便宜、不能单看校准指标的原因。
    q = torch.tensor([[0.0625, 0.25, 0.375, 0.25, 0.0625]])
    assert abs(float(binomial_unimodal_ce(q.log(), y)) - float(-(q * q.log()).sum())) < 1e-5, (
        "p=q 处的损失必须精确等于 H(q)"
    )
    onehot_like = torch.zeros(1, 5)
    onehot_like[0, 2] = 20.0
    assert float(binomial_unimodal_ce(onehot_like, y)) > float(binomial_unimodal_ce(q.log(), y)), (
        "极度自信的正确预测在该范式下损失更高，这是已知性质；反过来说明软标签算错了"
    )


def test_ordinal_encoding_handles_non_monotone_thresholds():
    """ordinal_encoding 的阈值可能非单调，转 K 类概率时必须仍然合法（clamp+归一）。"""
    from fedosp.losses import ordinal_levels, ordinal_logits_to_probs

    for z in [
        torch.tensor([[-2.0, 3.0, -1.0, 0.0]]),      # 非单调
        torch.tensor([[-9.0, -9.0, -9.0, -9.0]]),    # 全部否 -> 应判 0
        torch.tensor([[9.0, 9.0, 9.0, 9.0]]),        # 全部是 -> 应判 4
    ]:
        p = ordinal_logits_to_probs(z)
        assert (p >= 0).all() and abs(float(p.sum()) - 1.0) < 1e-5, f"{z.tolist()} 给出非法概率 {p}"
    assert int(ordinal_logits_to_probs(torch.tensor([[-9.0] * 4])).argmax()) == 0
    assert int(ordinal_logits_to_probs(torch.tensor([[9.0] * 4])).argmax()) == 4

    # 标签展开必须是「累积超过阈值」而不是 one-hot
    assert ordinal_levels(torch.tensor([3]), 5).tolist() == [[1.0, 1.0, 1.0, 0.0]]
    assert ordinal_levels(torch.tensor([0]), 5).tolist() == [[0.0, 0.0, 0.0, 0.0]]


# --------------------------------------------------------------------------- #
# 本机 pilot：降规模不能把结论一起降掉
# --------------------------------------------------------------------------- #
def _real_scale_manifest() -> pd.DataFrame:
    """造一份规模与类分布都接近真实 4 院的 manifest（用于 pilot 抽样检查）。"""
    rng = np.random.default_rng(0)
    real = {"eyepacs": 24600, "ddr": 6260, "aptos": 2560, "idrid": 372}
    dist = {
        "eyepacs": [.735, .070, .150, .023, .022],
        "ddr": [.40, .05, .35, .12, .08],
        "aptos": [.49, .10, .27, .05, .09],
        "idrid": [.36, .054, .27, .20, .12],
    }
    frames = []
    for c, n in real.items():
        p = np.asarray(dist[c], dtype=float)
        p /= p.sum()
        frames.append(pd.DataFrame({
            "client": c, "split": "train",
            "dr_grade": rng.choice(5, size=n, p=p),
            "path": [f"/{c}/{i}.png" for i in range(n)],
        }))
    return pd.concat(frames, ignore_index=True)


def _pilot_subsample(part: pd.DataFrame, frac: float, floor: int) -> pd.DataFrame:
    """复刻 ``make_client_loaders`` 里的 pilot 抽样逻辑（比例 + 下限保全）。"""
    if len(part) <= floor:
        return part
    return apply_label_budget(part, str(max(int(round(len(part) * frac)), floor)), 0)


def test_pilot_subsampling_preserves_client_imbalance():
    """★ pilot 按比例抽样必须保住客户端规模不平衡 —— 否则 C2 的结论不可迁移。

    这是整个本机 pilot 能否成立的前提。C2（精度加权聚合）的收益**完全来自**
    EyePACS(24600) 对 IDRiD(372) 的 66 倍规模差，$n_\\text{eff}=1.75$。

    如果降规模用"统一截到每院 N 张"，这个联邦会变成近似等规模，
    $n_\\text{eff}$ 被推到 3.3 以上，C2 的可改进空间被**人为消掉**；
    于是 pilot 会得出"C2 没用"的结论 —— 而这个结论只是降规模方式的产物。

    这条测试就是把"必须用比例、不能用统一上限"钉死。
    """
    from fedosp.fed.diagnostics import effective_client_count

    mf = _real_scale_manifest()
    clients = ["eyepacs", "ddr", "aptos", "idrid"]
    full = [int((mf["client"] == c).sum()) for c in clients]
    n_eff_full = effective_client_count(full)
    assert abs(n_eff_full - 1.75) < 0.05, f"基准 n_eff 应≈1.75，实际 {n_eff_full:.2f}"

    # 推荐做法：按比例 + 下限保全
    prop = [len(_pilot_subsample(mf[mf["client"] == c], 0.2, 500)) for c in clients]
    n_eff_prop = effective_client_count(prop)
    assert 1.7 <= n_eff_prop <= 2.1, (
        f"按比例抽样后 n_eff={n_eff_prop:.2f} 落在 [1.7, 2.1] 之外，"
        "规模不平衡没保住，C2 的 pilot 结论不可迁移"
    )

    # 反例：统一上限会破坏不平衡。这个断言的作用是「证明前一条不是白给的」
    capped = [min(n, 2000) for n in full]
    n_eff_cap = effective_client_count(capped)
    assert n_eff_cap > 3.0, (
        f"统一截断本应显著抬高 n_eff（实测 {n_eff_cap:.2f}），若它也接近 1.75，"
        "这条对照就不成立、上面的结论也失去意义"
    )
    assert n_eff_prop < n_eff_cap - 1.0, (
        f"按比例({n_eff_prop:.2f}) 必须明显优于统一截断({n_eff_cap:.2f})"
    )


def test_pilot_subsampling_keeps_every_grade():
    """pilot 抽样后每个 client 的**每个等级**都必须还有样本。

    IDRiD 的 grade 1 全院只有 20 张，按 20% 抽只剩 4 张；
    小院整体保全（``min_train=500``）就是为了这个。等级被抽没会让
    CB-CE 的该类权重变 0、QWK 的混淆矩阵缺行，指标不再可比。
    """
    mf = _real_scale_manifest()
    for c in ["eyepacs", "ddr", "aptos", "idrid"]:
        part = mf[mf["client"] == c]
        out = _pilot_subsample(part, 0.2, 500)
        counts = out["dr_grade"].value_counts().reindex(range(5), fill_value=0)
        assert (counts > 0).all(), (
            f"{c} 抽样后等级分布 {counts.tolist()} 有空档"
        )
    # IDRiD 只有 372 张 (<500)，必须**一张不少**地保全
    idrid = mf[mf["client"] == "idrid"]
    assert len(_pilot_subsample(idrid, 0.2, 500)) == len(idrid), (
        "IDRiD（372 张）应被完整保留：它只占 6% 机时，削它没有收益却会丢稀有等级"
    )


def test_exp_mse_constrains_only_the_first_moment():
    """``exp_MSE``（NLDL 2026 竞品损失）只约束分布**均值**，对形状不敏感。

    出处：Stelter, Corbetta, ..., Silva, *Preserving Ordinality in Diabetic
    Retinopathy Grading through a Distribution-Based Loss Function*,
    NLDL 2026, PMLR 307:405-414。与本文同任务、数据集重叠三个，
    所以必须进 B17 交叉组当对照。

    这条测试把它与平方 EMD 的**本质差异**钉下来：取 :math:`y=2, K=5` 时，
    极端双峰 ``[.5,0,0,0,.5]`` 与完全均匀 ``[.2]*5`` 的均值恰好都是 2，
    于是 ``exp_MSE`` 给它们**零惩罚**；而平方 EMD 约束整条累积分布，
    会分别罚 0.20 和 0.08。

    这不是为了贬低对手 —— 他们仓库里有 ``--lamda``，实际用法应是
    ``CE + λ·exp_MSE``，CE 会补掉这个退化。记录它是为了让消融
    （A5 / B17）出现"exp_MSE 单用时校准差"这类结果时，我们**事先就知道原因**，
    而不是事后编解释。
    """
    from fedosp.losses import (
        ORDINAL_LOSSES,
        expectation_mse_loss,
        ordinal_loss_by_type,
        squared_emd_loss,
    )

    assert "exp_mse" in ORDINAL_LOSSES, "exp_mse 必须注册进 ORDINAL_LOSSES，否则 CLI 认不出"

    K, y = 5, 2
    target = torch.tensor([y])
    # 用极大 logit 精确构造目标概率分布（softmax 的数值近似）
    def as_logits(p):
        return torch.log(torch.tensor([p], dtype=torch.float32).clamp_min(1e-12))

    cases = {
        "perfect":  [0.0, 0.0, 1.0, 0.0, 0.0],
        "adjacent": [0.0, 0.5, 0.5, 0.0, 0.0],
        "bimodal":  [0.5, 0.0, 0.0, 0.0, 0.5],
        "uniform":  [0.2, 0.2, 0.2, 0.2, 0.2],
    }
    got = {k: (float(expectation_mse_loss(as_logits(p), target, K)),
               float(squared_emd_loss(as_logits(p), target, K)))
           for k, p in cases.items()}

    # 均值恰为 2 的三种分布，exp_MSE 一律为 0
    for k in ("perfect", "bimodal", "uniform"):
        assert got[k][0] < 1e-6, (
            f"{k} 的均值是 2，exp_MSE 本应为 0，实际 {got[k][0]:.4f}"
        )
    # 而平方 EMD 必须能区分它们
    assert got["perfect"][1] < 1e-6, "完美单峰的 EMD 应为 0"
    assert got["bimodal"][1] > got["uniform"][1] > 1e-3, (
        f"平方 EMD 应当罚双峰 > 均匀 > 0，实际 "
        f"双峰 {got['bimodal'][1]:.4f} / 均匀 {got['uniform'][1]:.4f}"
    )
    # 相邻单峰：均值偏了，所以两者都非零（这是唯一 exp_MSE 有反应的情形）
    assert got["adjacent"][0] > 0 and got["adjacent"][1] > 0

    # 派发与梯度
    logits = torch.randn(4, K, requires_grad=True)
    loss = ordinal_loss_by_type("exp_mse", logits, torch.randint(0, K, (4,)), K)
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0, "梯度没传回来"


def test_real_data_path_end_to_end_on_fixture(tmp_path):
    """★ 走**真实数据路径**跑完整条流水线：fixture → manifest → 预处理 → 联邦训练 → 外部评估。

    这条测试是被两个真实 bug 逼出来的，它们的共同点是：**只在真实数据路径上出现，
    而当时所有单测和冒烟测试都是 ``--dry-run``**（合成数据集，不经过 ``FundusDataset``），
    所以 55 个测试全绿、13 个策略端到端全过，真实路径却是坏的：

    1. ``FundusDataset`` 持有 ``mp.Value``（跨 worker 统计读图失败数），
       而 ``evaluate_external`` 要 ``deepcopy(clients[0])`` 当探针 →
       ``RuntimeError: Synchronized objects should only be shared between
       processes through inheritance``。
    2. 修掉上一条后又撞上 ``LocalClient._iter``（半消耗的 DataLoader 迭代器）
       ``NotImplementedError: _SingleProcessDataLoaderIter cannot be pickled``。

    也就是说：**未见中心评估这一步从来没有被执行过**，而它是论文的第二个主指标。

    这条测试覆盖 dry-run 覆盖不到的部分：五个 builder 的真实目录布局解析、
    ``FundusDataset`` 真实读图、圆形裁剪预处理、以及 ``evaluate_external`` 的探针深拷贝。
    """
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "fixture"

    def run(args, what, ok_codes=(0,)):
        r = subprocess.run([sys.executable, *args], cwd=repo,
                           capture_output=True, text=True)
        both = r.stdout + r.stderr          # 日志走 stderr
        assert r.returncode in ok_codes, f"{what} 失败（退出码 {r.returncode}）：\n{both[-2500:]}"
        return both

    run(["scripts/make_fixture.py", "--out", str(root), "--img-size", "48"], "造 fixture")

    mf = root / "manifest.csv"
    # build_manifest 在有校验未通过时返回 1。fixture 必然在"与文献规模交叉核对"
    # 那几项上 FAIL（196 张 vs 53570 张），这是**故意的**设计：保证 fixture 的
    # manifest 不可能被误当成真实数据。所以这里接受 0/1，靠下面的结构性断言把关。
    out = run(["-m", "fedosp.data.build_manifest",
               "--data-root", str(root), "--out", str(mf)],
              "build_manifest", ok_codes=(0, 1))
    # 结构性检查必须全过；与文献规模的交叉核对会 FAIL，那是 fixture 的预期行为
    for must_pass in [
        "messidor2 全部为 test",          # 未见中心隔离
        "ddr 已剔除 ungradable",          # DDR 标签 5 不是第 6 个等级
        "所有 dr_grade 落在 0-4",
    ]:
        assert f"PASS {must_pass}" in out or f"PASS  {must_pass}" in out, (
            f"结构检查 {must_pass!r} 没有 PASS。build_manifest 输出：\n{out[-2000:]}"
        )
    assert "[eyepacs] train∩test 的 patient 交集为空" in out, (
        "EyePACS 的**病人级**防泄漏检查没跑到 —— 它是 DR 数据集最常见的泄漏点"
    )

    cached = root / "manifest_cached.csv"
    run(["-m", "fedosp.data.preprocess", "--manifest", str(mf),
         "--cache-dir", str(root / "cache"), "--out", str(cached),
         "--workers", "1", "--short-side", "64"], "preprocess")
    assert cached.exists(), "预处理没写出缓存版 manifest"

    df = pd.read_csv(cached)
    assert "raw_path" in df.columns, "缺 raw_path，无法追溯原图"
    assert all("cache" in str(p) for p in df["path"]), (
        "缓存版 manifest 的 path 仍指向原图 —— 训练会读全分辨率大图，"
        "预处理等于白做，而且不会报错"
    )
    assert all(Path(p).exists() for p in df["path"]), "有缓存图缺失"

    # 真实数据路径的联邦训练。fedosp 是唯一同时用到原型、FSR 和外部评估的策略。
    out_dir = tmp_path / "run"
    log = run(["-m", "fedosp.run_fed", "--manifest", str(cached),
               "--strategy", "fedosp", "--backbone", "debug_vit", "--img-size", "64",
               "--rounds", "2", "--batch-size", "4", "--min-steps", "2",
               "--max-steps", "3", "--num-workers", "0", "--device", "cpu",
               "--out", str(out_dir)], "run_fed（真实数据路径）")

    assert "[external:messidor2]" in log, (
        "未见中心评估没有执行 —— 它是论文第二个主指标，且正是之前两个 bug 的所在"
    )

    res = json.loads((out_dir / "result.json").read_text())
    prov = res["provenance"]
    assert prov["tier"] == "pilot", "fixture 的结果必须被标成 pilot"
    # 读图失败数必须是 0：非 0 说明 manifest 的 path 拼接有问题，
    # 而失败的图会被零图替代 —— 那是**静默污染训练数据**，不是崩溃。
    assert prov["n_failed_reads"] == 0, (
        f"有 {prov['n_failed_reads']} 张图读失败，说明路径拼接有问题"
    )
    # 外部评估必须真的产出了指标，而不是空 dict
    assert res["external"] and "qwk" in res["external"], (
        f"未见中心指标缺失：{res['external']}"
    )
    assert (out_dir / "predictions.npz").exists(), "per-sample 预测没落盘，统计检验做不了"


def test_squared_distances_matches_cdist_and_needs_no_mps_fallback():
    """★ 原型距离必须用 matmul 展开，不能用 ``torch.cdist``。

    这条是被一次真实回归逼出来的：把默认设备从 cpu 改成自动探测（Apple Silicon
    上会选 mps）之后，所有用到原型损失的策略（``fedosp`` / ``fedproto``）在反向
    传播时直接抛 ``NotImplementedError: aten::_cdist_backward is not currently
    implemented for the MPS device``。也就是说不设 ``PYTORCH_ENABLE_MPS_FALLBACK=1``
    就**完全跑不了** —— 而这正好是本机 pilot 的主力路径。

    值得记的一点：先前只按**速度**评估过要不要替换 cdist（matmul 只快 1.5×，
    结论是"不值得改"）。但真正的理由不是性能而是**可移植性** —— 判据选错了，
    结论就整个反过来。

    这里同时断言数值等价（与 cdist 的差必须在 1e-5 量级内）和梯度可传，
    防止有人为了"简洁"改回 cdist。
    """
    from fedosp.losses import squared_distances

    torch.manual_seed(0)
    for B, D, C in [(8, 64, 5), (32, 1024, 5), (1, 16, 3)]:
        a = torch.randn(B, D)
        b = torch.randn(C, D)
        got = squared_distances(a, b)
        want = torch.cdist(a, b).pow(2)
        assert got.shape == (B, C), f"形状应为 {(B, C)}，实际 {tuple(got.shape)}"
        assert torch.allclose(got, want, atol=1e-4), (
            f"与 cdist 不等价，最大差 {float((got - want).abs().max()):.2e}"
        )
        assert bool((got >= 0).all()), "平方距离出现负值（浮点误差没被 clamp 住）"

    # 单位球上应满足 ||f-p||^2 = 2 - 2 f·p（原型损失的实际用法）
    f = F.normalize(torch.randn(6, 32), dim=-1)
    p = F.normalize(torch.randn(5, 32), dim=-1)
    closed_form = 2.0 - 2.0 * (f @ p.t())
    assert torch.allclose(squared_distances(f, p), closed_form, atol=1e-5)

    # 梯度必须能传回特征
    feat = torch.randn(4, 32, requires_grad=True)
    squared_distances(feat, p).sum().backward()
    assert feat.grad is not None and float(feat.grad.abs().sum()) > 0, "梯度没传回来"

    # 源码里不应再有真实的 cdist 调用（改回去就红）。
    # 只匹配「torch.cdist(」这种调用形式，放过文档字符串里对它的引用
    # （``:func:`torch.cdist``` / 反引号包裹的说明）。
    call_re = re.compile(r"(?<!`)torch\.cdist\s*\(")
    for rel in ["fedosp/losses.py", "fedosp/models/prototypes.py"]:
        src = (Path(__file__).resolve().parents[1] / rel).read_text()
        offenders = [
            ln.strip() for ln in src.splitlines()
            if call_re.search(ln) and not ln.lstrip().startswith("#")
        ]
        assert not offenders, (
            f"{rel} 里仍有 torch.cdist 调用：{offenders}；"
            "它的反向在 MPS 上未实现，请用 squared_distances()"
        )


def test_every_matrix_row_is_reachable_by_exactly_one_stage():
    """★ 矩阵里每一行都必须能被 ``--stage all`` 跑到，且只归属一个阶段。

    这条测试是被一个真实事故逼出来的：``STAGES["main"]`` 原来只有
    ``["base_", "m_"]``，而上一轮新增的 **9 行 B17 交叉组**（``ord_*``，设计文档
    §3.5 称为"生死线"，是隔离"序数损失已知增益"的关键对照）和 ``comb_fedala_c2``
    都不匹配任何前缀 —— ``--stage all`` 会**静默跳过它们，不报任何错**。

    后果是：跑完整个矩阵、拿到一堆结果、而全文最关键的对照组根本没跑，
    却完全没有任何信号提示。这种"沉默的缺失"比崩溃危险得多。

    另一半同样重要：前缀**互不为前缀**，否则一个配置会被两个阶段各跑一遍
    （项目早期真的发生过：``"a"`` 表示消融、``"a12"`` 表示骨干，a12 被双重匹配）。
    """
    import csv

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "scripts"))
    from scheduler import STAGES

    prefixes = [p for v in STAGES.values() for p in v]
    overlap = [(a, b) for a in prefixes for b in prefixes
               if a != b and b.startswith(a)]
    assert not overlap, (
        f"阶段前缀互为前缀会导致同一配置被跑两遍：{overlap}"
    )

    for csv_name in ["experiment_matrix.csv", "pilot_local.csv"]:
        path = repo_root / "configs" / csv_name
        if not path.exists():
            continue
        with open(path) as fh:
            ids = [r["exp_id"] for r in csv.DictReader(fh) if (r.get("exp_id") or "").strip()]
        assert ids, f"{csv_name} 里没读到任何 exp_id"
        for exp_id in ids:
            hits = [name for name, pres in STAGES.items()
                    if any(exp_id.startswith(p) for p in pres)]
            assert len(hits) == 1, (
                f"{csv_name} 的 {exp_id!r} 匹配到 {len(hits)} 个阶段 {hits}；"
                "应当恰好 1 个。0 个 = --stage all 会静默跳过它；"
                ">1 个 = 会被重复跑"
            )


def test_result_tier_blocks_pilot_from_being_cited_as_main():
    """降规模 run 必须被标成 ``tier=pilot`` 并列出全部不达标原因。

    动机与 ``stage`` 硬闸门相同：pilot 的 ``result.json`` 在**格式上与正式结果
    完全相同**。写论文时翻出几十个 run 目录，靠文件名和记忆分辨哪份能引用，
    是必然出错的。所以把方案对"正式实验"的定义写成可执行判定。
    """
    from argparse import Namespace

    from fedosp.models.retfound_lora import FedOSPConfig
    from fedosp.run_fed import result_tier

    main_args = Namespace(stage="main", rounds=100, dry_run=False,
                          max_train_per_client=None, train_fraction=None)
    main_cfg = FedOSPConfig(backbone="vit_large_patch16_224", img_size=224)
    r = result_tier(main_args, main_cfg, "retfound:/w/RETFound.pth")
    assert r["tier"] == "main" and r["tier_violations"] == [], (
        f"完全合规的配置被误判成 pilot：{r['tier_violations']}"
    )

    # 逐个破坏一项，每项都必须单独把 tier 打成 pilot
    for tweak, cfg, ws in [
        ({"stage": "smoke"}, main_cfg, "retfound:/w/x.pth"),
        ({"rounds": 30}, main_cfg, "retfound:/w/x.pth"),
        ({"train_fraction": 0.2}, main_cfg, "retfound:/w/x.pth"),
        ({"dry_run": True}, main_cfg, "retfound:/w/x.pth"),
        ({}, FedOSPConfig(backbone="vit_base_patch16_224", img_size=224), "retfound:/w/x.pth"),
        ({}, main_cfg, "imagenet"),
        ({}, main_cfg, "random"),
    ]:
        a = Namespace(**{**vars(main_args), **tweak})
        got = result_tier(a, cfg, ws)
        assert got["tier"] == "pilot" and got["tier_violations"], (
            f"破坏 {tweak or (cfg.backbone, ws)} 之后仍被判为 main —— 闸门漏了"
        )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'全部通过' if not failures else str(failures) + ' 项失败'}")
    sys.exit(1 if failures else 0)
