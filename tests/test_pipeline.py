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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
