# -*- coding: utf-8 -*-
r"""R45 目标适配器的单元测试（不读真实数据）。每条断言配能失败的对照。

T1 bal 目标的早停权重与原引擎逐位相同；pop 目标为全 1；B 目标 A 轨为 0、B 轨之和为 1；专属臂三目标都与原引擎相同。
T2 activate 后引擎内部（fit_booster 调用的模块级名字）确实用到目标化函数：小样本合成数据跑一次合并拟合，计数 > 0；
   deactivate 后恢复原函数（对照：未 activate 时计数为 0）。
T3 score_target：bal = brier_bal；pop = 全体均值；B = B 轨均值；三者在 A/B 损失不同的构造数据上互不相等（对照）。
T4 早停留出缺一轨时，pop 与 B 目标照原引擎抛 FoldFailure（可执行集合相同）。
输出：27_R45_外部复现与机制核验/target/R45_TARGET_TESTS.json
"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pgss_xgb_engine as EN
import pgss_r45_target as TG


def main():
    res = []
    orig = EN.es_metric_weights
    rng = np.random.default_rng(1)
    z = rng.random(500) < 0.2
    # T1
    TG.activate("bal"); w_bal = EN.es_metric_weights(z, True); TG.deactivate()
    res.append(dict(测试="T1 bal 早停权重 = 原引擎", 通过=bool(np.array_equal(w_bal, orig(z, True)))))
    TG.activate("pop"); w_pop = EN.es_metric_weights(z, True); w_trk = EN.es_metric_weights(z, False); TG.deactivate()
    res.append(dict(测试="T1 pop 早停权重全 1，且 ≠ bal（对照）", 通过=bool(np.all(w_pop == 1) and not np.allclose(w_pop / w_pop.sum(), w_bal / w_bal.sum()))))
    res.append(dict(测试="T1 专属臂权重 = 原引擎", 通过=bool(np.array_equal(w_trk, orig(z, False)))))
    TG.activate("B"); w_B = EN.es_metric_weights(z, True); TG.deactivate()
    res.append(dict(测试="T1 B 目标：A 轨 0、B 轨和为 1", 通过=bool(np.all(w_B[z] == 0) and np.isclose(w_B[~z].sum(), 1.0))))
    # T2：引擎内部是否用到替换后的函数
    n = 400
    X = rng.normal(size=(n, 3)); zz = rng.random(n) < 0.3; y = (rng.random(n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    g = np.arange(n); uid = np.arange(n)
    TG.CALLS["es_pooled"] = 0
    EN.set_namespace(test="r45-T2")
    EN.fit_booster(X, y, np.ones(n), zz, g, uid, dict(max_depth=2, min_child_weight=1), pooled=True, role="t2")
    c0 = TG.CALLS["es_pooled"]
    TG.activate("pop")
    EN.fit_booster(X, y, np.ones(n), zz, g, uid, dict(max_depth=2, min_child_weight=1), pooled=True, role="t2")
    c1 = TG.CALLS["es_pooled"]; TG.deactivate()
    res.append(dict(测试="T2 activate 后 fit_booster 调用目标化早停权重（未 activate 计数 0 为对照）",
                    通过=bool(c0 == 0 and c1 >= 1), 未activate计数=c0, activate计数=c1))
    res.append(dict(测试="T2 deactivate 恢复原函数", 通过=bool(EN.es_metric_weights is orig)))
    # T3
    yy = np.array([1, 0, 1, 0, 1, 0, 0, 0], float); pp = np.array([.9, .1, .2, .8, .6, .5, .3, .1]); z3 = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    out = {}
    for t in ("bal", "pop", "B"):
        TG.activate(t); out[t] = TG.score_target(yy, pp, z3); TG.deactivate()
    l = (yy - pp) ** 2
    ok = (out["bal"] == EN.brier_bal(yy, pp, z3) and np.isclose(out["pop"], l.mean()) and np.isclose(out["B"], l[~z3].mean())
          and len({round(v, 12) for v in out.values()}) == 3)
    res.append(dict(测试="T3 三目标评分定义正确且互不相等", 通过=bool(ok), 值=out))
    # T4
    zz4 = np.zeros(50, bool)
    flags = {}
    for t in ("bal", "pop", "B"):
        TG.activate(t)
        try:
            EN.es_metric_weights(zz4, True); flags[t] = False
        except EN.FoldFailure:
            flags[t] = True
        TG.deactivate()
    res.append(dict(测试="T4 缺一轨时三目标都抛 FoldFailure", 通过=all(flags.values()), 明细=flags))
    summ = dict(全部通过=all(r["通过"] for r in res), 测试=res, code_sha=TG.code_sha())
    TG.OUT.mkdir(parents=True, exist_ok=True)
    (TG.OUT / "R45_TARGET_TESTS.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return summ


if __name__ == "__main__":
    sys.exit(0 if main()["全部通过"] else 1)
