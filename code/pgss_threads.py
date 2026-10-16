# -*- coding: utf-8 -*-
r"""线程固定前奏 —— **必须在 import numpy 之前导入**。

为什么需要
    `pgss_validate.py` 原先在导入时设 `OMP/OPENBLAS/MKL_NUM_THREADS=1`；
    我为了消除导入副作用（GPT 第 36 轮 §2.3）把那段一起删了，于是 BLAS 回到多线程，
    **求和顺序随线程数变化**。实测（MIMIC / eGFR降低 / 住院死亡）：

        G2（47 特征、最优点数值平坦）1 线程 vs 20 线程
            逐例预测 max|Δp| = 6.4e-07，L-BFGS 迭代数 200 vs 196
            收紧 tol 到 1e-10 / 1e-12 **无效** —— sklearn 只把 tol 传给 gtol，
            ftol 硬编码为 64·eps ≈ 1.4e-14，即已收敛到双精度极限
        换算到报告用的两轨等权 Brier：
            G0 0.0e+00 ｜ **G2 5.77e-10** ｜ G-SI 6.0e-16 ｜ FR-PP 1.0e-14
        而程序间效应量约 1e-5 ~ 1e-7 ⇒ 噪声比效应小 2~4 个数量级，**结论不受影响**，
        但**逐位重跑复现**需要固定线程。

纪律
    库模块（pgss_core 等）**不得**设环境变量；只有**入口脚本**可以，
    且必须在 numpy 被导入之前。本文件就是那个唯一允许有此副作用的地方。

用法（入口脚本第一行）
    import pgss_threads  # noqa: F401  （必须在 numpy 之前）
"""
import os

_PINS = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
             NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
for _k, _v in _PINS.items():
    os.environ.setdefault(_k, _v)

PINS = dict(_PINS)


def report():
    """返回当前实际生效的线程配置，供运行指纹记录。"""
    out = {k: os.environ.get(k) for k in _PINS}
    try:
        import threadpoolctl
        out["threadpool"] = [(d["internal_api"], d["num_threads"])
                             for d in threadpoolctl.threadpool_info()]
    except Exception:
        out["threadpool"] = "threadpoolctl 不可用"
    return out


def noise_floor():
    """已实测的数值噪声底（两轨等权 Brier 尺度），供报告引用、不得手写。"""
    return {"G0": 0.0, "G2": 5.769e-10, "G-SI": 5.967e-16, "FR-PP": 1.048e-14,
            "测量条件": "MIMIC / eGFR降低 / 住院死亡；1 线程 vs 20 线程；"
                    "同一数据、同一划分、同一 tol",
            "解释": "低于 G2 的 5.8e-10 的程序间差异不可解释；"
                  "已测效应量 1e-5~1e-7，比噪声底大 2~4 个数量级"}
