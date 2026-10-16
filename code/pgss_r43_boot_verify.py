# -*- coding: utf-8 -*-
r"""本地完整复算（GPT 第48轮补充裁决 §1）：只在本机读取联合重抽风险矩阵 readonly/R43_BOOT_RISKS.parquet，
用**另一条独立实现**（直接从逐 replicate 风险算相对差的分位数、三态、相对 0 方向与 max-T）复算 R43_READONLY.xlsx 中的
全部 Brier 与 log loss 对照及两套 max-T，逐项比较。

矩阵与患者级损失**不交付**；交付本脚本、复算结果 JSON（汇总量）与输入/输出 SHA。
用法：python pgss_r43_boot_verify.py   输出：readonly/R43_BOOT_VERIFY.json
"""
import sys, json, pathlib, hashlib
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

RO = HERE.parent / "24_R43_结果补齐与T1" / "readonly"
M_MAIN, MARGINS = 0.01, (0.005, 0.01, 0.02)


def sha16(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def main():
    B = pd.read_parquet(RO / "R43_BOOT_RISKS.parquet")
    X = pd.read_excel(RO / "R43_READONLY.xlsx", sheet_name=None)
    col = lambda db, oc, ph, met, arm, sc: B[f"{db}|{oc}|{ph}|{met}|{arm}|{sc}"].to_numpy(float)
    res = {}
    for met, sheet in (("brier", "5_Brier对照含新增"), ("logloss", "6_logloss对照")):
        T = X[sheet]; d_lo = d_hi = d_alo = 0.0; tri = dirn = 0
        for _, r in T.iterrows():
            def R(arm, sc):
                if sc == "bal":
                    return .5 * (col(r.库, r.结局, r.表型, met, arm, "A") + col(r.库, r.结局, r.表型, met, arm, "B"))
                return col(r.库, r.结局, r.表型, met, arm, {"pop": "P"}.get(sc, sc))
            rr, rc = R(r.参照, r.尺度), R(r.候选, r.尺度)
            rel = np.where(rr > 0, (rr - rc) / rr, np.nan); ab = rr - rc
            lo, hi = np.nanquantile(rel, .025), np.nanquantile(rel, .975)
            d_lo = max(d_lo, abs(lo - r.相对_P2_5)); d_hi = max(d_hi, abs(hi - r.相对_P97_5))
            d_alo = max(d_alo, abs(np.nanquantile(ab, .025) - r.绝对_P2_5))
            for m in MARGINS:
                t = "支持实质收益" if lo > m else "排除实质收益" if hi < m else "证据不足"
                tri += t != r[f"三态_m{m:.3f}"]
            dirn += ("全正" if lo > 0 else "全负" if hi < 0 else "跨零") != r.相对0方向
        res[met] = dict(对照行数=len(T), 相对下界最大差=d_lo, 相对上界最大差=d_hi, 绝对下界最大差=d_alo, 三态不一致=int(tri),
                        方向不一致=int(dirn))
    for met, sheet in (("brier", "7_maxT_Brier复现"), ("logloss", "8_maxT_logloss")):
        TH = X[sheet]; Z = []; sds = []
        for _, r in TH.iterrows():
            v = (1 - M_MAIN) * col(r.库, r.结局, r.表型, met, "P*", r.轨) - col(r.库, r.结局, r.表型, met, "HTFS", r.轨)
            sd = np.nanstd(v); sds.append(abs(sd - r.θ重抽SD)); Z.append((v - r.θ观察) / sd)
        c95 = float(np.nanquantile(np.nanmax(np.abs(np.vstack(Z)), axis=0), .95))
        lo = TH.θ观察 - c95 * TH.θ重抽SD; hi = TH.θ观察 + c95 * TH.θ重抽SD
        judge = np.where(lo > 0, "支持（收益>m）", np.where(hi < 0, "排除（收益<m）", "证据不足"))
        res[f"maxT_{met}"] = dict(项数=len(TH), 临界值_复算=c95, 临界值_表=float(TH["max-T临界值"].iloc[0]),
                                  重抽SD最大差=float(max(sds)), 判定不一致=int((judge != TH.同时判定_相对m.to_numpy()).sum()))
    out = dict(说明="本地完整复算：矩阵与患者级数据不出本机；本文件只含汇总量", 输入_重抽风险矩阵SHA=sha16(RO / "R43_BOOT_RISKS.parquet"),
               输入_只读补齐表SHA=sha16(RO / "R43_READONLY.xlsx"), 矩阵形状=list(B.shape), 结果=res)
    out["全部一致"] = bool(all(v.get("三态不一致", 0) == 0 and v.get("方向不一致", 0) == 0 and v.get("判定不一致", 0) == 0
                             for v in res.values()) and all(v.get("相对下界最大差", 0) < 1e-12 for v in res.values()))
    p = RO / "R43_BOOT_VERIFY.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
