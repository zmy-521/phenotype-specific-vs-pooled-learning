# -*- coding: utf-8 -*-
r"""R44 NHANES 跨领域复现：队列构建（运行任何模型之前生成并写指纹；定义见 R44_NHANES_PROTOCOL_v1.0.md）。

定位：**跨领域程序复现**——同一冻结学习程序（引擎 xgb-1.0、同网格、同候选、同 P* 规则、同外层 5 折 × 3 seed）、
同六个表型定义，换到一般人群与长期死亡结局。不是 ICU 模型迁移；NHANES 曾用于 PGSS 早期探索（糖尿病子队列、CVD 结局），
本队列（全体成人、死亡结局）与之不同，但数据并非未触碰。

队列：NHANES 1999-2000 … 2007-2008（10 年固定窗按设计口径「2019 − 周期末年 ≥ 10」整周期纳入）；
  纳入 = 年龄 ≥ 20、完成体检（MEC 权重 > 0）、可连接死亡随访（ELIGSTAT = 1）、非妊娠（RIDEXPRG ≠ 1）。
结局（体检起算 PERMTH_EXM）：O_MORT10 = 10 年内全因死亡；O_CVD10 = 10 年内心脏病或脑血管病死亡（UCOD_LEADING 001/005）。
特征（ICU 23 项池中 NHANES 可得的 21 项 + 性别；缺 SpO2、呼吸频率）：MCV、RDW、尿素氮、平均动脉压、年龄、心率（60 秒脉搏）、
  收缩压、白细胞计数、碳酸氢盐、舒张压、血小板、血氯、血糖（血清，非空腹）、血红蛋白、血细胞比容、血肌酐（1999-2000 按 Selvin 2007 校准）、
  血钙、血钠、血钾、阴离子间隙（Na − Cl − HCO3）、性别F。舒张压记录为 0 视为缺失（预先规定）。
表型（与 ICU 主实验同定义）：女性；年龄≥65；贫血（WHO：女 Hb<12、男 <13 g/dL）；eGFR 降低（CKD-EPI 2021 免种族 <60）；
  低白蛋白（血清白蛋白 <3.5 g/dL）；肥胖（BMI ≥ 30）。白蛋白与 BMI 不在特征池内（与 ICU 相同）。
可判定：该表型的定义输入非缺失。分组单位 = 个体（SEQN）。不使用调查权重（预测程序复现，不做总体推断）。
输出：25_R44_NHANES跨领域复现/data/{X,P,OK,meta}.parquet + COHORT.json（行数、事件数、指纹）
"""
import sys, json, pathlib, hashlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.append(r"D:/NAH公共数据库/NHANES/2_代码")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

OUT = HERE.parent / "25_R44_NHANES跨领域复现" / "data"
CYCLES = ["1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008"]
RAW = ["LBXMCVSI", "LBXRDW", "LBXSBU", "BPXPLS", "LBXWBCSI", "LBXSC3SI", "LBXPLTSI", "LBXSCLSI", "LBXSGL", "LBXHGB",
       "LBXHCT", "LBXSCR", "LBDSCR", "LBXSCA", "LBXSNASI", "LBXSKSI", "LBXSAL", "BMXBMI", "BMXHT", "BMXWT", "RIDEXPRG",
       "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]
HORIZON_M = 120
CVD_CODES = {1, 5}           # UCOD_LEADING：001 心脏病、005 脑血管病


def sha16_df(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def build():
    import nhanes as nh
    d = nh.add_mortality(nh.add_derived(nh.load(RAW, cycles=CYCLES)))
    n0 = len(d)
    age = d["D_年龄"]; sex = d["RIAGENDR"]
    steps = [("全部受访者", np.ones(n0, bool)),
             ("年龄≥20", age.ge(20).to_numpy()),
             ("完成体检", d["D_WTMEC"].gt(0).to_numpy()),
             ("可连接死亡随访", d["ELIGSTAT"].eq(1).to_numpy()),
             ("非妊娠", ~d["RIDEXPRG"].eq(1).to_numpy())]
    keep = np.ones(n0, bool); flow = []
    for nm, m in steps:
        keep &= m; flow.append(dict(步骤=nm, 剩余=int(keep.sum())))
    d = d[keep].reset_index(drop=True)
    dbp = d["D_舒张压"].where(d["D_舒张压"] > 0)
    sbp = d["D_收缩压"]
    X = pd.DataFrame({
        "MCV": d.LBXMCVSI, "RDW": d.LBXRDW, "尿素氮": d.LBXSBU, "平均动脉压": (sbp + 2 * dbp) / 3, "年龄": age[keep].to_numpy(),
        "心率": d.BPXPLS, "收缩压": sbp, "白细胞计数": d.LBXWBCSI, "碳酸氢盐": d.LBXSC3SI, "舒张压": dbp, "血小板": d.LBXPLTSI,
        "血氯": d.LBXSCLSI, "血糖": d.LBXSGL, "血红蛋白": d.LBXHGB, "血细胞比容": d.LBXHCT, "血肌酐": d["D_血肌酐_校准"],
        "血钙": d.LBXSCA, "血钠": d.LBXSNASI, "血钾": d.LBXSKSI, "阴离子间隙": d.LBXSNASI - d.LBXSCLSI - d.LBXSC3SI,
        "性别F": d.RIAGENDR.eq(2).astype(float).where(d.RIAGENDR.isin([1, 2]))}).astype(float)
    uid = d.SEQN.astype(np.int64).to_numpy(); X.index = uid
    # 全部按位置取值（.to_numpy()）：d 的行号与 uid 不同，用 Series 构表会按索引错位对齐
    fem = d.RIAGENDR.eq(2).to_numpy(); hgb = d.LBXHGB.to_numpy(); egfr = d["D_eGFR_CKDEPI2021"].to_numpy()
    alb = d.LBXSAL.to_numpy(); bmi = d["D_BMI"].to_numpy()
    with np.errstate(invalid="ignore"):
        P = pd.DataFrame({"女性": fem, "年龄≥65": X["年龄"].ge(65).to_numpy(),
                          "贫血": np.where(fem, hgb < 12.0, hgb < 13.0) & ~np.isnan(hgb),
                          "eGFR降低": egfr < 60, "低白蛋白": alb < 3.5, "肥胖": bmi >= 30}, index=uid)
    hgb, egfr, alb, bmi = (pd.Series(v) for v in (hgb, egfr, alb, bmi))
    OK = pd.DataFrame({"女性": d.RIAGENDR.isin([1, 2]).to_numpy(), "年龄≥65": X["年龄"].notna().to_numpy(),
                       "贫血": (hgb.notna() & d.RIAGENDR.isin([1, 2])).to_numpy(), "eGFR降低": egfr.notna().to_numpy(),
                       "低白蛋白": alb.notna().to_numpy(), "肥胖": bmi.notna().to_numpy()}, index=uid)
    died10 = d.MORTSTAT.eq(1) & d.PERMTH_EXM.le(HORIZON_M)
    meta = pd.DataFrame({"Cycle": d.Cycle.to_numpy(), "eligible": True, "group_id": d.SEQN.astype(np.int64).astype(str).to_numpy(),
                         "hospital_id": "NA", "sex": d.RIAGENDR.to_numpy(),
                         "O_MORT10": died10.to_numpy(),
                         "O_CVD10": (died10 & d.UCOD_LEADING.isin(CVD_CODES)).to_numpy(),
                         "PERMTH_EXM": d.PERMTH_EXM.to_numpy(), "MORTSTAT": d.MORTSTAT.to_numpy()}, index=uid)
    OUT.mkdir(parents=True, exist_ok=True)
    for nm, df in (("X", X), ("P", P), ("OK", OK), ("meta", meta)):
        df.to_parquet(OUT / f"{nm}.parquet")
    summ = dict(周期=CYCLES, 纳排=flow, 最终人数=len(X), 特征=list(X.columns), 表型=list(P.columns),
                结局事件={o: int(meta[o].sum()) for o in ("O_MORT10", "O_CVD10")},
                表型可判定人数={c: int(OK[c].sum()) for c in OK.columns},
                表型阳性人数={c: int((P[c] & OK[c]).sum()) for c in P.columns},
                特征缺失率={c: round(float(X[c].isna().mean()), 4) for c in X.columns},
                随访未满10年的存活者=int((d.MORTSTAT.eq(0) & d.PERMTH_EXM.lt(HORIZON_M)).sum()),
                指纹={nm: sha16_df(df) for nm, df in (("X", X), ("P", P), ("OK", OK), ("meta", meta))},
                代码SHA=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16])
    (OUT / "COHORT.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    build()
