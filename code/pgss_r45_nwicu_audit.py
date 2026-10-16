# -*- coding: utf-8 -*-
r"""R45 NWICU 外部数据准备：覆盖审计与候选可比任务清单（GPT 第48轮 §5：数据可用时只做字段映射与候选任务清单，**不计算外部模型性能**）。

按 MIMIC 主实验的同一口径（pgss_r35_cohort / data v5）统计，不拟合任何模型：
  队列：每次住院的首段 ICU；landmark = 入室 + 24 h；landmark 时仍在 ICU；年龄 ≥ 18（anchor_age + 入院年 − anchor_year）；landmark 前未死亡。
  特征窗口（v5）：charttime ∈ [intime, landmark) 且 storetime < landmark；记录每项在窗内有值的入住比例。
  身高/体重：storetime < landmark 的记录中距 intime 最近者（v5 基线体型）；BMI 由此计算（单位：英寸、盎司 → m、kg）。
  结局候选（**待与 GPT 冻结，不在此自行选定**）：
    HOSP_MORT_a：出院去向 = EXPIRED，死亡时点（deathtime，缺失用 dischtime）晚于 landmark；
    HOSP_MORT_b：出院去向 ∈ {EXPIRED, EXPIRED - HOSPICE}，同上；
    HOSP_MORT_c：deathtime 不晚于出院 + 1 天（不看去向），且晚于 landmark；
    （hospital_expire_flag 只在 89 次 ICU 住院为 1，与去向 EXPIRED 不一致，不作候选）
    LOS72：icustays.los > 3 天。
  表型：与 ICU 主实验同定义（女性、年龄≥65、贫血、eGFR<60、白蛋白<3.5、BMI≥30），基于 landmark 前最后一次可用值。
输出：26_R45_NWICU外部数据准备/R45_NWICU_AUDIT.xlsx、R45_NWICU_AUDIT.json（只含汇总计数，无患者标识）
"""
import sys, json, pathlib, hashlib, time
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

SRC = pathlib.Path(r"E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data")
OUT = HERE.parent / "26_R45_NWICU外部数据准备"
VITALS = {"心率": [320045], "收缩压": [320179, 320050], "舒张压": [320180, 320051], "呼吸频率": [320210], "SpO2": [320277],
          "体温F": [323761]}
BODY = {"身高_in": 326707, "体重_oz": 326531}
LABS = {"血红蛋白": [100007], "血细胞比容": [100006], "MCV": [100017], "RDW": [100008], "白细胞计数": [100016], "血小板": [100014],
        "尿素氮": [100004], "血肌酐": [100002], "血钠": [100010], "血钾": [100011], "血氯": [100012], "碳酸氢盐": [100013],
        "血糖": [100001], "血钙": [100003], "阴离子间隙": [100040, 100041], "白蛋白": [100021]}


def last_in_window(ev, st, items, id_col="stay_id"):
    """窗口 [intime, lm) 内、storetime < lm 的最后一次值（同时点多条取均值，与 v4/v5 规则同向）。"""
    e = ev[ev.itemid.isin(items)].merge(st[[id_col, "intime", "lm"]], on=id_col)
    e = e[(e.charttime >= e.intime) & (e.charttime < e.lm) & (e.storetime < e.lm) & e.valuenum.notna()]
    if e.empty:
        return pd.Series(dtype=float)
    last = e.groupby(id_col).charttime.transform("max")
    return e[e.charttime == last].groupby(id_col).valuenum.mean()


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(SRC / "nw_hosp/patients.csv.gz")
    a = pd.read_csv(SRC / "nw_hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime", "deathtime"])
    s = pd.read_csv(SRC / "nw_icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
    flow = [dict(步骤="ICU 入住记录", 人次=len(s))]
    st = s.sort_values(["hadm_id", "intime"]).drop_duplicates("hadm_id").copy()
    flow.append(dict(步骤="每次住院首段 ICU", 人次=len(st)))
    st["lm"] = st.intime + pd.Timedelta("24h")
    st = st.merge(a, on=["subject_id", "hadm_id"], how="left").merge(p, on="subject_id", how="left")
    st["age"] = st.anchor_age + (st.admittime.dt.year - st.anchor_year)
    st["death_ts"] = st.deathtime.fillna(st.dischtime.where(st.discharge_location.isin(["EXPIRED", "EXPIRED - HOSPICE"])))
    dead_before_lm = st.death_ts.notna() & (st.death_ts <= st.lm) & (st.death_ts <= st.dischtime + pd.Timedelta("1D"))
    m_icu = st.outtime > st.lm; m_adult = st.age >= 18
    st["eligible"] = m_icu & m_adult & ~dead_before_lm
    flow += [dict(步骤="landmark 时仍在 ICU", 人次=int(m_icu.sum())), dict(步骤="且年龄≥18", 人次=int((m_icu & m_adult).sum())),
             dict(步骤="且 landmark 前未死亡（合格）", 人次=int(st.eligible.sum()))]
    E = st[st.eligible].copy()
    after = E.death_ts.notna() & (E.death_ts > E.lm)
    E["HOSP_MORT_a"] = E.discharge_location.eq("EXPIRED") & after
    E["HOSP_MORT_b"] = E.discharge_location.isin(["EXPIRED", "EXPIRED - HOSPICE"]) & after
    E["HOSP_MORT_c"] = E.deathtime.notna() & (E.deathtime <= E.dischtime + pd.Timedelta("1D")) & (E.deathtime > E.lm)
    E["LOS72"] = E.los > 3
    oc = pd.DataFrame([dict(结局=c, 事件=int(E[c].sum()), 合格人次=len(E), 事件率=float(E[c].mean()))
                       for c in ("HOSP_MORT_a", "HOSP_MORT_b", "HOSP_MORT_c", "LOS72")])
    # 特征（分块读，只保留需要的 itemid）
    keep_c = sum(VITALS.values(), []) + list(BODY.values())
    ch = pd.concat([c[c.itemid.isin(keep_c)] for c in pd.read_csv(SRC / "nw_icu/chartevents.csv.gz", chunksize=2_000_000,
                    usecols=["stay_id", "itemid", "charttime", "storetime", "valuenum"], parse_dates=["charttime", "storetime"])])
    keep_l = sum(LABS.values(), [])
    lb = pd.concat([c[c.itemid.isin(keep_l)] for c in pd.read_csv(SRC / "nw_hosp/labevents.csv.gz", chunksize=2_000_000,
                    usecols=["hadm_id", "itemid", "charttime", "storetime", "valuenum"], parse_dates=["charttime", "storetime"])])
    F = pd.DataFrame(index=E.stay_id)
    for k, items in VITALS.items():
        F[k] = last_in_window(ch, E, items)
    lbs = lb.merge(E[["hadm_id", "stay_id"]], on="hadm_id")
    for k, items in LABS.items():
        F[k] = last_in_window(lbs, E, items)
    # 基线体型：storetime < lm 中距 intime 最近
    for k, it in BODY.items():
        b = ch[ch.itemid == it].merge(E[["stay_id", "intime", "lm"]], on="stay_id")
        b = b[(b.storetime < b.lm) & b.valuenum.notna()].copy()
        b["d"] = (b.charttime - b.intime).abs()
        F[k] = b.sort_values(["stay_id", "d"]).drop_duplicates("stay_id").set_index("stay_id").valuenum
    F["平均动脉压"] = (F["收缩压"] + 2 * F["舒张压"]) / 3
    F["BMI"] = (F["体重_oz"] * 0.0283495) / (F["身高_in"] * 0.0254) ** 2
    cov = pd.DataFrame([dict(变量=c, 窗内有值比例=float(F[c].notna().mean()), 中位数=float(F[c].median()),
                             P1=float(F[c].quantile(.01)), P99=float(F[c].quantile(.99))) for c in F.columns])
    # 表型
    Ei = E.set_index("stay_id")
    fem = Ei.gender.eq("F"); hgb = F["血红蛋白"]; scr = F["血肌酐"]; age = Ei.age
    k_ = np.where(fem, 0.7, 0.9); a_ = np.where(fem, -0.241, -0.302); r = scr / k_
    egfr = 142 * np.minimum(r, 1) ** a_ * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(fem, 1.012, 1.0)
    PH = pd.DataFrame({"女性": fem, "年龄≥65": age >= 65, "贫血": np.where(fem, hgb < 12, hgb < 13),
                       "eGFR降低": egfr < 60, "低白蛋白": F["白蛋白"] < 3.5, "肥胖": F["BMI"] >= 30})
    OK = pd.DataFrame({"女性": Ei.gender.isin(["F", "M"]), "年龄≥65": age.notna(), "贫血": hgb.notna() & Ei.gender.isin(["F", "M"]),
                       "eGFR降低": egfr.notna(), "低白蛋白": F["白蛋白"].notna(), "肥胖": F["BMI"].notna() & F["BMI"].between(10, 80)})
    tasks = []
    for o in ("HOSP_MORT_a", "HOSP_MORT_b", "HOSP_MORT_c", "LOS72"):
        y = Ei[o]
        for ph in PH.columns:
            m = OK[ph]; z = PH[ph].fillna(False).astype(bool) & m
            tasks.append(dict(结局=o, 表型=ph, 可判定人次=int(m.sum()), A轨人次=int(z.sum()), A轨比例=float(z[m].mean()),
                              事件=int(y[m].sum()), A轨事件=int(y[z].sum()), B轨事件=int(y[m & ~z].sum()),
                              唯一患者=int(Ei.subject_id[m].nunique())))
    TK = pd.DataFrame(tasks)
    summ = dict(数据="NWICU 0.1.0（PhysioNet，认证访问；SHA256 已按官方清单校验）", 合格人次=len(E), 唯一患者=int(E.subject_id.nunique()),
                特征可得=(f"{sum(c in F for c in ['心率','收缩压','舒张压','平均动脉压','呼吸频率','SpO2'] + [k for k in LABS if k != '白蛋白']) + 2} 项"
                          "（ICU 23 项池：6 项体征含平均动脉压、15 项化验、年龄、性别；白蛋白与 BMI 只用于表型）"),
                生命体征窗内覆盖=float(F["心率"].notna().mean()),
                说明="只做覆盖审计与候选任务清单，未拟合任何模型、未计算任何外部性能；结局定义待与 GPT 冻结",
                秒=round(time.time() - t0))
    with pd.ExcelWriter(OUT / "R45_NWICU_AUDIT.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([summ]).T.reset_index().to_excel(xw, sheet_name="0_汇总", index=False)
        pd.DataFrame(flow).to_excel(xw, sheet_name="1_队列流程", index=False)
        oc.to_excel(xw, sheet_name="2_结局候选", index=False)
        cov.to_excel(xw, sheet_name="3_特征覆盖", index=False)
        TK.to_excel(xw, sheet_name="4_候选任务", index=False)
    (OUT / "R45_NWICU_AUDIT.json").write_text(json.dumps(dict(summ, 队列流程=flow, 结局候选=oc.to_dict("records")),
                                                         ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1)); print(pd.DataFrame(flow).to_string()); print(oc.to_string())
    print(cov.to_string()); print(TK.to_string())


if __name__ == "__main__":
    main()
