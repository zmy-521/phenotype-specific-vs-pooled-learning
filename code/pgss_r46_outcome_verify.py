# -*- coding: utf-8 -*-
r"""R46 结局核实（方案 28_R46_最终核验与T2/outcome_fix/R46_OUTCOME_FIX_PROTOCOL_v1.0.md §1–§2）。只读原始表与 R44/R45 冻结数据，不拟合。

1. NWICU：R45 合格队列中 discharge_location = UNKNOWN 的人次，按方案 §1.3 的证据规则分为 K1/K2/K3/K3a/K4；
   输出逐人次住院死亡可评价性与 v2 标签（本地，含 stay_id）以及无标识汇总。
2. NHANES：10 年内死亡但 UCOD_LEADING 缺失者 → O_CVD10 未知；列出其所在且已纳入的 O_CVD10 任务（只判断是否在任务人群中）。
输出：
  本地  28_R46_最终核验与T2/nwicu_v2/_local_only/MORT_STATUS.parquet、nhanes_v2/_local_only/CVD_STATUS.parquet
  汇总  28_R46_最终核验与T2/outcome_fix/R46_OUTCOME_VERIFY.xlsx、.json（无标识）
"""
import sys, json, pathlib, hashlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.append(r"D:/NAH公共数据库/NHANES/2_代码")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
R46 = ROOT / "28_R46_最终核验与T2"
OF = R46 / "outcome_fix"
SRC = pathlib.Path(r"E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data")
NW1 = ROOT / "27_R45_外部复现与机制核验" / "nwicu" / "data"
R44D = ROOT / "25_R44_NHANES跨领域复现" / "data"
INC = ROOT / "27_R45_外部复现与机制核验" / "target" / "INCLUDED_TASKS.json"
PHENOS = ["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"]
H24 = pd.Timedelta("24h")


def nwicu():
    meta = pd.read_parquet(NW1 / "meta.parquet"); P = pd.read_parquet(NW1 / "P.parquet"); OK = pd.read_parquet(NW1 / "OK.parquet")
    st = pd.read_csv(SRC / "nw_icu/icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "intime"], parse_dates=["intime"])
    a = pd.read_csv(SRC / "nw_hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime", "deathtime"])
    p = pd.read_csv(SRC / "nw_hosp/patients.csv.gz", parse_dates=["dod"])
    c = meta.join(st.set_index("stay_id"), how="left")
    assert c.hadm_id.notna().all() and (c.subject_id.astype(str) == c.group_id).all(), "stay→hadm/subject 映射失败"
    c = c.merge(a[["hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"]], on="hadm_id", how="left",
                validate="many_to_one").set_index(meta.index)
    c = c.join(p.set_index("subject_id")[["dod"]], on="subject_id")
    c["lm"] = c.intime + H24
    U = c[c.discharge_location.eq("UNKNOWN")].copy()
    # 后续入院：同一患者、其他住院、admittime ≥ 本次 dischtime
    oth = a[["subject_id", "hadm_id", "admittime"]].rename(columns={"hadm_id": "h2", "admittime": "adm2"})
    j = U.reset_index()[["stay_id", "subject_id", "hadm_id", "dischtime"]].merge(oth, on="subject_id")
    j = j[(j.h2 != j.hadm_id) & (j.adm2 >= j.dischtime)]
    nxt = j.groupby("stay_id").adm2.min()
    U["E_next"] = U.index.isin(nxt.index)
    U["距后续入院_天"] = (nxt.reindex(U.index) - U.dischtime).dt.total_seconds() / 86400
    dd = U.dischtime.dt.normalize()
    U["E_dt_le_td"] = U.deathtime.notna() & (U.deathtime <= U.dischtime)
    U["E_dod_lt_dd"] = U.dod.notna() & (U.dod < dd)
    D = (U.deathtime.notna() & (U.deathtime <= U.dischtime + H24)) | (U.dod.notna() & (U.dod <= dd + pd.Timedelta("1D"))) \
        | U.hospital_expire_flag.eq(1)
    L = (U.deathtime.notna() & (U.deathtime > U.dischtime + H24)) | (U.dod.notna() & (U.dod >= dd + pd.Timedelta("2D"))) | U.E_next
    K1 = U.E_dt_le_td | U.E_dod_lt_dd
    cls = np.select([K1, D & L, L & ~D, D & ~L], ["K1", "K4", "K2", "K3a"], "K3")
    U["类别"] = cls
    U["依据_E_dt晚于出院24h"] = U.deathtime.notna() & (U.deathtime > U.dischtime + H24)
    U["依据_E_dod晚于出院2天"] = U.dod.notna() & (U.dod >= dd + pd.Timedelta("2D"))
    U["D"] = D; U["L"] = L
    k1_event = (U.类别 == "K1") & U.deathtime.notna() & (U.deathtime > U.lm)
    # v2 标签与可评价性（全部合格人次）
    S = pd.DataFrame(index=meta.index)
    S["UNKNOWN"] = meta.discharge_location.eq("UNKNOWN")
    S["类别"] = U.类别.reindex(S.index)
    S["住院死亡可评价"] = ~S.UNKNOWN
    S.loc[U.index[(U.类别 == "K2") | k1_event], "住院死亡可评价"] = True
    y = meta.OUT_HOSP_MORT.astype(float).copy()
    y[k1_event[k1_event].index] = 1.0
    y[~S.住院死亡可评价] = np.nan
    S["OUT_HOSP_MORT_v2"] = y
    S["标签改变"] = S.住院死亡可评价 & (S.OUT_HOSP_MORT_v2 != meta.OUT_HOSP_MORT.astype(float))
    loc = R46 / "nwicu_v2" / "_local_only"; loc.mkdir(parents=True, exist_ok=True)
    S.to_parquet(loc / "MORT_STATUS.parquet")
    # 汇总（无标识）
    cnt = pd.Series(U.类别).value_counts().reindex(["K1", "K2", "K3", "K3a", "K4"], fill_value=0)
    K2 = U[U.类别 == "K2"]
    ev = pd.DataFrame([dict(项="UNKNOWN 人次（合格队列）", N=len(U)),
                       *[dict(项=f"类别 {k}", N=int(v)) for k, v in cnt.items()],
                       dict(项="K1 中按既定规则记事件（E_dt > landmark）", N=int(k1_event.sum())),
                       dict(项="K2 依据：后续入院", N=int(K2.E_next.sum())),
                       dict(项="K2 依据：deathtime 晚于出院 24 h", N=int(K2.依据_E_dt晚于出院24h.sum())),
                       dict(项="K2 依据：dod 晚于出院日 ≥ 2 天", N=int(K2.依据_E_dod晚于出院2天.sum())),
                       dict(项="K2 只有一种依据", N=int((K2[["E_next", "依据_E_dt晚于出院24h", "依据_E_dod晚于出院2天"]].sum(axis=1) == 1).sum())),
                       dict(项="UNKNOWN 中 hospital_expire_flag = 1", N=int(U.hospital_expire_flag.eq(1).sum())),
                       dict(项="UNKNOWN 中 deathtime 非空", N=int(U.deathtime.notna().sum())),
                       dict(项="UNKNOWN 中 dod 非空", N=int(U.dod.notna().sum())),
                       dict(项="住院死亡不可评价（排除）合计", N=int((~S.住院死亡可评价).sum())),
                       dict(项="住院死亡标签改变（可评价者中）", N=int(S.标签改变.sum())),
                       dict(项="排除人次涉及患者数", N=int(meta.group_id[~S.住院死亡可评价].nunique())),
                       dict(项="其中该患者在合格队列中的全部人次都被排除", N=int((meta.assign(ex=~S.住院死亡可评价).groupby("group_id").ex.all()).sum()))])
    gaps = pd.DataFrame([dict(量="K2 距后续入院（天）", 人次=int(K2.E_next.sum()),
                              中位=float(K2.距后续入院_天.median()), P5=float(K2.距后续入院_天.quantile(.05)),
                              P95=float(K2.距后续入院_天.quantile(.95)), 最小=float(K2.距后续入院_天.min()))])
    tk = []
    for ph in PHENOS:
        m = OK[ph].to_numpy(bool); z = P[ph].to_numpy(bool) & m; ex = (~S.住院死亡可评价).to_numpy() & m
        yv = meta.OUT_HOSP_MORT.to_numpy(bool)
        tk.append(dict(结局="OUT_HOSP_MORT", 表型=ph, R45任务人次=int(m.sum()), 排除人次=int(ex.sum()), 排除_A轨=int((ex & z).sum()),
                       排除_B轨=int((ex & ~z & m).sum()), v2任务人次=int((m & ~ex).sum()), R45事件=int(yv[m].sum()),
                       v2事件=int(np.nansum(S.OUT_HOSP_MORT_v2.to_numpy()[m & ~ex])), v2_A轨事件=int(np.nansum(S.OUT_HOSP_MORT_v2.to_numpy()[z & ~ex])),
                       v2_A轨人次=int((z & ~ex).sum())))
    TK = pd.DataFrame(tk)
    rerun = int(cnt[["K1", "K3", "K3a", "K4"]].sum() - k1_event.sum()) > 0 or int(S.标签改变.sum()) > 0
    return dict(EV=ev, GAPS=gaps, TK=TK, 需要重跑=bool(rerun), 状态SHA=hashlib.sha256(
        pd.util.hash_pandas_object(S, index=True).values.tobytes()).hexdigest()[:16])


def nhanes():
    import nhanes as nh
    meta = pd.read_parquet(R44D / "meta.parquet"); P = pd.read_parquet(R44D / "P.parquet"); OK = pd.read_parquet(R44D / "OK.parquet")
    seqn = meta.index.to_numpy().astype(np.int64)
    mort = pd.read_parquet(nh.ROOT / "1_数据" / "死亡随访" / "死亡随访_1999_2018.parquet")
    m = mort[mort.SEQN.isin(seqn) & mort.Cycle.isin(meta.Cycle.unique())].drop_duplicates(["Cycle", "SEQN"]).set_index("SEQN").reindex(seqn)
    died10 = m.MORTSTAT.eq(1) & m.PERMTH_EXM.le(120)
    unk = (died10 & m.UCOD_LEADING.isna()).to_numpy()
    assert np.array_equal(died10.to_numpy(), meta.O_MORT10.to_numpy()), "与 R44 O_MORT10 不一致"
    S = pd.DataFrame({"O_CVD10可评价": ~unk, "O_CVD10_v2": np.where(unk, np.nan, meta.O_CVD10.astype(float))}, index=meta.index)
    loc = R46 / "nhanes_v2" / "_local_only"; loc.mkdir(parents=True, exist_ok=True)
    S.to_parquet(loc / "CVD_STATUS.parquet")
    inc = [tuple(t) for t in json.loads(INC.read_text(encoding="utf-8"))["纳入任务"]]
    rows = []
    for ph in PHENOS:
        intask = bool((OK[ph].to_numpy(bool) & unk).any())
        rows.append(dict(结局="O_CVD10", 表型=ph, R44已纳入=("O_CVD10", ph) in inc, 未知者在任务人群中=intask,
                         未知者所在轨=("A" if (P[ph].to_numpy(bool) & unk & OK[ph].to_numpy(bool)).any() else "B") if intask else "",
                         需要重跑=bool(intask and ("O_CVD10", ph) in inc)))
    T = pd.DataFrame(rows)
    return dict(N未知=int(unk.sum()), TK=T, 重跑任务=[["O_CVD10", r.表型] for r in T.itertuples() if r.需要重跑],
                状态SHA=hashlib.sha256(pd.util.hash_pandas_object(S, index=True).values.tobytes()).hexdigest()[:16])


def main():
    OF.mkdir(parents=True, exist_ok=True)
    a = nwicu(); b = nhanes()
    summ = dict(NWICU=dict(需要重跑=a["需要重跑"], 状态SHA=a["状态SHA"], 计数=a["EV"].set_index("项").N.to_dict()),
                NHANES=dict(未知死因人数=b["N未知"], 重跑任务=b["重跑任务"], 状态SHA=b["状态SHA"]),
                代码SHA=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16])
    with pd.ExcelWriter(OF / "R46_OUTCOME_VERIFY.xlsx", engine="openpyxl") as xw:
        a["EV"].to_excel(xw, sheet_name="1_NWICU_UNKNOWN核实", index=False)
        a["GAPS"].to_excel(xw, sheet_name="2_NWICU_K2时间间隔", index=False)
        a["TK"].to_excel(xw, sheet_name="3_NWICU_住院死亡任务变化", index=False)
        b["TK"].to_excel(xw, sheet_name="4_NHANES_CVD未知死因", index=False)
    (OF / "R46_OUTCOME_VERIFY.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    for t in (a["EV"], a["GAPS"], a["TK"], b["TK"]):
        print(t.to_string())


if __name__ == "__main__":
    main()
