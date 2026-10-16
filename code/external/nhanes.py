# -*- coding: utf-8 -*-
"""NHANES 本地库取数工具。以后做分析 import 这一个文件就够了，不用再下载。

    import sys; sys.path.append(r"E:\\公共数据库\\NHANES\\2_代码")
    import nhanes as nh

    nh.find("hemoglobin")                          # 找变量：叫什么、哪些周期有、覆盖多少人
    nh.where("LBXGH"); nh.codes("DMDEDUC2"); nh.question("BPQ020")
    df = nh.load(["LBXGH", "BMXBMI", "DIQ010"])    # 取数，自动跨周期拼好
    df = nh.add_derived(df)                        # 加派生变量（eGFR/BMI分组/糖尿病定义…）
    df = nh.add_weight(df)                         # 加多周期合并权重
    df = nh.add_mortality(df)                      # 接死亡随访
    nh.export(df, "C:/path/to/output")   # 出成品表 + 数据字典
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
IDX = ROOT / "1_数据" / "索引"
WIDE = ROOT / "1_数据" / "分周期宽表"

# 11 个互不重叠的标准区块。2017-2018 单独存在但**不在**默认口径里：
# 它的人已经包含在 2017-2020（P_* 预大流行文件）里，同时取会重复计人。
STANDARD = ["1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008",
            "2009-2010", "2011-2012", "2013-2014", "2015-2016", "2017-2020", "2021-2023"]
# 每个区块代表的年数，合并多周期权重时按这个折算
CYCLE_YEARS = {c: 2.0 for c in STANDARD}
CYCLE_YEARS["2017-2020"] = 3.2      # 2017 年 1 月 - 2020 年 3 月，官方口径 3.2 年

_cache: dict = {}


def _idx(name: str) -> pd.DataFrame:
    if name not in _cache:
        _cache[name] = pd.read_csv(IDX / f"{name}.csv")
    return _cache[name]


# ------------------------------------------------------------------ 查变量
def find(keyword: str, in_label: bool = True) -> pd.DataFrame:
    """按变量名或标签关键词搜索，返回它在哪些周期有、总共覆盖多少人。"""
    g = _idx("变量总览")
    k = keyword.upper()
    hit = g.variable.str.contains(k, case=False, na=False)
    if in_label:
        hit |= g.label.str.contains(keyword, case=False, na=False)
    return g[hit].sort_values("total_nonnull", ascending=False)[
        ["variable", "label", "n_cycles", "cycles", "files", "total_nonnull"]]


def where(variable: str) -> pd.DataFrame:
    """这个变量在每个周期分别放在哪个文件里、覆盖多少人。"""
    v = _idx("变量索引")
    return v[v.variable.str.upper() == variable.upper()][
        ["years", "file", "component", "label", "n_rows", "n_nonnull", "pct_nonnull"]]


def codes(variable: str, cycle: str | None = None) -> pd.DataFrame:
    """这个变量每个编码是什么意思（来自官方 codebook）。"""
    a = _idx("取值标签表")
    s = a[a.variable.str.upper() == variable.upper()]
    if cycle:
        s = s[s.years == cycle]
    return s[["years", "file", "code", "description", "count"]]


def question(variable: str) -> pd.DataFrame:
    """问卷原话与目标人群。"""
    d = _idx("变量说明")
    return d[d.variable.str.upper() == variable.upper()][
        ["years", "file", "sas_label", "english_text", "target"]]


# ------------------------------------------------------------------ 取数
def load(variables, cycles=None, keep_design: bool = True) -> pd.DataFrame:
    """按需从周期宽表取任意变量，跨周期纵向拼接。

    variables : 变量名列表（大小写不敏感）。某周期没有的变量自动补 NaN。
    cycles    : 默认 11 个标准区块；也可传 ["2017-2018"] 等任意周期名。
    keep_design : 自动带上 SEQN / Cycle / 权重 / PSU / STRATA / 年龄 / 性别。
    """
    variables = [v.upper() for v in ([variables] if isinstance(variables, str) else variables)]
    cycles = cycles or STANDARD
    design = ["SDMVPSU", "SDMVSTRA", "WTINT2YR", "WTMEC2YR", "WTINTPRP", "WTMECPRP",
              "WTINT4YR", "WTMEC4YR", "RIDAGEYR", "RIAGENDR"] if keep_design else []
    want = list(dict.fromkeys(variables + design))
    frames, missing = [], {}
    for c in cycles:
        p = WIDE / f"NHANES_{c}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"没有这个周期的宽表：{c}")
        d = pd.read_parquet(p)
        out = {"Cycle": d["Cycle"], "SEQN": d["SEQN"]}
        for v in want:
            if v in d.columns:
                out[v] = d[v].values
            else:
                sib = [x for x in d.columns if x.startswith(v + "@")]
                if sib:      # 同名冲突被拆开了，取覆盖最多的那份并提示
                    best = max(sib, key=lambda x: d[x].notna().sum())
                    out[v] = d[best].values
                    missing.setdefault(v, []).append(f"{c}:取自{best}")
                else:
                    out[v] = np.nan
                    if v in variables:
                        missing.setdefault(v, []).append(f"{c}:无")
        frames.append(pd.DataFrame(out))
    m = pd.concat(frames, ignore_index=True)
    m.insert(1, "CycleStart", m.Cycle.str[:4].astype(int))
    if keep_design:
        m["D_WTINT"] = m["WTINT2YR"].fillna(m["WTINTPRP"])
        m["D_WTMEC"] = m["WTMEC2YR"].fillna(m["WTMECPRP"])
    for v, notes in missing.items():
        print(f"  提示 {v}: " + "; ".join(notes))
    return m



# ------------------------------------------------------------------ 派生变量
def _num(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


def coalesce(df: pd.DataFrame, names) -> pd.Series:
    """按顺序取第一个非空值——用于同一指标在不同周期改过名的情况。"""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for n in names:
        out = out.fillna(_num(df, n))
    return out


def flag(s: pd.Series, true_vals=(1,)) -> pd.Series:
    """转成可空布尔：缺失仍是缺失，不会被当成「否」。"""
    return s.isin(true_vals).astype("boolean").mask(s.isna())


def combine(conds, avails) -> pd.Series:
    """多条判定标准合成一个可空布尔。

    任一条成立 -> True；都不成立、但至少有一条有数据 -> False；全都没数据 -> 缺失。
    直接写 `A | B` 会走 Kleene 逻辑，把「有测量值但问卷缺失」也算成缺失，
    结果 False 全部消失、算出 100% 患病率。
    """
    pos = any_ = None
    for c, a in zip(conds, avails):
        c = c.fillna(False).astype(bool)
        pos = c if pos is None else (pos | c)
        any_ = a if any_ is None else (any_ | a)
    return pos.astype("boolean").mask(~any_)


def rowmean(df: pd.DataFrame, cols) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    # 血压读数里 0 是「未测到」，不是真值
    return df[cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan).mean(axis=1)


def add_derived(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    # —— 抽样设计与权重（不替用户算多周期合并权重，见 README） ——
    d["D_WTINT"] = coalesce(d, ["WTINT2YR", "WTINTPRP"])
    d["D_WTMEC"] = coalesce(d, ["WTMEC2YR", "WTMECPRP"])
    d["D_权重口径"] = np.where(d.Cycle == "2017-2020", "2017-3月2020 预大流行权重", "2年权重")
    d["D_PSU"] = coalesce(d, ["SDMVPSU"])
    d["D_STRATA"] = coalesce(d, ["SDMVSTRA"])

    # —— 人口学 ——
    d["D_年龄"] = coalesce(d, ["RIDAGEYR"])
    sex = coalesce(d, ["RIAGENDR"])
    d["D_性别"] = sex.map({1: "男", 2: "女"})
    race = coalesce(d, ["RIDRETH3", "RIDRETH1"])
    d["D_种族"] = race.map({1: "墨西哥裔美国人", 2: "其他西班牙裔", 3: "非西班牙裔白人",
                            4: "非西班牙裔黑人", 6: "非西班牙裔亚裔", 7: "其他/多种族"})
    d["D_教育_成人"] = coalesce(d, ["DMDEDUC2"])
    d["D_收入贫困比"] = coalesce(d, ["INDFMPIR"])

    # —— 体格 ——
    d["D_身高cm"] = coalesce(d, ["BMXHT"])
    d["D_体重kg"] = coalesce(d, ["BMXWT"])
    bmi = coalesce(d, ["BMXBMI"])
    calc = d["D_体重kg"] / (d["D_身高cm"] / 100) ** 2
    d["D_BMI"] = bmi.fillna(calc)
    d["D_BMI来源"] = np.where(bmi.notna(), "BMXBMI",
                              np.where(calc.notna(), "由身高体重算得", None))
    d["D_腰围cm"] = coalesce(d, ["BMXWAIST"])
    d["D_BMI分组"] = pd.cut(d["D_BMI"], [0, 18.5, 25, 30, np.inf],
                            labels=["偏瘦", "正常", "超重", "肥胖"], right=False).astype("object")

    # —— 血压：1999-2016 听诊法(BPX)，2017 起示波法(BPXO)，两套仪器不同，分开保留再合成 ——
    a_s = rowmean(d, ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"])
    a_d = rowmean(d, ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
    o_s = rowmean(d, ["BPXOSY1", "BPXOSY2", "BPXOSY3"])
    o_d = rowmean(d, ["BPXODI1", "BPXODI2", "BPXODI3"])
    d["D_收缩压_听诊"], d["D_舒张压_听诊"] = a_s, a_d
    d["D_收缩压_示波"], d["D_舒张压_示波"] = o_s, o_d
    d["D_收缩压"] = a_s.fillna(o_s)
    d["D_舒张压"] = a_d.fillna(o_d)
    d["D_血压测量方式"] = np.where(a_s.notna(), "听诊法(BPX)",
                                   np.where(o_s.notna(), "示波法(BPXO)", None))

    # —— 实验室：跨周期改过名的指标在这里对齐 ——
    d["D_血肌酐mgdl"] = coalesce(d, ["LBXSCR", "LBDSCR"])       # 2001-2002 叫 LBDSCR
    d["D_尿素氮"] = coalesce(d, ["LBXSBU"])
    d["D_尿酸"] = coalesce(d, ["LBXSUA"])
    d["D_白蛋白"] = coalesce(d, ["LBXSAL"])
    d["D_ALT"] = coalesce(d, ["LBXSATSI"])
    d["D_AST"] = coalesce(d, ["LBXSASSI"])
    d["D_GGT"] = coalesce(d, ["LBXSGTSI"])
    d["D_总胆固醇"] = coalesce(d, ["LBXTC", "LBXSCH"])
    d["D_HDL"] = coalesce(d, ["LBDHDD", "LBXHDD", "LBDHDL"])
    d["D_LDL"] = coalesce(d, ["LBDLDL"])
    d["D_甘油三酯"] = coalesce(d, ["LBXTR", "LBXSTR"])
    d["D_空腹血糖"] = coalesce(d, ["LBXGLU"])
    d["D_HbA1c"] = coalesce(d, ["LBXGH"])
    d["D_胰岛素"] = coalesce(d, ["LBXIN"])
    d["D_CRP"] = coalesce(d, ["LBXCRP"])
    d["D_hsCRP"] = coalesce(d, ["LBXHSCRP"])
    d["D_白细胞"] = coalesce(d, ["LBXWBCSI"])
    d["D_血红蛋白"] = coalesce(d, ["LBXHGB"])
    d["D_血小板"] = coalesce(d, ["LBXPLTSI"])
    d["D_中性粒比例"] = coalesce(d, ["LBXNEPCT"])
    d["D_淋巴比例"] = coalesce(d, ["LBXLYPCT"])
    d["D_25OHD_nmolL"] = coalesce(d, ["LBXVIDMS", "LBDVIDMS"])
    d["D_尿白蛋白"] = coalesce(d, ["URXUMA"])
    d["D_尿肌酐"] = coalesce(d, ["URXUCR"])
    d["D_UACR"] = coalesce(d, ["URDACT"]).fillna(d["D_尿白蛋白"] / d["D_尿肌酐"] * 100)

    # 1999-2000 的血肌酐官方要求按 Selvin 2007 校正后再算 eGFR
    d["D_血肌酐_校准"] = np.where(d.Cycle == "1999-2000",
                                  1.013 * d["D_血肌酐mgdl"] + 0.147, d["D_血肌酐mgdl"])
    d["D_血肌酐_校准说明"] = np.where(d.Cycle == "1999-2000",
                                      "Selvin 2007: 1.013×SCr+0.147", "原始值")
    scr, age = d["D_血肌酐_校准"], d["D_年龄"]
    female = sex.eq(2)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    ratio = scr / k
    egfr = (142 * np.minimum(ratio, 1) ** a * np.maximum(ratio, 1) ** -1.200
            * 0.9938 ** age * np.where(female, 1.012, 1.0))
    d["D_eGFR_CKDEPI2021"] = pd.Series(egfr, index=d.index).where(scr.notna() & age.notna())

    # —— 常用二分定义（口径写在数据字典里，不满意可自己按原始列重算） ——
    # 这些问卷题都有跳转逻辑：上一问答「否」就不再问下一问，所以判断「有没有信息」要
    # 看**门槛那一问**答没答，而不是看这一问本身空不空，否则 False 会大批变成缺失。
    q = {x: coalesce(d, [x]) for x in
         ("DIQ010", "DIQ050", "DIQ070", "BPQ020", "BPQ040A", "BPQ050A", "BPQ150",
          "BPQ080", "BPQ100D", "BPQ101D", "SMQ020", "SMQ040", "ALQ101", "ALQ111",
          "SLD010H", "SLD012")}
    d["D_糖尿病_自报"] = flag(q["DIQ010"])
    d["D_降糖药"] = combine([q["DIQ070"].eq(1), q["DIQ050"].eq(1)],
                            [q["DIQ010"].notna(), q["DIQ070"].notna(), q["DIQ050"].notna()])
    d["D_糖尿病_综合"] = combine(
        [q["DIQ010"].eq(1), d["D_降糖药"], d["D_HbA1c"].ge(6.5), d["D_空腹血糖"].ge(126)],
        [q["DIQ010"].notna(), d["D_降糖药"].notna(), d["D_HbA1c"].notna(),
         d["D_空腹血糖"].notna()])
    d["D_高血压_自报"] = flag(q["BPQ020"])
    # 2021-2023 把降压药那一问改成了 BPQ150（BPQ040A/BPQ050A 已取消）
    d["D_降压药"] = combine([q["BPQ040A"].eq(1), q["BPQ050A"].eq(1), q["BPQ150"].eq(1)],
                            [q["BPQ020"].notna(), q["BPQ040A"].notna(),
                             q["BPQ050A"].notna(), q["BPQ150"].notna()])
    # 同理，降脂药 2021-2023 改叫 BPQ101D
    d["D_降脂药"] = combine([q["BPQ100D"].eq(1), q["BPQ101D"].eq(1)],
                            [q["BPQ080"].notna(), q["BPQ100D"].notna(), q["BPQ101D"].notna()])
    d["D_高血压_JNC7"] = combine(
        [d["D_收缩压"].ge(140), d["D_舒张压"].ge(90), d["D_降压药"]],
        [d["D_收缩压"].notna(), d["D_舒张压"].notna(), d["D_降压药"].notna()])
    d["D_高血压_2017ACC"] = combine(
        [d["D_收缩压"].ge(130), d["D_舒张压"].ge(80), d["D_降压药"]],
        [d["D_收缩压"].notna(), d["D_舒张压"].notna(), d["D_降压药"].notna()])
    d["D_曾吸烟"] = flag(q["SMQ020"])
    # SMQ040 只问抽过 100 支以上的人，分母必须是 SMQ020 答过的全体
    d["D_当前吸烟"] = combine([q["SMQ040"].isin([1, 2])], [q["SMQ020"].notna()])
    d["D_曾饮酒"] = combine([q["ALQ101"].eq(1), q["ALQ111"].eq(1)],
                            [q["ALQ101"].notna(), q["ALQ111"].notna()])
    d["D_睡眠时长h"] = coalesce(d, ["SLD012", "SLD010H"])   # 2015 起改名 SLD012
    d["D_CKD"] = combine(
        [d["D_eGFR_CKDEPI2021"].lt(60), d["D_UACR"].ge(30)],
        [d["D_eGFR_CKDEPI2021"].notna(), d["D_UACR"].notna()])
    return d

# ------------------------------------------------------------------ 权重与随访
def add_weight(df: pd.DataFrame, kind: str = "MEC", col: str = "权重_合并") -> pd.DataFrame:
    """按所选周期算多周期合并权重（CDC 规则：各周期权重 × 该周期年数 / 总年数）。

    kind="MEC" 用体检权重（做实验室/体格测量分析），"INT" 用访谈权重（只做问卷分析）。
    1999-2002 若两个周期都在选中范围内，官方推荐用 4 年权重 WTMEC4YR/WTINT4YR，
    这里会自动改用它并按 4 年折算。
    """
    d = df.copy()
    cycles = list(d.Cycle.unique())
    years = {c: CYCLE_YEARS.get(c, 2.0) for c in cycles}
    base = f"WT{kind}2YR"
    four = f"WT{kind}4YR"
    use4 = {"1999-2000", "2001-2002"} <= set(cycles) and four in d.columns
    if use4:
        years["1999-2000"] = years["2001-2002"] = 2.0     # 4 年权重覆盖两个周期，各算 2 年
    total = sum(years[c] for c in cycles)
    w = d[base] if base in d.columns else d.get(f"D_WT{kind}")
    if w is None:
        raise KeyError(f"数据里没有 {base}，先用 load(..., keep_design=True) 取")
    w = pd.to_numeric(w, errors="coerce")
    if f"WT{kind}PRP" in d.columns:
        w = w.fillna(pd.to_numeric(d[f"WT{kind}PRP"], errors="coerce"))
    if use4:
        m4 = d.Cycle.isin(["1999-2000", "2001-2002"])
        w = w.mask(m4, pd.to_numeric(d[four], errors="coerce"))
        d.loc[m4, col + "_说明"] = f"{four} × 4/{total:g}"
        d[col] = np.where(m4, w * 4 / total, w * d.Cycle.map(years) / total)
    else:
        d[col] = w * d.Cycle.map(years) / total
    note = d[col + "_说明"].fillna("") if col + "_说明" in d.columns else pd.Series("", index=d.index)
    src = lambda c: f"WT{kind}PRP" if c == "2017-2020" else base   # P_ 文件的权重列名不一样
    d[col + "_说明"] = note.where(
        note != "", d.Cycle.map(lambda c: f"{src(c)} × {years.get(c, 2):g}/{total:g}"))
    return d


def add_mortality(df: pd.DataFrame) -> pd.DataFrame:
    """接官方连接死亡随访（随访至 2019-12-31）。

    注意：2017-2020（P_*）和 2021-2023 接不上——P_ 文件用的是全新的 SEQN 段
    109263-124822，与 2017-2018 的 93703-102956 完全不重叠，官方也没发布对应的随访文件。
    要做随访到 2019 年的生存分析，请用 cycles=[..., "2017-2018"] 而不是 "2017-2020"。
    """
    mort = pd.read_parquet(ROOT / "1_数据" / "死亡随访" / "死亡随访_1999_2018.parquet")
    keep = ["Cycle", "SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
            "PERMTH_INT", "PERMTH_EXM", "UCOD_LEADING_中文"]
    out = df.merge(mort[keep], on=["Cycle", "SEQN"], how="left")
    n = out.MORTSTAT.notna().sum()
    bad = sorted(set(df.Cycle) & {"2017-2020", "2021-2023"})
    if bad:
        print(f"  注意：{', '.join(bad)} 无法连接死亡随访（SEQN 段不同 / 官方未发布）")
    print(f"  可连接 {n:,} 人，其中死亡 {int(out.MORTSTAT.fillna(0).eq(1).sum()):,}")
    return out


def decode(df: pd.DataFrame, variable: str, cycle: str | None = None,
           new_col: str | None = None) -> pd.DataFrame:
    """把某个编码变量翻成文字（用官方 codebook 的取值表）。"""
    v = variable.upper()
    c = codes(v, cycle)
    c = c[c.code.astype(str).str.match(r"^-?\d+(\.\d+)?$")]
    mp = {float(x): y for x, y in zip(c.code, c.description)}
    d = df.copy()
    d[new_col or (v + "_文字")] = pd.to_numeric(d[v], errors="coerce").map(mp)
    return d


def export(data, out_path, cycles=None, adults=False, drop_sparse: float = 0.0) -> pathlib.Path:
    """导出一张成品表：parquet + csv + csv.gz + 数据字典。产出放桌面，别放回数据库文件夹。

    data       : 已经取好的 DataFrame，或一个变量名列表（会自动 load + add_derived）
    adults     : True 只保留 20 岁及以上
    drop_sparse: 丢掉非空率低于这个百分比的列（例如 1.0 = 覆盖不到 1% 的列不要）
    """
    out_path = pathlib.Path(out_path)
    d = data.copy() if isinstance(data, pd.DataFrame) else add_derived(load(data, cycles=cycles))
    if adults:
        age = d["D_年龄"] if "D_年龄" in d.columns else d["RIDAGEYR"]
        d = d[age >= 20].reset_index(drop=True)
    if drop_sparse > 0:
        d = d[[c for c in d.columns
               if c in ("Cycle", "SEQN", "CycleStart") or
               100 * d[c].notna().mean() >= drop_sparse]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out_path.with_suffix(".parquet"), index=False, compression="zstd")
    csv = out_path.with_suffix(".csv")
    d.to_csv(csv, index=False, encoding="utf-8-sig", float_format="%.6g")
    d.to_csv(out_path.with_suffix(".csv.gz"), index=False, encoding="utf-8-sig",
             float_format="%.6g", compression="gzip")

    vi = _idx("变量索引")
    vi = vi[vi.label.notna() & (vi.label != "")]
    lab = vi.groupby("variable").label.agg(lambda s: s.mode().iloc[0]).to_dict()
    dic = pd.DataFrame({"列名": list(d.columns)})
    dic["含义"] = dic.列名.str.replace(r"@.*", "", regex=True).map(lab).fillna("")
    dic["非空数"] = [int(d[c].notna().sum()) for c in d.columns]
    dic["非空率%"] = (dic.非空数 / len(d) * 100).round(1)
    dic["类型"] = [str(d[c].dtype) for c in d.columns]
    dic.to_csv(out_path.with_name(out_path.stem + "_数据字典.csv"),
               index=False, encoding="utf-8-sig")
    gz = out_path.with_suffix(".csv.gz")
    print(f"  {len(d):,} 行 × {d.shape[1]} 列 -> {out_path.parent}")
    print(f"    {csv.name} {csv.stat().st_size/1024**2:.1f}MB | "
          f"{gz.name} {gz.stat().st_size/1024**2:.1f}MB | .parquet | _数据字典.csv")
    return csv


def nhanes3(dataset: str | None = None):
    """NHANES III (1988-1994)。不传参数则列出全部 80 个数据集。"""
    if dataset is None:
        return pd.read_csv(ROOT / "1_数据" / "索引" / "文件索引_NHANES3.csv")
    return pd.read_parquet(ROOT / "1_数据" / "NHANES3" / f"{dataset}.parquet")


def raw(cycle: str, file: str, cleaned: bool = True) -> pd.DataFrame:
    """直接读某个单独的文件。一人多行的 137 个文件（膳食明细、处方用药、DXA 多重插补等）
    不在分周期宽表里，只能这样读。cleaned=False 读 3_存档 里未做缺失码处理的原值版。"""
    sub = ROOT / ("1_数据/分文件" if cleaned else "3_存档/未清洗parquet")
    return pd.read_parquet(sub / cycle / f"{file.upper()}.parquet")
