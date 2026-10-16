# -*- coding: utf-8 -*-
r"""R45 第一部分（续）：R44 材料补齐（GPT 第 49 轮 §5）。不重新训练，只读 R44 队列与本地 NHANES 死亡随访文件。

1. 死亡随访字段的实际缺失数（R44 队列 23,513 人）：MORTSTAT、PERMTH_EXM、UCOD_LEADING（死亡者），
   以及会被逻辑表达式静默编码为阴性的组合（死亡但随访月数缺失；10 年内死亡但死因缺失）。
2. 表型定义表的更正：贫血 = 女性 Hb < 12、男性 Hb < 13 g/dL（R44 表 A3 只写了 12，是展示不完整；代码 pgss_r44_nhanes_cohort.py 一直用两个切点）——
   这里从代码源文本中逐项取出切点，避免再次手写。
3. 外部辅助模块 nhanes.py 的登记：路径、SHA、R44 实际调用的函数（load、add_derived、add_mortality）及其源码行数；随包交付该模块。
4. O_CVD10×低白蛋白 不可执行单元的逐单元原因（来自冻结前路径审计）。
输出：27_R45_外部复现与机制核验/r44supp/R45_R44SUPP.xlsx、.json
"""
import sys, json, pathlib, hashlib, inspect, re
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.append(r"D:/NAH公共数据库/NHANES/2_代码")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

R44 = HERE.parent / "25_R44_NHANES跨领域复现"
OUT = HERE.parent / "27_R45_外部复现与机制核验" / "r44supp"
NH = pathlib.Path(r"D:/NAH公共数据库/NHANES/2_代码/nhanes.py")
HORIZON_M = 120


def main():
    import nhanes as nh
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(R44 / "data" / "meta.parquet")
    seqn = meta.index.to_numpy().astype(np.int64)
    mort = pd.read_parquet(nh.ROOT / "1_数据" / "死亡随访" / "死亡随访_1999_2018.parquet")
    m = mort[mort.SEQN.isin(seqn) & mort.Cycle.isin(meta.Cycle.unique())].drop_duplicates(["Cycle", "SEQN"])
    m = m.set_index("SEQN").reindex(seqn)
    dead = m.MORTSTAT.eq(1)
    died10 = dead & m.PERMTH_EXM.le(HORIZON_M)
    rows = [dict(项="队列人数", N=len(m)),
            dict(项="ELIGSTAT ≠ 1（应为 0：纳入条件）", N=int((~m.ELIGSTAT.eq(1)).sum())),
            dict(项="MORTSTAT 缺失", N=int(m.MORTSTAT.isna().sum())),
            dict(项="PERMTH_EXM 缺失（全体）", N=int(m.PERMTH_EXM.isna().sum())),
            dict(项="死亡（MORTSTAT=1）", N=int(dead.sum())),
            dict(项="死亡但 PERMTH_EXM 缺失（会被 10 年窗静默判为非事件）", N=int((dead & m.PERMTH_EXM.isna()).sum())),
            dict(项="死亡者 UCOD_LEADING 缺失", N=int((dead & m.UCOD_LEADING.isna()).sum())),
            dict(项="10 年内死亡", N=int(died10.sum())),
            dict(项="10 年内死亡但 UCOD_LEADING 缺失（会被心脑血管死亡静默判为非事件）", N=int((died10 & m.UCOD_LEADING.isna()).sum())),
            dict(项="存活且随访 < 120 个月（10 年窗不完整）", N=int((m.MORTSTAT.eq(0) & m.PERMTH_EXM.lt(HORIZON_M)).sum())),
            dict(项="与 R44 meta 的 O_MORT10 逐例一致", N=int((died10.to_numpy() == meta.O_MORT10.to_numpy()).sum()))]
    MISS = pd.DataFrame(rows)
    # 定义表：从队列代码源文本取切点
    src = (HERE / "pgss_r44_nhanes_cohort.py").read_text(encoding="utf-8")
    hb = re.search(r'"贫血": np\.where\(fem, hgb < ([\d.]+), hgb < ([\d.]+)\)', src)
    defs = pd.DataFrame([
        dict(表型="贫血", 定义=f"女性 Hb < {hb.group(1)} g/dL；男性 Hb < {hb.group(2)} g/dL（WHO）", 女性切点=float(hb.group(1)), 男性切点=float(hb.group(2)),
             代码出处="pgss_r44_nhanes_cohort.py（正则取自源文本）", R44表A3原写法="只写了 12（展示不完整，代码一直用两个切点）"),
        dict(表型="心脑血管死亡 O_CVD10", 定义="10 年固定窗内 UCOD_LEADING ∈ {001 心脏病, 005 脑血管病} 的死亡；其他原因死亡记 0",
             解释="固定 10 年内该类死亡发生的概率（含竞争死亡的累积发生概率），不是原因特异风险率")])
    # nhanes.py 登记
    funcs = []
    for fn in ("load", "add_derived", "add_mortality"):
        f = getattr(nh, fn); lines, start = inspect.getsourcelines(f)
        funcs.append(dict(函数=fn, 起始行=start, 行数=len(lines), 源码SHA=hashlib.sha256("".join(lines).encode()).hexdigest()[:16]))
    nhreg = dict(路径=str(NH), 文件SHA=hashlib.sha256(NH.read_bytes()).hexdigest()[:16], 总行数=len(NH.read_text(encoding="utf-8").splitlines()),
                 R44调用函数=funcs, 说明="R44 队列脚本通过它完成血压（多次读数）、1999–2000 血肌酐校准（Selvin 2007）与 eGFR（CKD-EPI 2021）等衍生；本轮随 R45_CODE_TESTS.zip 交付整个模块")
    PA = pd.read_csv(R44 / "R44_PATHAUDIT.csv")
    FA = PA[PA.结果 != "OK"][["结局", "表型", "seed", "外层折", "结果", "信息"]]
    summ = dict(死亡字段缺失=MISS.set_index("项").N.to_dict(), nhanes模块=nhreg, 不可执行单元=len(FA),
                不可执行原因=sorted(FA.信息.str.split(" @ ").str[0].unique().tolist()))
    with pd.ExcelWriter(OUT / "R45_R44SUPP.xlsx", engine="openpyxl") as xw:
        MISS.to_excel(xw, sheet_name="1_死亡字段缺失", index=False)
        defs.to_excel(xw, sheet_name="2_定义更正", index=False)
        pd.DataFrame(funcs).assign(模块路径=str(NH), 模块SHA=nhreg["文件SHA"]).to_excel(xw, sheet_name="3_nhanes模块", index=False)
        FA.to_excel(xw, sheet_name="4_不可执行单元", index=False)
    (OUT / "R45_R44SUPP.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str)); print(defs.to_string())


if __name__ == "__main__":
    main()
