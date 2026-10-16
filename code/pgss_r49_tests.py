# -*- coding: utf-8 -*-
r"""R49 包测试：每项检查先用于真实对象（必须通过），再用于一个变异对照（必须被检出）。结果写 31_R49_只读补齐/R49_PACKAGE_TESTS.json。

  1 自包含：zip 解压后隔离解释器导入 code/ 全部模块；变异 = 删 pgss_r49_cohort.py 后 pgss_r49_report 导入必须失败
  2 data v5 META ≡ R35 冻结队列（逐条）；变异 = 翻转一个 eligible
  3 最终合格人数由 data v5 重算 = R35 运行记录；变异 = 记录改 1
  4 任务人数/事件/A/B 与 R41、R42、R47 逐项一致；变异 = R41 的一个 A 轨人数 + 1
  5 R35 运行记录 ≡ R35 交付工作簿；变异 = 改一个 N
  6 任务一表中不出现 R34 独有的人数（从 R34 交付表读取与 R35 记录不同的数）；变异 = 往 C2 塞入一个
  7 汇总函数从 R43 B2/B1 重算 B3/B4 一致；变异 = R43 B3 一个中位数 + 1e-9
  8 D1 与 R46 X1 交叉核对；变异 = X1 一个值 + 1e-9
  9 R46 重跑任务集合（D0 判定）= R46 自己的记录（NWICU 住院死亡被排除人次的任务、NHANES C1「需要重跑」）；变异 = v2 未重跑任务的一个值被改
 10 D1、D2 由另一份实现（纯 Python 循环）独立复算；变异 = D1 一个值 + 1e-9
 11 每行都是 3 个 seed，评价人数跨 seed 不变；变异 = 一行 seed 数改为 2
 12 D5 与 R46 X1 的目标一致性追加臂逐值相同；变异 = 改一个值
 13 数字溯源与隐私：真实 zip 与报告两层通过；编造/改动数字、逐人预测文件、标识列、口令必须被检出
"""
import sys, io, re, json, zipfile, pathlib, tempfile, subprocess, shutil, os, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r49_package as P
import pgss_r49_cohort as CO
import pgss_r49_aux as AX
from pgss_r47_package_tests import import_all
from pgss_r46_package_tests import bump, pick
from pgss_r45_package_tests import mkzip, pqb

PGVAR = "PG" + "PASSWORD"
HAND = CO.HAND


def eq_v5_r35(M, C):
    C = C.loc[M.index]
    return int((M.eligible.to_numpy() != (C.eligible & C.mort_source_ok).to_numpy()).sum()) + \
        sum(int((M[o].to_numpy() != C[o].to_numpy()).sum()) for o in ("OUT_HOSP_MORT", "OUT_LOS72"))


def recon(c2, r41, r42, r47):
    bad = 0
    for r in c2.itertuples():
        db = "mimic" if r.数据库 == "MIMIC-IV" else "eicu"; k = (db, r.结局, r.表型)
        a, b, c = r41.loc[k], r42.loc[k], r47.loc[k]
        mine = (r.任务n, r.事件, r.A轨n, r.A轨事件, r.B轨n, r.B轨事件)
        bad += mine != (a.n, a.事件, a.全体A_n, a.全体A_事件, a.全体B_n, a.全体B_事件)
        bad += r.任务n != b.n_units
        bad += mine != (c.n, c.事件, c.A轨n, c.A轨事件, c.B轨n, c.B轨事件)
    return bad


def r34_only_numbers():
    x = pd.read_excel(HAND / "34" / "R34_PGSS_ALL_RESULTS.xlsx", sheet_name="11_队列计数")
    r35 = pd.read_excel(CO.R35D / "R35_COHORT_OUTCOMES.xlsx", sheet_name=None)
    keep = set(pd.to_numeric(pd.concat([r35["1_队列与结局"].N, r35["1_队列与结局"].事件, r35["2_未知与冲突登记"].N]), errors="coerce").dropna().astype(int))
    vals = set(pd.to_numeric(pd.concat([x.N, x.事件]), errors="coerce").dropna().astype(int))
    return sorted(v for v in vals - keep if v > 100)


def numbers_in(frames):
    s = set()
    for f in frames:
        for c in f.columns:
            v = pd.to_numeric(f[c], errors="coerce").dropna()
            s |= set(int(x) for x in v if float(x).is_integer())
    return s


def version_set(v2, v1, key):
    num = [c for c in v2.columns if c not in key + ["库"] and v2[c].dtype.kind in "fi"]
    m = v2.merge(v1, on=key, suffixes=("_v2", "_v1"))
    out = set()
    for (oc, ph), h in m.groupby(["结局", "表型"]):
        if not all(np.array_equal(h[f"{c}_v2"].to_numpy(float), h[f"{c}_v1"].to_numpy(float), equal_nan=True) for c in num):
            out.add((oc, ph))
    return out


def documented_rerun():
    b3 = AX.rd("R46证据表", "B3_NWICU_住院死亡任务变化"); c1 = AX.rd("R46证据表", "C1_NHANES_未知死因")
    return {("nwicu", r.结局, r.表型) for r in b3.itertuples() if r.排除人次 > 0} | {("nhanes", r.结局, r.表型) for r in c1.itertuples() if bool(r.需要重跑)}


def indep_d1(db_frames):
    """纯 Python：按 (库, 结局, 表型, 臂, 范围) 收集三个 seed 的值，算均值、最小、最大。"""
    out = {}
    for lab, a in db_frames.items():
        for r in a.to_dict("records"):
            k = (lab, r["结局"], r["表型"], r["臂"], r["范围"])
            out.setdefault(k, {}).setdefault("_s", []).append(r)
    res = {}
    for k, v in out.items():
        rs = v["_s"]; d = {"seed数": len(rs)}
        for m in AX.MET:
            xs = [float(r[m]) for r in rs if r[m] == r[m]]
            if xs:
                d[f"{m}_均值"] = sum(xs) / len(xs); d[f"{m}_最小"] = min(xs); d[f"{m}_最大"] = max(xs)
        res[k] = d
    return res


def cmp_indep(d1, ind, tol=1e-14):
    worst, miss = 0.0, 0
    for r in d1.to_dict("records"):
        k = (r["数据库"], r["结局"], r["表型"], r["臂"], r["范围"])
        if k not in ind:
            miss += 1; continue
        for c, v in ind[k].items():
            if c == "seed数":
                miss += int(v != r[c])
            elif r.get(c) == r.get(c):
                worst = max(worst, abs(v - r[c]))
    return miss == 0 and worst <= tol, worst, miss


def x1_check(d1, x1):
    worst = 0.0; n = 0
    for db, lab in AX.X1_LAB.items():
        key = ["结局", "表型", "臂", "范围"]
        xa = x1[x1.库 == lab].rename(columns=lambda c: c if c in key else f"X1_{c}")
        m = d1[(d1.数据库 == AX.DBL[db]) & d1.范围.isin(["全体", "A", "B"])].merge(xa, on=key)
        n += len(m)
        for a, b in (("校准截距_均值", "X1_校准截距"), ("校准斜率_均值", "X1_校准斜率"), ("Brier_均值", "X1_Brier"), ("校准斜率_最大", "X1_校准斜率_最大")):
            worst = max(worst, float(np.nanmax(np.abs(m[a].to_numpy(float) - m[b].to_numpy(float)))))
    return worst < 1e-12 and n > 600, worst, n


def main():
    rows = []
    add = lambda 检查, 通过, 说明="": rows.append(dict(检查=检查, 通过=bool(通过), 说明=str(说明)))
    zb, code = P.build_zip(with_tests=False)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="r49pkg_"))
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        z.extractall(tmp)
    cd = tmp / "code"
    xl = pd.read_excel(P.XL, sheet_name=None)

    # 1 自包含
    mods = sorted(p.stem for p in cd.glob("*.py"))
    res = import_all(cd, mods, {PGVAR: "placeholder-not-a-password"})
    bad = {k: v for k, v in res.items() if v}
    add("自包含：隔离解释器导入 code/ 全部模块", res and not bad, f"{len(mods)} 个模块" + (f"；失败 {bad}" if bad else ""))

    # 2 v5 ≡ R35
    for db in CO.DBS:
        M, C = CO.v5(db, "META"), CO.r35(db)
        n = eq_v5_r35(M, C)
        add(f"data v5 META ≡ R35 冻结队列（{CO.DBS[db]['name']}）", n == 0, f"不同 {n}")
        Mm = M.copy(); i = Mm.index[0]; Mm.loc[i, "eligible"] = not Mm.loc[i, "eligible"]
        add(f"变异：翻转一个 eligible 必须被检出（{CO.DBS[db]['name']}）", eq_v5_r35(Mm, C) == 1)

    # 3 最终合格人数
    rec = pd.read_excel(CO.R35D / "R35_COHORT_OUTCOMES.xlsx", sheet_name="1_队列与结局")
    fin = lambda rec_, db: int(rec_[(rec_.数据库 == CO.DBS[db]["name"]) & rec_.项.str.contains("最终合格队列")].N.iloc[0]) == int(CO.v5(db, "META").eligible.sum())
    add("最终合格人数：data v5 重算 = R35 运行记录（两库）", all(fin(rec, db) for db in CO.DBS))
    rm = rec.copy(); j = rm.index[rm.项.str.contains("最终合格队列")][0]; rm.loc[j, "N"] += 1
    add("变异：运行记录的最终人数改 1 必须被检出", not all(fin(rm, db) for db in CO.DBS))

    # 4 任务对账
    r41 = pd.read_excel(CO.ROOT / "22_R41_Stage0" / "R41_Stage0_证据表.xlsx", sheet_name="A1_任务与轨").set_index(["库", "结局", "表型"])
    r42 = pd.read_excel(CO.ROOT / "23_R42_正式主实验" / "inference" / "R42_INFERENCE.xlsx", sheet_name="1_任务完整度").set_index(["库", "结局", "表型"])
    r47 = pd.read_excel(CO.ROOT / "29_R47_写作交接" / "R47_冻结数字包.xlsx", sheet_name="B1_数据与任务概况")
    r47 = r47[r47.库.isin(CO.DBS)].set_index(["库", "结局", "表型"])
    c2 = xl["C2_表型任务"]
    add("任务人数、事件与 A/B 轨计数与 R41、R42、R47 逐项一致（交付工作簿）", recon(c2, r41, r42, r47) == 0, f"{len(c2)} 个任务")
    r41m = r41.copy(); r41m.iloc[5, r41m.columns.get_loc("全体A_n")] += 1
    add("变异：R41 一个 A 轨人数 + 1 必须被检出", recon(c2, r41m, r42, r47) > 0)

    # 5 R35 记录 ≡ 交付
    loc = pd.read_excel(CO.R35D / "R35_COHORT_OUTCOMES.xlsx", sheet_name="1_队列与结局")
    dlv = pd.read_excel(HAND / "35" / "R35_CORRECTED_BENCHMARK.xlsx", sheet_name="10_队列与结局")
    add("R35 运行记录与 R35 交付工作簿逐值相同", AX.same_frame(loc, dlv)[0])
    lm = loc.copy(); lm.loc[2, "N"] += 1
    add("变异：运行记录改一个 N 必须被检出", not AX.same_frame(lm, dlv)[0])

    # 6 不含 R34 独有人数
    r34 = r34_only_numbers()
    cfr = [xl[k] for k in ("C1_纳排流程", "C1b_各步排除原因", "C1c_R35未知与冲突登记", "C2_表型任务", "C4_计数口径")]
    hit = sorted(set(r34) & numbers_in(cfr))
    add("任务一各表不出现 R34 独有的人数（R34 交付表中与 R35 记录不同的数）", r34 and not hit, f"R34 独有 {len(r34)} 个；命中 {hit}")
    c2m = c2.copy(); c2m.loc[0, "事件"] = r34[0]
    add("变异：往 C2 塞入一个 R34 独有人数必须被检出", bool(set(r34) & numbers_in([c2m])))

    # 7 R43 格式
    a5 = AX.a_validate_r43()
    add("汇总函数从 R43 B2/B1 重算 B3/B4 一致", a5.一致.all(), "；".join(f"{r.最大绝对差:.1e}" for r in a5.itertuples()))
    ref = AX.rd("R43证据表", "B3_配对差汇总"); ref.loc[3, "Brier差中位"] += 1e-9
    add("变异：R43 B3 一个中位数 + 1e-9 必须被检出", not AX.same_frame(AX.r43_b3(AX.rd("R43证据表", "B2_配对差逐seed")), ref, tol=1e-12)[0])

    # 8 X1
    d1 = xl["D1_各模型跨seed汇总"]; x1 = AX.rd("R46证据表", "X1_各模型自身校准_全表")
    ok, w, n = x1_check(d1, x1)
    add("D1 与 R46 X1 交叉核对（四库）", ok, f"{n} 行，最大差 {w:.1e}")
    x1m = x1.copy(); jj = x1m.index[x1m.库 == "nwicu（R46 v2）"][7]; x1m.loc[jj, "校准斜率"] += 1e-9
    add("变异：X1 一个值 + 1e-9 必须被检出", not x1_check(d1, x1m)[0])

    # 9 重跑集合
    doc = documented_rerun()
    k6 = ["结局", "表型", "seed", "臂", "范围", "评价对象"]
    got = {("nwicu",) + t for t in version_set(AX.rd("NWICU_v2", "6_辅助指标逐seed"), AX.rd("NWICU_v1", "6_辅助指标逐seed"), k6)} | \
          {("nhanes",) + t for t in version_set(AX.rd("NHANES_v2", "6_辅助指标逐seed"), AX.rd("NHANES_v1", "6_辅助指标逐seed"), k6)}
    d0 = xl["D0_版本判定"]; d0s = {("nwicu" if r.数据库 == "NWICU" else "nhanes", r.结局, r.表型) for r in d0[(d0.表 == "6_辅助指标逐seed") & ~d0.与v1逐值相同].itertuples()}
    add("R46 重跑任务集合 = R46 自己的记录（NWICU B3 有排除人次的任务；NHANES C1 需要重跑）= 交付的 D0 判定", got == doc == d0s, f"{len(doc)} 个任务")
    v2m = AX.rd("NHANES_v2", "6_辅助指标逐seed"); im = v2m.index[v2m.结局 == "O_MORT10"][0]; v2m.loc[im, "AUROC"] += 1e-9
    got_m = {("nhanes",) + t for t in version_set(v2m, AX.rd("NHANES_v1", "6_辅助指标逐seed"), k6)}
    add("变异：未重跑任务的 v2 值被改必须被检出（集合不再等于 R46 记录）", got_m != {t for t in doc if t[0] == "nhanes"})

    # 10 独立复算
    frames = {AX.DBL[db]: AX.aux(db) for db in ("mimic", "eicu", "nwicu", "nhanes")}
    ind = indep_d1(frames)
    ok, w, miss = cmp_indep(d1, ind)
    add("D1 由另一份实现独立复算", ok, f"最大差 {w:.1e}；缺 {miss}")
    d1m = d1.copy(); i1 = d1m.index[d1m.AUROC_均值.notna()][11]       # 有限值（NaN 单元格不参与比较，改它会让变异落空）
    d1m.loc[i1, "AUROC_均值"] += 1e-9
    add("变异：D1 一个值 + 1e-9 必须被检出", not cmp_indep(d1m, ind)[0], f"第 {i1} 行")
    d2 = xl["D2_配对差跨seed汇总"]; worst2, n2 = 0.0, 0
    for lab, db in (("MIMIC-IV", "mimic"), ("eICU-CRD", "eicu"), ("NWICU", "nwicu"), ("NHANES", "nhanes")):
        p = AX.pairs(db)
        for (oc, ph, c, r_, sc), h in p.groupby(["结局", "表型", "候选", "参照", "范围"]):
            row = d2[(d2.数据库 == lab) & (d2.结局 == oc) & (d2.表型 == ph) & (d2.候选 == c) & (d2.参照 == r_) & (d2.范围 == sc)]
            n2 += len(row)
            xs = [float(v) for v in h["AUROC_候选减参照"] if v == v]
            if xs and len(row):
                worst2 = max(worst2, abs(sum(xs) / len(xs) - float(row["AUROC_候选减参照_均值"].iloc[0])), abs(min(xs) - float(row["AUROC_候选减参照_最小"].iloc[0])))
    add("D2 由另一份实现独立复算（AUROC 差的均值与最小值）", worst2 < 1e-14 and n2 == len(d2), f"{n2}/{len(d2)} 行，最大差 {worst2:.1e}")

    # 11 seed 数与 n
    ok11 = (d1.seed数 == 3).all() and (d2.seed数 == 3).all() and d1.loc[d1.n.notna(), "n恒定"].astype(bool).all()
    add("每行都是 3 个 seed，评价人数跨 seed 不变", ok11)
    add("变异：一行 seed 数改为 2 必须被检出", not (d1.assign(seed数=d1.seed数.where(d1.index != 4, 2)).seed数 == 3).all())

    # 12 D5
    d5 = xl["D5_NHANES目标一致性追加臂"]; src = x1[x1.库 == "nhanes（目标一致性追加臂）"].reset_index(drop=True)
    add("D5 与 R46 X1 的目标一致性追加臂逐值相同", AX.same_frame(d5[src.columns], src)[0], f"{len(src)} 行")
    s5 = src.copy(); s5.loc[0, "校准斜率"] += 1e-9
    add("变异：改一个值必须被检出", not AX.same_frame(d5[src.columns], s5)[0])

    # 13 数字溯源与隐私
    md0 = P.MD.read_text(encoding="utf-8")
    probs, n, na = P.audit_all(zb, P.MD, P.XL)
    add("数字溯源基线：真实 zip 与当前报告两层均通过", not probs, f"报告数字 {n} 个" + (f"；{probs}" if probs else ""))
    p6 = pick(md0, r"(?<![\d.])-?\d+\.\d{6}(?![\d%])")
    rz = io.BytesIO(zb)
    cases = {"报告插入编造的六位小数": (zb, md0 + "\n编造 0.123457\n"), "报告改动一个六位小数": (zb, md0.replace(p6, bump(p6), 1)),
             "zip 含逐人预测文件": (mkzip(rz, {"x/runs/u/preds_test.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0),
             "parquet 含 stay_id 列": (mkzip(rz, {"x/y.parquet": pqb(pd.DataFrame({"stay_id": [1]}))}), md0),
             "json 含 subject_id 键": (mkzip(rz, {"x/y.json": b'{"subject_id": 3}'}), md0),
             "py 含真实口令": (mkzip(rz, {"code/b.py": ("PGPASS" + "WORD=" + "hunt" + "er2\n").encode()}), md0)}
    for nm, (z_, md) in cases.items():
        f = tmp / "r.md"; f.write_text(md, encoding="utf-8")
        a, _, _ = P.selfaudit(z_, f, P.XL); b = P.trace_audit(f)
        add(f"变异：{nm}", bool(a or b))

    (cd / "pgss_r49_cohort.py").unlink()
    mut = import_all(cd, ["pgss_r49_report"])
    add("自包含变异：删 pgss_r49_cohort.py 后 pgss_r49_report 导入必须失败", bool(mut.get("pgss_r49_report")))
    shutil.rmtree(tmp, ignore_errors=True)
    out = dict(全部通过=all(r["通过"] for r in rows), 检查=rows, 代码文件=len(code))
    (P.S / "R49_PACKAGE_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for r in rows:
        print("OK " if r["通过"] else "XX ", r["检查"], "|", r["说明"][:150])
    print("全部通过", out["全部通过"])
    return out


if __name__ == "__main__":
    sys.exit(0 if main()["全部通过"] else 1)
