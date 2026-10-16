# -*- coding: utf-8 -*-
r"""R52 分支测试：每条检查配一个必须失败的变异对照（GPT R52 指令 §E 列出的七类 + 冻结、锚点、STOP 灵敏度、报告数字溯源）。
用法：python pgss_r52_tests.py [交付目录]
"""
import sys, json, re, pathlib, shutil, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r52_sens as S

RAWBIO = pathlib.Path(r"D:/NAH公共数据库/NHANES/1_数据/分文件/2005-2006/BIOPRO_D.parquet")
FORBID = {"SEQN", "unit_id", "uid", "subject_id", "group_id", "hadm_id", "stay_id"}


def frames(root):
    return {t: pd.read_parquet(root / f"{t}.parquet") for t in ("X", "P", "OK", "meta")}


def run(deliver=None):
    res = []
    def add(name, ok, mut, det=""):
        res.append(dict(检查=name, 通过=bool(ok), 变异对照检出=bool(mut), 说明=str(det)[:300]))
    F, G = frames(S.ORIG_DATA), frames(S.DATA)
    cyc = F["meta"].Cycle.to_numpy(); s0 = F["X"]["血肌酐"].to_numpy(float); s1 = G["X"]["血肌酐"].to_numpy(float)
    exp, m = S.calibrate(s0, cyc)
    eq = lambda a, b: bool(np.array_equal(a, b, equal_nan=True))
    # 1 公式只施加一次
    twice, _ = S.calibrate(exp, cyc)
    add("1 分支血肌酐 = 官方公式作用于冻结值一次（重复校准误判）", eq(s1, exp), not eq(twice, s1), f"改动 {int(m.sum())} 人")
    # 2 非目标周期不变
    D = S.diff_frames(F, G, m)
    Gm = {k: v.copy() for k, v in G.items()}; i = np.flatnonzero((cyc == "2007-2008") & np.isfinite(s0))[0]
    Gm["X"].iloc[i, Gm["X"].columns.get_loc("血肌酐")] += 0.01
    Dm = S.diff_frames(F, Gm, m)
    add("2 非目标周期与非目标列未被改动", all(D.values()), not Dm["血肌酐改变的行 = 目标周期且非缺失（不多不少）"], f"{len(D)} 项")
    # 3 源字段是未校准原值（独立读本地 BIOPRO_D 原始文件）
    bio = pd.read_parquet(RAWBIO, columns=["SEQN", "LBXSCR"]); bio = bio.set_index(bio.SEQN.astype(np.int64)).LBXSCR
    idx = F["X"].index[cyc == "2005-2006"]
    raw = bio.reindex(idx).to_numpy(float)
    add("3 冻结 2005–2006 血肌酐 = BIOPRO_D 原始 LBXSCR（未校准输入）", eq(F["X"].loc[idx, "血肌酐"].to_numpy(float), raw),
        not eq(G["X"].loc[idx, "血肌酐"].to_numpy(float), raw), f"{len(idx)} 人")
    # 4 资格 / 结局 / 分组不变
    Gm = {k: v.copy() for k, v in G.items()}; Gm["meta"].iloc[0, Gm["meta"].columns.get_loc("O_MORT10")] = not bool(Gm["meta"].O_MORT10.iloc[0])
    add("4 资格、结局、分组、随访逐格不变", D["meta（资格、分组、性别、结局、随访）逐格相同"] and D["OK 六列逐人相同"],
        not S.diff_frames(F, Gm, m)["meta（资格、分组、性别、结局、随访）逐格相同"])
    # 5 其余五个表型逐人不变；eGFR 迁移只在改动行且只单向
    Gm = {k: v.copy() for k, v in G.items()}; Gm["P"].iloc[0, Gm["P"].columns.get_loc("贫血")] = not bool(Gm["P"]["贫血"].iloc[0])
    mig = F["P"]["eGFR降低"].to_numpy() != G["P"]["eGFR降低"].to_numpy()
    add("5 其余五表型逐人相同；eGFR降低 只在改动行变化", D["P 除 eGFR降低 外五个表型逐人相同"] and D["P eGFR降低 只在改动行变化"],
        not S.diff_frames(F, Gm, m)["P 除 eGFR降低 外五个表型逐人相同"], f"迁移 {int(mig.sum())} 人")
    # 6 QC 锚点
    A = pd.read_excel(S.BR / "R52_BUILD.xlsx", sheet_name="2_QC锚点")
    Am = A.copy(); Am.loc[6, "R52重算"] += 1
    ok6 = lambda a: bool(np.isclose(a.R52重算.astype(float), a.R51登记.astype(float), rtol=0, atol=1e-12).all())
    add("6 QC 锚点与 R51 T6 逐值一致", ok6(A), not ok6(Am), f"{len(A)} 项")
    # 7 冻结文件未改动
    fz = json.loads((S.BR / "R52_FREEZE.json").read_text(encoding="utf-8"))
    am = json.loads((S.BR / "R52_AMENDMENT.json").read_text(encoding="utf-8")) if (S.BR / "R52_AMENDMENT.json").exists() else None
    allowed = lambda f, v: {v} | ({am["修订后SHA"]} if (am and f == "pgss_r52_sens.py" and v == am["冻结SHA"]) else set())
    chk7 = lambda reg: [f for f, v in reg["文件SHA"].items() if S.fsha(S.HERE / f) not in allowed(f, v)] + \
                       [f for f, v in reg["数据文件SHA"].items() if S.fsha(S.BR / f) != v]
    fzm = json.loads(json.dumps(fz)); k0 = next(k for k in fzm["文件SHA"] if k != "pgss_r52_sens.py"); fzm["文件SHA"][k0] = fzm["文件SHA"][k0][::-1]
    bad = chk7(fz)
    add("7 冻结的代码与分支数据未改动（冻结值或登记修订值；含原冻结学习程序）", not bad, bool(chk7(fzm)),
        bad or f"{len(fz['文件SHA'])} 个代码文件；登记修订 {1 if am else 0} 处")
    done = (S.BR / "_R52_DONE").exists()
    if done:
        # 8 患者不泄漏：每个 seed 的五个测试折互不重叠、并集 = 任务全体
        import pgss_r44_nhanes_run as NR
        bad8, mutok = [], None
        for oc, ph in S.included_tasks():
            A1, keep, Ac, names, gs, fp = S.arrays(oc, ph)
            pop = np.sort(A1["uid"][keep])
            for s in NR.SEEDS:
                parts = [pd.read_parquet(S.DIRS["main"] / S.tag_of("main", oc, ph, s, f) / "preds_test.parquet", columns=["unit_id"]).unit_id.to_numpy()
                         for f in range(NR.K_OUTER)]
                cat = np.concatenate(parts)
                if not (len(cat) == len(np.unique(cat)) and np.array_equal(np.sort(cat), pop)):
                    bad8.append(f"{oc}×{ph}×{s}")
                if mutok is None:
                    c2 = np.concatenate([parts[0], parts[1][:1]] + parts[1:])
                    mutok = len(c2) != len(np.unique(c2))
        add("8 患者不泄漏：各 seed 五个测试折互不重叠且并集 = 任务全体", not bad8, mutok, bad8[:5] or "11 任务 × 3 seed")
        # 9 旧结果替代：新单元与新推断都带分支数据指纹
        newfp = json.loads((S.DATA / "COHORT.json").read_text(encoding="utf-8"))["指纹"]
        metas = [json.loads((p / "meta.json").read_text(encoding="utf-8")) for d in (S.DIRS["main"], S.DIRS["pop"]) for p in d.iterdir() if (p / "meta.json").exists()]
        fpok = lambda ms: all({k: x["data_fp"].get(k) for k in newfp} == newfp and x["data_fp"].get("branch") == S.BRANCH for x in ms)
        old = json.loads((S.frozen_dir("main", "O_MORT10", "女性", 20261003, 0) / "meta.json").read_text(encoding="utf-8"))
        inf = json.loads((S.BR / "_INFER_new.json").read_text(encoding="utf-8"))
        add("9 没有旧结果替代：全部新单元与新推断带分支数据指纹", fpok(metas) and inf["数据指纹"] == newfp and "35_R52" in json.dumps(inf["路由"], ensure_ascii=False),
            not fpok(metas + [old]), f"{len(metas)} 个单元")
        # 10 STOP 规则灵敏：把一个两轨等权 E1 的三态改写后，判定必须变为 SCIENTIFIC_DECISION_REQUIRED
        import pgss_r52_compare as C
        def mutated(fn):
            tmp = pathlib.Path(tempfile.mkdtemp(dir=S.BR))
            try:
                (tmp / "inference").mkdir(); (tmp / "target_inference").mkdir()
                Wn = pd.read_excel(S.BR / "inference" / "R44_INFERENCE.xlsx", sheet_name=None)
                fn(Wn)
                with pd.ExcelWriter(tmp / "inference" / "R44_INFERENCE.xlsx", engine="openpyxl") as xw:
                    for k, v in Wn.items():
                        v.to_excel(xw, sheet_name=k, index=False)
                shutil.copy2(S.BR / "target_inference" / "R45_TARGET_INFERENCE.xlsx", tmp / "target_inference" / "R45_TARGET_INFERENCE.xlsx")
                return C.main(str(tmp))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        def flip_bal(Wn):
            b = Wn["2_Brier对照"]; j = b.index[(b.对照 == "E1") & (b.尺度 == "bal")][0]
            b.loc[j, "三态_m0.010"] = "支持实质收益" if b.loc[j, "三态_m0.010"] != "支持实质收益" else "排除实质收益"
        def maxt_support(Wn):
            t = Wn["5_E1_maxT"]; j = t.index[~t["同时判定_相对m"].astype(str).str.startswith("支持")][0]
            t.loc[j, "同时判定_相对m"] = "支持（收益>m）"
        m1, m2 = mutated(flip_bal), mutated(maxt_support)
        real = json.loads((S.BR / "R52_COMPARE.json").read_text(encoding="utf-8"))
        add("10 STOP 规则灵敏：改写一个两轨等权三态必须触发 D-1；注入一个 max-T「支持」必须触发 D-3",
            (not real["试跑"]) and real["新结果来源"].endswith("35_R52_NHANES_SCr2005_06_SENS"),
            "D-1" in m1["触发规则"] and "D-3" in m2["触发规则"], f"正式判定：{real['判定'][:40]}")
    # 11 交付物不含个人信息
    ev = S.BR / "R52_NHANES_SCr_SENS_EVIDENCE.xlsx"
    if ev.exists():
        W = pd.read_excel(ev, sheet_name=None)
        clean = lambda ws: all(len(v) <= 1000 and not (set(map(str, v.columns)) & FORBID) for v in ws.values())
        Wm = dict(W); Wm["x"] = pd.DataFrame({"SEQN": [1]})
        add("11 证据表只含汇总（无 SEQN 等标识列、每表 ≤1000 行）", clean(W), not clean(Wm), f"{len(W)} 张表")
    # 12 报告数字溯源
    md = S.BR / "R52_NHANES_SCr_SENS_REPORT.md"
    if md.exists():
        import pgss_r52_report as RP
        txt = md.read_text(encoding="utf-8"); trace = json.loads((S.BR / "R52_NUM_TRACE.json").read_text(encoding="utf-8"))
        u = RP.unregistered(txt, trace)
        add("12 报告中每个数字都已登记到结果表来源", not u, bool(RP.unregistered(txt + "\n虚构 12,345 人。", trace)), u[:8])
        src = (HERE / "pgss_r52_report.py").read_text(encoding="utf-8")
        add("13 报告源码不含手填数字", not RP.lint(src), bool(RP.lint(src + "\nx = n(61, 'T')\n")))
        add("14 报告由当前结果确定性重生成", RP.main(write=False) == txt, RP.main(write=False) != txt + " ")
    if deliver:
        dd = pathlib.Path(deliver)
        pairs = [(S.BR / f, dd / f) for f in ("R52_NHANES_SCr_SENS_REPORT.md", "R52_NHANES_SCr_SENS_EVIDENCE.xlsx")]
        same = all(a.read_bytes() == b.read_bytes() for a, b in pairs)
        add("15 交付副本与工作目录逐字节相同", same, pairs[0][0].read_bytes() != pairs[0][1].read_bytes() + b" ")
    df = pd.DataFrame(res)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
    print(df.to_string()); print(f"通过 {int(df.通过.sum())}/{len(df)}；变异对照检出 {int(df.变异对照检出.sum())}/{len(df)}")
    return df


if __name__ == "__main__":
    df = run(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(0 if df.通过.all() and df.变异对照检出.all() else 1)
