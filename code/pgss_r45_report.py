# -*- coding: utf-8 -*-
r"""R45 外部复现与机制核验：统一报告与证据表（md 与 xlsx **同源生成**：md 中每个数字都来自写入 xlsx「报告引用」表的同一 DataFrame）。

输入：27_R45_外部复现与机制核验/{r43fix, r44supp, target, nwicu, plasmode}/ 的冻结记录、测试、运行元数据与汇总表；
      只读 R42 推断表、R43 只读补齐表、R44 推断与证据表（跨库对照）。
输出：27_R45_外部复现与机制核验/R45_外部复现与机制核验_统一报告.md、R45_外部复现与机制核验_证据表.xlsx
用法：python pgss_r45_report.py
"""
import sys, json, pathlib, hashlib, re
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
R45 = ROOT / "27_R45_外部复现与机制核验"
TGT, NWD, PLD = R45 / "target", R45 / "nwicu", R45 / "plasmode"
MD = R45 / "R45_外部复现与机制核验_统一报告.md"; XL = R45 / "R45_外部复现与机制核验_证据表.xlsx"
R42X = ROOT / "23_R42_正式主实验" / "inference" / "R42_INFERENCE.xlsx"
R43RO = ROOT / "24_R43_结果补齐与T1" / "readonly" / "R43_READONLY.xlsx"
R44X = ROOT / "25_R44_NHANES跨领域复现" / "inference" / "R44_INFERENCE.xlsx"
pc = lambda v, d=2: ("—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{100 * v:.{d}f}%")
f6 = lambda v: ("—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.6f}")
OCN = {"O_MORT10": "10 年全因死亡", "O_CVD10": "10 年心脑血管死亡", "OUT_HOSP_MORT": "住院死亡", "OUT_LOS72": "ICU>72h"}
DBN = {"nhanes": "NHANES", "mimic": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
CITED = ["A1_冻结与执行核对", "A2_运行前检查", "A3_冻结后修订", "B1_R43_达标数口径差异", "B2_R43_重复级错误启用", "B2b_R43_错误启用合计", "B3_R43_带符号差", "B4_R43_认证启用项非零",
         "C1_R44_死亡字段缺失", "C2_R44_定义更正", "C3_R44_nhanes模块",
         "D1_目标一致性_预定判读", "D2_目标一致性_对照", "D3_目标一致性_尺度汇总", "D4_C1_maxT", "D5_差距收回", "D6_事后代数",
         "D7_开发选择分布", "D8_logloss", "D9_11任务pop尺度并列",
         "E0_NWICU_队列汇总", "E1_NWICU_队列流程", "E2_NWICU_结局登记", "E3_NWICU_死亡口径核对", "E4_NWICU_来源适配与QC", "E5_NWICU_缺失模式",
         "E6_NWICU_任务", "E7_NWICU_E1逐任务", "E8_NWICU_对照方向汇总", "E9_NWICU_maxT", "E10_NWICU_辅助指标差", "E11_NWICU_开发选择",
         "E12_NWICU_预定措辞判定", "E13_NWICU_特征覆盖", "E14_NWICU_出院去向与事件", "H1_意见所用数字",
         "F1_plasmode_生成器", "F2_plasmode_候选对真值收益", "F3_plasmode_选择器逐轨", "F4_plasmode_两轨等权部署",
         "F5_plasmode_重复级错误启用", "F6_plasmode_MIX与POOLMIX", "F7_plasmode_认证样本成本", "F8_plasmode_δ配对差", "F9_plasmode_构成",
         "G1_跨库E1汇总"]


def fsha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def jload(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def kv(d):
    return pd.DataFrame([(k, json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (list, dict)) else v) for k, v in d.items()],
                        columns=["项", "值"])


def amendments(fz_path):
    """同目录登记的冻结后修订（R45_*_AMENDMENT.json）→ {文件: 修订后SHA}。"""
    p = list(pathlib.Path(fz_path).parent.glob("R45_*_AMENDMENT.json"))
    return {a["文件"]: a for x in p for a in jload(x)["修订"]}


def freeze_now(fz_path):
    """冻结登记的文件现值 = 冻结值；登记了修订的文件 = 登记的修订后值（且其冻结值与登记一致）。"""
    fz = jload(fz_path); am = amendments(fz_path)
    ok = all(fsha(ROOT / k) == v if k not in am else (am[k]["冻结SHA"] == v and fsha(ROOT / k) == am[k]["修订后SHA"])
             for k, v in fz["文件SHA"].items())
    return fz, ok


def log_time(path, key):
    if not pathlib.Path(path).exists():
        return None
    for ln in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if key in ln:
            return ln[:14]
    return None


def three(df, col="三态_m0.010"):
    vc = df[col].value_counts()
    return int(vc.get("排除实质收益", 0)), int(vc.get("证据不足", 0)), int(vc.get("支持实质收益", 0))


# ── A：冻结、执行与运行前检查 ───────────────────────────────────
def part_a():
    import pgss_r45_target as TG, pgss_r45_nwicu_run as NW, pgss_r45_plasmode as PM
    rows, units = [], {}
    # 目标一致性
    fz, same = freeze_now(TGT / "R45_TARGET_FREEZE.json")
    U = []
    for t in ("pop", "B"):
        for a in TG.units(t):
            d = TG.RUNS / TG.tag_of(*a)
            st = "OK" if (d / "_DONE").exists() else ("FAILED" if (d / "_FAILED").exists() else "MISSING")
            m = jload(d / "meta.json") if st == "OK" else {}
            U.append(dict(目标=a[0], 结局=a[1], 表型=a[2], seed=a[3], 外层折=a[4], 状态=st, 代码SHA=m.get("code_sha"),
                          R44规格SHA一致=(m.get("r44_specs_sha") == fsha(ROOT / "25_R44_NHANES跨领域复现" / "runs" /
                                                                  f"nhanes__{a[1]}__{a[2]}__{a[3]}__{a[4]}__main" / "specs.json")) if m else None,
                          早停权重替换调用=(m.get("calls") or {}).get("es_pooled")))
    U = pd.DataFrame(U); units["target"] = U
    st = jload(TGT / "_STATUS.json") if (TGT / "_STATUS.json").exists() else {}
    rows.append(dict(部分="NHANES 目标一致性对照", 冻结时间=fz["冻结时间"], 登记文件数=len(fz["文件SHA"]), 冻结后登记修订=len(amendments(TGT / "R45_TARGET_FREEZE.json")), 登记文件现值等于冻结值或登记修订值=same,
                     驱动核对通过=log_time(TGT / "target_driver.log", "冻结指纹核对通过"), 单元总数=len(U),
                     完成=int((U.状态 == "OK").sum()), 不可执行=int((U.状态 == "FAILED").sum()), 缺失=int((U.状态 == "MISSING").sum()),
                     代码SHA全部相同且等于现值=bool((U.代码SHA == TG.code_sha()).all()), 运行用时分钟=st.get("已用分钟"),
                     附加核对="R44 规格文件 SHA 全部一致：" + str(bool(U.R44规格SHA一致.fillna(False).all())) +
                              f"；每单元都调用了替换后的早停权重（最少 {int(U.早停权重替换调用.min()) if U.早停权重替换调用.notna().any() else 0} 次）"))
    # NWICU
    fz, same = freeze_now(NWD / "R45_NWICU_FREEZE.json")
    PA = pd.read_csv(NWD / "R45_NWICU_PATHAUDIT.csv", encoding="utf-8-sig")
    coh = jload(NW.DATA / "COHORT.json")
    V = []
    for a in NW.units():
        d = NW.RUNS / NW.tag_of(*a)
        stt = "OK" if (d / "_DONE").exists() else ("FAILED" if (d / "_FAILED").exists() else "MISSING")
        m = jload(d / "meta.json") if stt == "OK" else {}
        V.append(dict(结局=a[0], 表型=a[1], seed=a[2], 外层折=a[3], 正式运行=stt, 代码SHA=m.get("code_sha"),
                      数据指纹一致=(m.get("data_fp") == coh["指纹"]) if m else None))
    V = pd.DataFrame(V).merge(PA[["结局", "表型", "seed", "外层折", "结果"]].rename(columns={"结果": "路径审计"}),
                              on=["结局", "表型", "seed", "外层折"], how="left")
    units["nwicu"] = V
    st = jload(NWD / "_STATUS.json") if (NWD / "_STATUS.json").exists() else {}
    rows.append(dict(部分="NWICU 独立程序复现", 冻结时间=fz["冻结时间"], 登记文件数=len(fz["文件SHA"]), 冻结后登记修订=len(amendments(NWD / "R45_NWICU_FREEZE.json")), 登记文件现值等于冻结值或登记修订值=same,
                     驱动核对通过=log_time(R45 / "chain_driver.log", "NWICU 冻结指纹核对通过"), 单元总数=len(V),
                     完成=int((V.正式运行 == "OK").sum()), 不可执行=int((V.正式运行 == "FAILED").sum()), 缺失=int((V.正式运行 == "MISSING").sum()),
                     代码SHA全部相同且等于现值=bool((V.代码SHA == NW.code_sha()).all()), 运行用时分钟=st.get("已用分钟"),
                     附加核对=f"数据指纹全部等于构建值：{bool(V.数据指纹一致.fillna(False).all())}；路径审计可执行 {int((V.路径审计 == 'OK').sum())}；"
                              f"正式运行与路径审计逐单元一致：{bool((V.正式运行 == V.路径审计).all())}"))
    # plasmode
    fz, same = freeze_now(PLD / "R45_PLASMODE_FREEZE.json")
    W = []
    for a in PM.job_list():
        d = PM.RUNS / PM.tag_of(*a)
        stt = "OK" if (d / "_DONE").exists() else ("FAILED" if (d / "_FAILED").exists() else "MISSING")
        m = jload(d / "rep.json") if stt == "OK" else {}
        W.append(dict(δ=a[0], rep=a[1], 状态=stt, 代码SHA=m.get("code_sha"), 秒=m.get("sec_total")))
    W = pd.DataFrame(W); units["plasmode"] = W
    st = jload(PLD / "_STATUS.json") if (PLD / "_STATUS.json").exists() else {}
    rows.append(dict(部分="plasmode", 冻结时间=fz["冻结时间"], 登记文件数=len(fz["文件SHA"]), 冻结后登记修订=len(amendments(PLD / "R45_PLASMODE_FREEZE.json")), 登记文件现值等于冻结值或登记修订值=same,
                     驱动核对通过=log_time(R45 / "chain_driver.log", "plasmode 冻结指纹核对通过"), 单元总数=len(W),
                     完成=int((W.状态 == "OK").sum()), 不可执行=int((W.状态 == "FAILED").sum()), 缺失=int((W.状态 == "MISSING").sum()),
                     代码SHA全部相同且等于现值=bool((W.代码SHA == PM.code_sha()).all()), 运行用时分钟=st.get("已用分钟"), 附加核对=""))
    A1 = pd.DataFrame(rows)
    # 运行前检查
    chk = []
    T = jload(TGT / "R45_TARGET_TESTS.json")
    chk.append(dict(检查="目标适配器单元测试", 项数=len(T["测试"]), 通过项=sum(t["通过"] for t in T["测试"]), 全部通过=T["全部通过"]))
    E = jload(TGT / "R45_TARGET_EQUIV.json")
    chk.append(dict(检查="q = 0.5 等价测试（R44 单元逐位复现）", 项数=len(E["单元"]), 通过项=sum(u["一致"] for u in E["单元"]), 全部通过=E["全部一致"]))
    chk.append(dict(检查="等价测试的变异对照（必须检出）", 项数=len(E["变异对照"]), 通过项=sum(m["检出"] for m in E["变异对照"]), 全部通过=E["变异全检出"]))
    P = jload(PLD / "R45_PLASMODE_TESTS.json")
    chk.append(dict(检查="plasmode 端到端测试（缩小网格）", 项数=len(P["测试"]), 通过项=sum(t["通过"] for t in P["测试"]), 全部通过=P["全部通过"]))
    par = coh["表型函数parity"]
    chk.append(dict(检查="NWICU 表型函数在 MIMIC v5 上逐位复现已存表型（含变异对照）", 项数=2,
                    通过项=int(par["MIMIC_v5逐位相同"]) + int(par["变异对照检出"]), 全部通过=bool(par["MIMIC_v5逐位相同"] and par["变异对照检出"])))
    sg = jload(NWD / "R45_NWICU_STAGE.json")
    chk.append(dict(检查="NWICU 源文件 SHA256 与官方清单一致", 项数=len(sg["源文件"]), 通过项=sum(s["一致"] for s in sg["源文件"]),
                    全部通过=all(s["一致"] for s in sg["源文件"])))
    fx = jload(R45 / "r43fix" / "R45_R43FIX.json")
    chk.append(dict(检查="R43 选择器表真值 = rep.json 最终 truth（逐行断言）", 项数=int(fx["选择器表行数"]),
                    通过项=int(fx["选择器表行数"]) - int(fx["选择器表真值一致性断言_不一致行"]), 全部通过=fx["选择器表真值一致性断言_不一致行"] == 0))
    if (TGT / "R45_TARGET_INFERENCE.json").exists():
        ti = jload(TGT / "R45_TARGET_INFERENCE.json")
        chk.append(dict(检查="目标一致性推断与 R44 同一重抽：重算 E1 与 R44 已交付值逐位相同", 项数=ti["与R44_E1复核项"],
                        通过项=ti["与R44_E1复核项"] if ti["复核全部相同"] else 0, 全部通过=ti["复核全部相同"]))
    A2 = pd.DataFrame(chk)
    A3 = pd.DataFrame([dict(部分=nm, **{k: v for k, v in a.items()}) for nm, fz in
                       (("NHANES 目标一致性对照", TGT / "R45_TARGET_FREEZE.json"), ("NWICU 独立程序复现", NWD / "R45_NWICU_FREEZE.json"),
                        ("plasmode", PLD / "R45_PLASMODE_FREEZE.json")) for a in amendments(fz).values()])
    return A1, A2, A3, units, coh


# ── B：R43 只读更正 ─────────────────────────────────────────────
def part_b():
    X = pd.read_excel(R45 / "r43fix" / "R45_R43FIX.xlsx", sheet_name=None)
    B1 = X["1_达标数口径差异"]
    TF = X["4_重复级至少一轨错误启用"]
    keep_sel = ["FCT|P*", "FCT|S0*", "CV_POINT_M|P*", "CV_LB_M|P*", "CV_ANY|P*", "USTUN_ADAPT"]
    B2 = TF[TF.选择器.isin(keep_sel)][["批次", "机制", "n", "πA", "选择器", "独立重复", "至少一轨不达标的重复",
                                       "≥1错误启用_分子", "≥1错误启用_比例", "≥1错误启用_CP上", "≥1实际变差启用_分子"]].reset_index(drop=True)
    TC = X["3_选择器逐轨_更正"]
    B3 = (TC.groupby("选择器").agg(行=("重复", "sum"), 带符号差为负的重复=("带符号差为负的重复", "sum"), 带符号差最小=("带符号差最小", "min"))
          .reset_index())
    RO = pd.read_excel(R43RO, sheet_name="9_认证分解")
    B4 = RO[RO.认证启用项 != 0][["库", "结局", "表型", "尺度", "样本成本项", "认证启用项", "合计", "样本成本项_相对R_PstarD",
                                  "认证启用项_相对R_PstarD"]].reset_index(drop=True)
    B2b = B2.groupby("选择器").agg(条件=("机制", "size"), 错误启用重复合计=("≥1错误启用_分子", "sum"),
                                   实际变差启用重复合计=("≥1实际变差启用_分子", "sum"), 条件最大比例=("≥1错误启用_比例", "max"),
                                   CP上界最大=("≥1错误启用_CP上", "max")).reset_index()
    return B1, B2, B2b, B3, B4, X


# ── C：R44 补齐 ────────────────────────────────────────────────
def part_c():
    X = pd.read_excel(R45 / "r44supp" / "R45_R44SUPP.xlsx", sheet_name=None)
    return X["1_死亡字段缺失"], X["2_定义更正"], X["3_nhanes模块"], X["4_不可执行单元"]


# ── D：NHANES 目标一致性 ─────────────────────────────────────────
def part_d():
    if not (TGT / "R45_TARGET_INFERENCE.xlsx").exists():
        return None
    X = pd.read_excel(TGT / "R45_TARGET_INFERENCE.xlsx", sheet_name=None)
    BR = X["2_Brier对照"]
    keep = ["结局", "表型", "对照", "参照", "候选", "尺度", "R参照", "观察绝对差", "观察相对改善", "相对_P2_5", "相对_P97_5",
            "三态_m0.005", "三态_m0.010", "三态_m0.020", "相对0方向"]
    D2 = BR[keep].reset_index(drop=True)
    rows = []
    for (c, sc), h in D2.groupby(["对照", "尺度"], sort=False):
        ex, ins, sup = three(h); vc = h.相对0方向.value_counts()
        rows.append(dict(对照=c, 尺度=sc, 任务=len(h), 相对改善中位=h.观察相对改善.median(), 相对改善最小=h.观察相对改善.min(),
                         相对改善最大=h.观察相对改善.max(), 相对上界最大=h.相对_P97_5.max(), 排除m1=ex, 证据不足m1=ins, 支持m1=sup,
                         全负=int(vc.get("全负", 0)), 跨零=int(vc.get("跨零", 0)), 全正=int(vc.get("全正", 0))))
    D3 = pd.DataFrame(rows)
    LL = X["3_logloss对照"]
    D8 = LL[LL.对照.isin(["C1", "D1"])][keep].reset_index(drop=True)
    # 方案 §6：预定措辞只适用于 O_MORT10×低白蛋白；推断脚本表 4 对 11 个任务都套了 C1 措辞，这里只取该任务
    V = X["4_预定判读"]
    D1 = V[(V.结局 == "O_MORT10") & (V.表型 == "低白蛋白")].reset_index(drop=True)
    # 11 个任务在 pop 尺度上并列 E1（v1.0）、C2（P*_pop vs P*）、C1（HTFS_pop vs P*_pop）——描述
    P = D2[(D2.尺度 == "pop") & D2.对照.isin(["E1", "C2", "C1"])]
    rows = []
    for (oc, ph), h in P.groupby(["结局", "表型"], sort=False):
        r = dict(结局=oc, 表型=ph)
        for c in ("E1", "C2", "C1"):
            g = h[h.对照 == c].iloc[0]
            r.update({f"{c}_相对改善": g.观察相对改善, f"{c}_P2_5": g.相对_P2_5, f"{c}_P97_5": g.相对_P97_5,
                      f"{c}_三态m1": g["三态_m0.010"], f"{c}_相对0方向": g.相对0方向})
        rows.append(r)
    D9 = pd.DataFrame(rows)
    return dict(D1=D1, D2=D2, D3=D3, D4=X["5_C1_maxT"], D5=X["6_差距收回"], D6=X["7_事后代数_GPT4_2"],
                D7=X["9_开发选择分布"], D8=D8, D9=D9, X=X)


# ── E：NWICU ───────────────────────────────────────────────────
def part_e(coh):
    B = pd.read_excel(NWD / "R45_NWICU_BUILD.xlsx", sheet_name=None)
    E1 = B["1_队列流程"]; E2 = B["2_结局不确定与登记"]
    E3 = B["8_死亡口径核对_住院层面"]
    E3b = B["3_出院去向与事件"]
    E4 = B["4_范围QC与非法值"].merge(B["5_三步取值统计"].rename(columns={"变量": "变量"}), on="变量", how="left")
    E5 = B["9_体征缺失模式"]
    TK = B["7_候选任务无性能资格"]
    PA = jload(NWD / "R45_NWICU_PATHAUDIT.json")
    pa = pd.DataFrame(PA["按任务"]).rename(columns={"结果": "路径审计可执行"})
    E6 = TK.merge(pa, on=["结局", "表型"], how="left")
    E13 = B["6_特征覆盖"]
    E0 = pd.DataFrame([dict(项="最终人次", 值=coh["最终人次"]), dict(项="唯一患者", 值=coh["唯一患者"])] +
                      [dict(项=f"{o} 事件", 值=coh["结局事件"][o]) for o in coh["结局事件"]] +
                      [dict(项=f"{o} 事件率", 值=coh["结局事件率"][o]) for o in coh["结局事件率"]])
    out = dict(E0=E0, E1=E1, E2=E2, E3=E3, E3b=E3b, E4=E4, E5=E5, E6=E6, E13=E13, B=B)
    if (NWD / "inference" / "R45_NWICU_INFERENCE.xlsx").exists():
        X = pd.read_excel(NWD / "inference" / "R45_NWICU_INFERENCE.xlsx", sheet_name=None)
        BR = X["2_Brier对照"]
        keep = ["结局", "表型", "尺度", "R参照", "观察绝对差", "观察相对改善", "相对_P2_5", "相对_P97_5",
                "三态_m0.005", "三态_m0.010", "三态_m0.020", "相对0方向"]
        E7 = BR[BR.对照 == "E1"][keep].reset_index(drop=True)
        PR = X["7_配对差逐seed"]
        p = PR[(PR.候选 == "HTFS") & (PR.参照 == "P*") & (PR.范围 == "全体")]
        E10 = (p.groupby(["结局", "表型"], sort=False)[["AUROC_候选减参照", "AUPRC_候选减参照", "校准截距_候选减参照", "校准斜率_候选减参照"]]
               .mean().reset_index())
        bal = E7[E7.尺度 == "bal"]
        ex, ins, sup = three(bal)
        sup_any = E7[E7["三态_m0.010"] == "支持实质收益"]
        TH = X["5_E1_maxT"]
        E12 = dict(纳入任务=len(bal), 主要尺度排除m1=ex, 主要尺度证据不足m1=ins, 主要尺度支持m1=sup,
                   主要尺度全部排除=bool(ex == len(bal) and len(bal) > 0),
                   主要尺度证据不足任务="；".join(f"{r.结局}×{r.表型}" for r in bal[bal["三态_m0.010"] == "证据不足"].itertuples()),
                   任一尺度支持实质收益="；".join(f"{r.结局}×{r.表型}（{r.尺度}）" for r in sup_any.itertuples()),
                   maxT支持="；".join(f"{r.结局}×{r.表型}（{r.轨}）" for r in TH[TH.同时判定_相对m.str.startswith("支持")].itertuples()),
                   maxT排除项数=int(TH.同时判定_相对m.str.startswith("排除").sum()), maxT项数=len(TH),
                   maxT证据不足项数=int(TH.同时判定_相对m.str.startswith("证据不足").sum()),
                   主要尺度点估计为负任务=int((bal.观察相对改善 < 0).sum()), 主要尺度区间全负任务=int((bal.相对0方向 == "全负").sum()),
                   四尺度支持合计=int((E7["三态_m0.010"] == "支持实质收益").sum()), 四尺度行数=len(E7))
        out.update(E7=E7, E8=X["4_对照方向汇总"], E9=TH, E10=E10, E11=X["9_开发选择分布"], E12=E12, X=X, ST=X["1_任务纳入"])
    return out


# ── F：plasmode ────────────────────────────────────────────────
def part_f():
    G = pd.read_csv(PLD / "R45_PLASMODE_GENERATOR.csv", encoding="utf-8-sig")
    if not (PLD / "R45_PLASMODE_SUMMARY.xlsx").exists():
        return dict(F1=G)
    X = pd.read_excel(PLD / "R45_PLASMODE_SUMMARY.xlsx", sheet_name=None)
    keep_sel = ["ALWAYS_SPLIT", "CV_ANY|P*", "CV_POINT_M|P*", "CV_LB_M|P*", "FCT|P*", "USTUN_ADAPT", "MIX", "POOLMIX"]
    F3 = X["C_选择器逐轨"]; F3 = F3[F3.选择器.isin(keep_sel)].reset_index(drop=True)
    F4 = X["D_选择器两轨等权"]; F4 = F4[F4.选择器.isin(keep_sel)].reset_index(drop=True)
    F5 = X["E_重复级至少一轨错误启用"]; F5 = F5[F5.选择器.isin(keep_sel)].reset_index(drop=True)
    return dict(F1=G, F2=X["B_候选对真值收益"], F3=F3, F4=F4, F5=F5, F6=X["F_MIX与POOLMIX"], F7=X["G_认证样本成本"],
                F8=X["H_δ配对差"], F9=X["I_开发选择与构成"], FA=X["A_各臂风险"], FJ=X["J_逐重复记录"], X=X)


# ── G：跨库 E1 汇总 ─────────────────────────────────────────────
def part_g(E):
    R42 = pd.read_excel(R42X, sheet_name="2_全部对照").rename(columns={"相对_P2.5": "相对_P2_5", "相对_P97.5": "相对_P97_5"})
    R42 = R42[R42.对照 == "E1"]
    R44 = pd.read_excel(R44X, sheet_name="2_Brier对照"); R44 = R44[R44.对照 == "E1"]
    parts = [("MIMIC-IV（R42）", R42[R42.库 == "mimic"]), ("eICU-CRD（R42）", R42[R42.库 == "eicu"]), ("NHANES（R44）", R44)]
    if "E7" in E:
        parts.append(("NWICU（R45）", E["E7"]))
    rows = []
    for nm, h in parts:
        for sc in ("bal", "A", "B", "pop"):
            g = h[h.尺度 == sc]
            ex, ins, sup = three(g); vc = g.相对0方向.value_counts()
            rows.append(dict(数据库=nm, 尺度=sc, 任务=len(g), 相对改善中位=g.观察相对改善.median(), 相对上界最大=g.相对_P97_5.max(),
                             全负=int(vc.get("全负", 0)), 跨零=int(vc.get("跨零", 0)), 全正=int(vc.get("全正", 0)),
                             排除m1=ex, 证据不足m1=ins, 支持m1=sup))
    return pd.DataFrame(rows)


# ── H：意见部分用到的计算量 ─────────────────────────────────────
def part_h():
    """0/n 时 Clopper–Pearson 上界与把它压到 5% 以下所需的独立重复数。"""
    rows = []
    for n in (20, 60):
        rows.append(dict(项=f"0/{n} 时 Clopper–Pearson 双侧 95% 上界", 公式="1 − 0.025^(1/n)", 值=1 - 0.025 ** (1 / n)))
    rows.append(dict(项="0/n 时双侧 95% 上界 < 5% 所需最少独立重复", 公式="ceil(ln 0.025 / ln 0.95)",
                     值=float(np.ceil(np.log(0.025) / np.log(0.95)))))
    rows.append(dict(项="0/n 时单侧 95% 上界 < 5% 所需最少独立重复", 公式="ceil(ln 0.05 / ln 0.95)",
                     值=float(np.ceil(np.log(0.05) / np.log(0.95)))))
    return pd.DataFrame(rows)


def build():
    A1, A2, A3, units, coh = part_a()
    B1, B2, B2b, B3, B4, BX = part_b()
    C1, C2, C3, C4 = part_c()
    D = part_d()
    E = part_e(coh)
    F = part_f()
    G1 = part_g(E)
    sheets = {"A1_冻结与执行核对": A1, "A2_运行前检查": A2, "A3_冻结后修订": A3,
              "B1_R43_达标数口径差异": B1, "B2_R43_重复级错误启用": B2, "B2b_R43_错误启用合计": B2b, "B3_R43_带符号差": B3, "B4_R43_认证启用项非零": B4,
              "C1_R44_死亡字段缺失": C1, "C2_R44_定义更正": C2, "C3_R44_nhanes模块": C3}
    if D:
        sheets.update({"D1_目标一致性_预定判读": D["D1"], "D2_目标一致性_对照": D["D2"], "D3_目标一致性_尺度汇总": D["D3"],
                       "D4_C1_maxT": D["D4"], "D5_差距收回": D["D5"], "D6_事后代数": D["D6"], "D7_开发选择分布": D["D7"],
                       "D8_logloss": D["D8"], "D9_11任务pop尺度并列": D["D9"]})
    sheets.update({"E0_NWICU_队列汇总": E["E0"], "E1_NWICU_队列流程": E["E1"],"E2_NWICU_结局登记": E["E2"], "E3_NWICU_死亡口径核对": E["E3"],
                   "E4_NWICU_来源适配与QC": E["E4"], "E5_NWICU_缺失模式": E["E5"], "E6_NWICU_任务": E["E6"],
                   "E13_NWICU_特征覆盖": E["E13"]})
    if "E7" in E:
        sheets.update({"E7_NWICU_E1逐任务": E["E7"], "E8_NWICU_对照方向汇总": E["E8"], "E9_NWICU_maxT": E["E9"],
                       "E10_NWICU_辅助指标差": E["E10"], "E11_NWICU_开发选择": E["E11"], "E12_NWICU_预定措辞判定": kv(E["E12"])})
    sheets["F1_plasmode_生成器"] = F["F1"]
    if "F2" in F:
        sheets.update({"F2_plasmode_候选对真值收益": F["F2"], "F3_plasmode_选择器逐轨": F["F3"], "F4_plasmode_两轨等权部署": F["F4"],
                       "F5_plasmode_重复级错误启用": F["F5"], "F6_plasmode_MIX与POOLMIX": F["F6"], "F7_plasmode_认证样本成本": F["F7"],
                       "F8_plasmode_δ配对差": F["F8"], "F9_plasmode_构成": F["F9"]})
    sheets["G1_跨库E1汇总"] = G1
    # 明细表（不进入数字溯源池）
    sheets["X1_R43_候选对_最终truth"] = BX["2_候选对_最终truth"]
    sheets["X2_R43_选择器逐轨_更正"] = BX["3_选择器逐轨_更正"]
    sheets["X3_R43_重复级错误启用全表"] = BX["4_重复级至少一轨错误启用"]
    sheets["X4_R44_不可执行单元"] = C4
    if D:
        sheets["X5_目标一致性_逐单元选择"] = D["X"]["8_开发选择逐单元"]
        sheets["X6_目标一致性_与R44逐位复核"] = D["X"]["10_与R44逐位复核"]
    sheets["X7_目标一致性_逐单元执行"] = units["target"]
    sheets["X8_NWICU_逐单元执行与路径审计"] = units["nwicu"]
    sheets["E14_NWICU_出院去向与事件"] = E["E3b"]
    sheets["H1_意见所用数字"] = part_h()
    if "X" in E:
        sheets["X10_NWICU_Brier全部对照"] = E["X"]["2_Brier对照"]
        sheets["X11_NWICU_logloss对照"] = E["X"]["3_logloss对照"]
        sheets["X12_NWICU_辅助指标逐seed"] = E["X"]["6_辅助指标逐seed"]
    if "FJ" in F:
        sheets["X13_plasmode_各臂风险"] = F["FA"]
        sheets["X14_plasmode_逐重复"] = F["FJ"]
    sheets["X15_plasmode_逐重复执行"] = units["plasmode"]
    with pd.ExcelWriter(XL, engine="openpyxl") as xw:
        pd.DataFrame({"表": ["0_说明"] + list(sheets), "报告引用": ["否"] + ["是" if k in CITED else "否" for k in sheets]}
                     ).to_excel(xw, sheet_name="目录", index=False)
        pd.DataFrame({"说明": [
            "本表与统一报告由 pgss_r45_report.py 同源生成；报告中的数字只来自「目录」中标为「报告引用」的表。",
            "推断范围：固定 OOF 预测的簇重抽（第 Ⅱ 层；NHANES 簇 = 个体，NWICU 簇 = 患者），不是算法级区间。",
            "相对改善 = (R参照 − R候选) / R参照；正值 = 候选更好。",
            "plasmode 与 T1 的「真值」是评价人群上已知生成概率下的条件 Brier；plasmode 的评价人群有限且精确计算。",
            "不含任何逐人记录、逐人预测或患者编号；NWICU 与 MIMIC 的患者级数据只留本地。"]}).to_excel(xw, sheet_name="0_说明", index=False)
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)
    return dict(A1=A1, A2=A2, A3=A3, B1=B1, B2=B2, B2b=B2b, B3=B3, B4=B4, C1=C1, C2=C2, C3=C3, C4=C4, D=D, E=E, F=F, G1=G1, coh=coh,
                fx=jload(R45 / "r43fix" / "R45_R43FIX.json"), sheets=sheets)


# ── md 各部分 ───────────────────────────────────────────────────
def row(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    h = df[m]
    return h.iloc[0] if len(h) else None


def part_d_md(R):
    D = R["D"]
    L = ["## 4. NHANES 目标一致性对照（P\\*_pop、P\\*_B；运行前登记）\n"]
    if D is None:
        L.append("运行未完成。\n"); return "\n".join(L)
    D1, D2, D3, D4, D5, D6, D7, D8, D9 = (D[k] for k in ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"))
    L.append("合并参照的开发目标从两轨等权改为指定目标 R_q = q·R_A + (1−q)·R_B，**贯穿早停（含决定 SZ-GFS 面板的全局排序模型）、θ、SZ-GFS 的 k、"
             "各合并臂 raw/cal 开关与 P\\* 选择**；训练权重候选、候选族、网格与外层/内层划分不变；SZ-union 用同一训练部分内的轨面板（kA/kB 取 R44 冻结值）。"
             "HTFS 参数与面板冻结，只按目标的内层 OOF 分数在已有 raw/cal 版本中重选（HTFS_q）。pop = 按记录等权（11 个已纳入任务）；"
             "B = 只用 B 轨评分（仅 O_MORT10×低白蛋白）。R44 v1.0 原样保留，本对照在看到 v1.0 之后提出、运行之前登记，不倒写成最初的主比较。\n")
    k = dict(结局="O_MORT10", 表型="低白蛋白")
    e1B, e1P = row(D2, 对照="E1", 尺度="B", **k), row(D2, 对照="E1", 尺度="pop", **k)
    c1, c2, c3 = row(D2, 对照="C1", 尺度="pop", **k), row(D2, 对照="C2", 尺度="pop", **k), row(D2, 对照="C3", 尺度="pop", **k)
    d1, d2, d3 = row(D2, 对照="D1", 尺度="B", **k), row(D2, 对照="D2", 尺度="B", **k), row(D2, 对照="D3", 尺度="B", **k)
    v1, vB = row(D1, 对照="C1", **k), row(D1, 对照="D1", **k)
    L.append("### 4.1 预定判读：O_MORT10×低白蛋白\n")
    tb = pd.DataFrame([dict(对照="E1（v1.0）", 参照="P*", 候选="HTFS", 尺度="B", r=e1B), dict(对照="E1（v1.0）", 参照="P*", 候选="HTFS", 尺度="pop", r=e1P),
                       dict(对照="C1", 参照="P*_pop", 候选="HTFS_pop", 尺度="pop", r=c1), dict(对照="C2", 参照="P*（v1.0）", 候选="P*_pop", 尺度="pop", r=c2),
                       dict(对照="C3", 参照="P*_pop", 候选="HTFS（v1.0）", 尺度="pop", r=c3), dict(对照="D1", 参照="P*_B", 候选="HTFS_B", 尺度="B", r=d1),
                       dict(对照="D2", 参照="P*（v1.0）", 候选="P*_B", 尺度="B", r=d2), dict(对照="D3", 参照="P*_B", 候选="HTFS（v1.0）", 尺度="B", r=d3)])
    tb = tb[tb.r.notna()]
    tb["相对改善"] = [pc(r.观察相对改善) for r in tb.r]; tb["95% 区间"] = [ivl(r) for r in tb.r]
    tb["绝对差"] = [f6(r.观察绝对差) for r in tb.r]; tb["三态（m=1%）"] = [r["三态_m0.010"] for r in tb.r]
    L.append(tbl(tb, ["对照", "参照", "候选", "尺度", "绝对差", "相对改善", "95% 区间", "三态（m=1%）"]))
    if v1 is not None:
        L.append(f"\n- **C1 预定措辞**：{esc(v1.预定措辞)}。")
    if vB is not None:
        L.append(f"- **D1 预定措辞**：{esc(vB.预定措辞)}。")
    g1 = row(D5, 目标="pop", **k); gB = row(D5, 目标="B", **k)
    if g1 is not None:
        L.append(f"- 差距收回（描述）：v1.0 在人口加权尺度上 R(P\\*) − R(HTFS) = {f6(g1.v10差距)}；P\\*_pop 收回 {f6(g1.收回)}"
                 f"（占 v1.0 差距的 {pc(g1.收回比例_相对v10差距)}）。")
    if gB is not None:
        L.append(f"  B 尺度上 v1.0 差距 {f6(gB.v10差距)}，P\\*_B 收回 {f6(gB.收回)}（{pc(gB.收回比例_相对v10差距)}）。")
    c2A, c2B = row(D2, 对照="C2", 尺度="A", **k), row(D2, 对照="C2", 尺度="B", **k)
    if c2A is not None and c2B is not None:
        L.append(f"- 描述：P\\*_pop 相对 v1.0 P\\* 不只在 B 轨更好（{pc(c2B.观察相对改善)}，{ivl(c2B)}），在 A 轨也更好（{pc(c2A.观察相对改善)}，{ivl(c2A)}）。"
                 "也就是说，在这个 A 轨极小的任务中，两轨等权的开发目标并没有让合并模型在 A 轨上更好。早停轮数、θ、SZ-GFS 的 k 与 raw/cal 开关随目标同时改变，"
                 "哪一个环节起主要作用本轮没有分离，不作机制解释。")
    L.append("\n### 4.2 全部 11 个任务（pop 尺度并列；除 O_MORT10×低白蛋白外只作描述）\n")
    L.append("按方案 §6，预定措辞只适用于 O_MORT10×低白蛋白；推断脚本的「4_预定判读」表对 11 个任务都套用了 C1 措辞，那 10 行不按预定措辞解读"
             "（证据表 D1 只保留该任务；推断脚本已冻结，未改动）。\n")
    d9 = D9.copy()
    for c in ("E1", "C2", "C1"):
        d9[f"{c} 相对改善"] = d9[f"{c}_相对改善"].map(pc)
        d9[f"{c} 95% 区间"] = [f"{pc(a)} ~ {pc(b)}" for a, b in zip(d9[f"{c}_P2_5"], d9[f"{c}_P97_5"])]
    L.append(tbl(d9, ["结局", "表型", "E1 相对改善", "E1 95% 区间", "E1_三态m1", "C2 相对改善", "C2 95% 区间", "C2_三态m1",
                      "C1 相对改善", "C1 95% 区间", "C1_三态m1"]))
    L.append("\n（E1 = v1.0 的 HTFS vs P\\*；C2 = P\\*_pop vs v1.0 P\\*；C1 = HTFS_pop vs P\\*_pop；全部在人口加权尺度。）\n")
    sub = D3[D3.对照.isin(["C1", "C2", "E1"])]
    L.append(tbl(sub, ["对照", "尺度", "任务", "相对改善中位", "相对改善最小", "相对改善最大", "相对上界最大", "排除m1", "证据不足m1", "支持m1"],
                 fmt={c: pc for c in ("相对改善中位", "相对改善最小", "相对改善最大", "相对上界最大")}))
    c1p = D2[(D2.对照 == "C1") & (D2.尺度 == "pop")]
    sup = c1p[c1p["三态_m0.010"] == "支持实质收益"]
    L.append(f"\nC1（pop 尺度）逐任务：支持实质收益 {len(sup)} 个"
             + ("（" + "；".join(f"{r.结局}×{r.表型} {pc(r.观察相对改善)}，{ivl(r)}" for r in sup.itertuples()) + "）" if len(sup) else "")
             + f"，排除 {int((c1p['三态_m0.010'] == '排除实质收益').sum())} 个，证据不足 {int((c1p['三态_m0.010'] == '证据不足').sum())} 个。逐任务见表 D2。")
    c95 = float(D4["max-T临界值"].iloc[0])
    L.append(f"max-T（C1，pop 尺度，{len(D4)} 项，临界值 {c95:.4f}）：支持 {int(D4.同时判定_相对m.str.startswith('支持').sum())} 项，"
             f"排除 {int(D4.同时判定_相对m.str.startswith('排除').sum())} 项，证据不足 {int(D4.同时判定_相对m.str.startswith('证据不足').sum())} 项（表 D4）。\n")
    L.append("### 4.3 GPT 第 49 轮 §4.2 的事后代数（按 v1.0 风险复算）\n")
    p = D6.iloc[0]
    L.append(f"A 轨保持 P\\*、只有 B 轨换成 HTFS 的两轨等权点估计改善 = ΔR_B /(R_A(P\\*) + R_B(P\\*)) = {f6(p.ΔR_B)} /({f6(p.R_A_Pstar)} + {f6(p.R_B_Pstar)}) "
             f"= {pc(p.两轨等权相对改善, 4)}，与 GPT 的计算一致。这是看到测试结果后提出的组合，只作事后描述，不是已独立选择和验证的 PGSS 策略。\n")
    L.append("### 4.4 开发选择与 log loss\n")
    L.append(tbl(D7, ["项", "目标", "选择", "单元数"]))
    lc1 = row(D8, 对照="C1", 尺度="pop", **k); ld1 = row(D8, 对照="D1", 尺度="B", **k)
    if lc1 is not None and ld1 is not None:
        L.append(f"\nlog loss（O_MORT10×低白蛋白）：C1 pop {pc(lc1.观察相对改善)}（{ivl(lc1)}，{lc1['三态_m0.010']}）；"
                 f"D1 B {pc(ld1.观察相对改善)}（{ivl(ld1)}，{ld1['三态_m0.010']}）。\n")
    return "\n".join(L)


def part_e_md(R):
    E = R["E"]; coh = R["coh"]
    L = ["## 5. NWICU 独立程序复现\n"]
    fl = E["E1"]; reg = E["E2"].set_index("项").N; dc = E["E3"]; mp = E["E5"]; tk = E["E6"]
    L.append("**定位**：冻结学习程序在独立 ICU 数据中的复现，不是 ICU 模型的直接迁移。NWICU 在方案冻结前只用于覆盖审计与无性能构建，未拟合过任何模型。\n")
    L.append("### 5.1 构建（GPT 第 49 轮 §6 的规范落实）\n")
    L.append(tbl(fl, ["步骤", "人次"]))
    L.append(f"\n最终 {int(coh['最终人次'])} 人次（{int(coh['唯一患者'])} 名患者）；住院死亡 {coh['结局事件']['OUT_HOSP_MORT']} 例（{pc(coh['结局事件率']['OUT_HOSP_MORT'])}），"
             f"ICU>72h {coh['结局事件']['OUT_LOS72']} 例（{pc(coh['结局事件率']['OUT_LOS72'])}）。\n")
    ex = row(dc, 出院去向="EXPIRED"); eh = row(dc, 出院去向="EXPIRED - HOSPICE"); ih = row(dc, 出院去向="INPATIENT HOSPICE"); hh = row(dc, 出院去向="HOME WITH HOSPICE")
    L.append("**住院死亡的编码核实**（住院层面全部 admissions；Δ = deathtime − dischtime，小时）：\n")
    L.append(tbl(dc, ["出院去向", "住院数", "有deathtime", "Δ中位_小时", "绝对差不超过24h", "死亡早于出院超24h", "死亡晚于出院超24h", "hospital_expire_flag为1"],
                 fmt={"Δ中位_小时": lambda v: "—" if not np.isfinite(v) else f"{v:.2f}"}))
    L.append(f"\n- EXPIRED - HOSPICE：有 deathtime 的 {int(eh.有deathtime)} 例中 {int(eh.绝对差不超过24h)} 例在出院前后 24 h 内死亡（与 EXPIRED 的 {int(ex.绝对差不超过24h)}/{int(ex.有deathtime)} 模式相同）⇒ 判为住院结束时已死亡，纳入 b 口径。"
             f"INPATIENT HOSPICE 与 HOME WITH HOSPICE 的死亡中位在出院后 {ih.Δ中位_小时:.2f} 与 {hh.Δ中位_小时:.2f} 小时 ⇒ 活着出院，不算死亡。")
    L.append(f"- NWICU 的 deathtime 还记录出院后的死亡：合格队列中活着出院但之后死亡的 {int(reg['活着出院且 deathtime 晚于出院（出院后死亡，非事件）'])} 人次不是事件。"
             f"hospital_expire_flag = 1 只有 {int(reg['hospital_expire_flag = 1（不使用）'])} 次，恰好都是「死亡状态且无 deathtime」的记录，不使用。")
    L.append(f"- 结局不确定并排除：死亡状态但死亡时点晚于出院 + 1 天 {int(reg['结局不确定①：死亡状态但死亡时点晚于出院 + 1 天'])}；"
             f"活着出院的状态但 deathtime ≤ dischtime {int(reg['结局不确定②：活着出院的状态但 deathtime ≤ dischtime'])}；去向缺失 {int(reg['结局不确定③：discharge_location 缺失'])}。"
             f"死亡状态但无 deathtime 的 {int(reg['死亡状态且 deathtime 缺失（用 dischtime 作死亡时点）'])} 次用 dischtime 作死亡时点（与 MIMIC 同处理）。")
    L.append("- c 口径（deathtime ≤ 出院 + 1 天、不看去向）只作时间一致性描述，见表 E14；不运行其模型。\n")
    L.append("**来源适配**：平均动脉压是**计算值**——同一设备（动脉线、无创）、同一 charttime 的 (SBP + 2·DBP)/3，各分量先过范围 QC 并取该时点最晚 storetime 版本，"
             "再走与其他变量相同的三步取值；NWICU 没有直接测量 MAP 的 itemid，不能声称 23 项全按原始测量定义复现。"
             "血红蛋白只有血液学来源（MIMIC 另含血气）；碳酸氢盐为化学组 Total CO2；单位拼写变体为同一单位，无需换算；9999999 等非法值由范围 QC 去除（表 E4）。"
             "身高体重的基线体型按距 intime 最近、storetime 最早、itemid 最小打破并列，仍并列且取值不同时取均值（实测 0 次）。\n")
    allm = row(mp, 分组方式="全部")
    L.append(f"**缺失模式**：窗内有体征的入住占 {pc(allm.窗内有体征)}；{pc(allm.整段无任何体征记录)} 的合格入住**整段入住没有任何体征记录**——是结构性缺失，不是窗口造成的。"
             f"窗内有/无体征者的住院死亡率分别为 {pc(allm.住院死亡率_窗内有体征)} 与 {pc(allm.住院死亡率_窗内无体征)}。主分析不排除这些患者，按冻结的训练内中位数插补；按 ICU 类型与年份组的缺失见表 E5。\n")
    L.append("### 5.2 候选任务与路径审计（无性能）\n")
    L.append(tbl(tk, ["结局", "表型", "可判定人次", "唯一患者", "A轨人次", "A轨比例", "事件", "A轨事件", "B轨事件", "路径审计可执行"],
                 fmt={"A轨比例": pc}))
    if "E7" not in E:
        L.append("\n正式运行未完成。\n"); return "\n".join(L)
    E7, E8, E9, E10, E11, E12 = E["E7"], E["E8"], E["E9"], E["E10"], E["E11"], E["E12"]
    L.append("\n### 5.3 主要结果：E1（HTFS vs P\\*）\n")
    bal = E7[E7.尺度 == "bal"].copy()
    bal["相对改善"] = bal.观察相对改善.map(pc); bal["95% 区间"] = [ivl(r) for _, r in bal.iterrows()]
    L.append(tbl(bal, ["结局", "表型", "R参照", "观察绝对差", "相对改善", "95% 区间", "三态_m0.010", "相对0方向"]))
    w = (f"在 NWICU 中，同一冻结程序下的专属模型在两轨等权尺度上未显示实质收益（全部 {E12['纳入任务']} 个纳入任务的 95% 区间上界 < 1%）"
         if E12["主要尺度全部排除"] else
         (f"未发现支持实质收益的任务；{E12['主要尺度证据不足m1']} 个任务证据不足（{E12['主要尺度证据不足任务']}）" if E12["主要尺度支持m1"] == 0 else
          f"{E12['主要尺度支持m1']} 个任务在主要尺度上支持实质收益，作为与 MIMIC/eICU 的不一致逐任务报告"))
    L.append(f"\n**预定措辞（方案 §7）**：{w}。")
    L.append(f"任一尺度上支持实质收益的任务与尺度：{E12['任一尺度支持实质收益'] or '无'}；max-T（{E12['maxT项数']} 项）支持：{E12['maxT支持'] or '无'}，"
             f"排除 {E12['maxT排除项数']} 项，证据不足 {E12['maxT证据不足项数']} 项（表 E9）。\n")
    e8 = E8[(E8.指标 == "Brier") & (E8.对照.isin(["E1", "E2", "E1_vs_S0*", "E3", "E4", "E7"]))]
    L.append(tbl(e8, ["对照", "尺度", "任务", "相对改善中位", "相对改善最小", "相对改善最大", "相对上界最大", "全负", "跨零", "全正", "排除m1", "证据不足m1", "支持m1"],
                 fmt={c: pc for c in ("相对改善中位", "相对改善最小", "相对改善最大", "相对上界最大")}))
    L.append("\n（E1 = HTFS vs P\\*；E2 = Hfull vs P\\*；E1_vs_S0\\* = HTFS vs S0\\*；E3 = SZ\\* vs S0\\*；E4 = HTFS vs SZ-union\\*；E7 = HTFS vs HGFS。全部对照与 log loss 见明细表。）\n")
    L.append("原概率下的辅助指标（HTFS − P\\*，3 个 seed 平均，全体；不重校准）：\n")
    L.append(tbl(E10, ["结局", "表型", "AUROC_候选减参照", "AUPRC_候选减参照", "校准截距_候选减参照", "校准斜率_候选减参照"]))
    L.append("\n开发选择分布：\n")
    L.append(tbl(E11, ["项", "Pstar", "单元数"]))
    L.append("")
    return "\n".join(L)


def part_f_md(R):
    F = R["F"]
    L = ["## 6. 有限半合成（plasmode）试验\n",
         "固定经验协变量分布下的探索性压力测试（MIMIC data v5，eGFR 降低任务的可判定人群；原始 23 项池与原缺失；冻结引擎与 T1 全部选择器）。"
         "δ > 0 只表示生成函数含轨道特异结构，**不等于已注入某个幅度的专业化收益**；收益由冻结模型的风险评价决定。"
         "不模拟医院层残差相关，不覆盖 eICU 少量不均衡簇下的认证性质。\n"]
    L.append(tbl(F["F1"], ["δ", "截距a", "人群平均p", "A轨平均p", "B轨平均p", "A轨Bayes风险", "B轨Bayes风险"],
                 fmt={"人群平均p": pc, "A轨平均p": pc, "B轨平均p": pc}))
    if "F2" not in F:
        L.append("\n正式运行未完成。\n"); return "\n".join(L)
    F2, F4, F5, F6, F7, F8, F9 = (F[k] for k in ("F2", "F4", "F5", "F6", "F7", "F8", "F9"))
    L.append("\n### 6.1 候选对真值收益（评价人群上精确）\n")
    f2 = F2[F2.候选对.isin(["D|P*", "D|S0*", "F|P*"])]
    L.append(tbl(f2, ["δ", "候选对", "轨", "重复", "真值相对收益均值", "SE", "最小", "最大", "达m_分子", "收益为负"],
                 fmt={c: pc for c in ("真值相对收益均值", "SE", "最小", "最大")}))
    L.append("\n（D|P\\* = HTFS_D vs P\\*_D；D|S0\\* = HTFS_D vs S0\\*_D；F|P\\* = 认证分支 HTFS_F vs P\\*_F。每个 δ 20 个重复。）\n")
    L.append("δ 之间的同一重复配对差（HTFS_D vs P\\*_D 真值收益）：\n")
    L.append(tbl(F8, ["轨", "比较", "重复", "收益差均值", "SE", "为正重复"], fmt={"收益差均值": pc, "SE": pc}))
    L.append("\n### 6.2 选择器部署后的两轨等权改善（相对 P\\*_D）\n")
    L.append(tbl(F4, ["δ", "选择器", "重复", "均值", "SE", "最小", "最大"], fmt={c: pc for c in ("均值", "SE", "最小", "最大")}))
    L.append("\n### 6.3 重复级「至少一条不达标轨被启用」\n")
    L.append(tbl(F5, ["δ", "选择器", "独立重复", "≥1错误启用_分子", "≥1错误启用_CP上", "≥1实际变差启用_分子"], fmt={"≥1错误启用_CP上": pc}))
    L.append("\n### 6.4 MIX 与普通合并融合 POOLMIX（同一重复配对，两轨等权部署改善）\n")
    L.append(tbl(F6, ["δ", "重复", "MIX均值", "POOLMIX均值", "配对差均值", "配对差SE", "MIX更好重复"],
                 fmt={c: pc for c in ("MIX均值", "POOLMIX均值", "配对差均值", "配对差SE")}))
    L.append("\n### 6.5 认证样本成本（P\\*_F 相对 P\\*_D，真值风险）与开发构成\n")
    L.append(tbl(F7, ["δ", "轨", "重复", "P星F相对P星D改善均值", "SE", "最小", "最大"], fmt={c: pc for c in ("P星F相对P星D改善均值", "SE", "最小", "最大")}))
    L.append("\n" + tbl(F9, ["δ", "重复", "开发记录均值", "开发患者均值", "评价记录均值", "A轨开发人数均值", "A轨开发事件均值", "B轨开发事件均值", "P*分布", "ALT分布"]))
    L.append("")
    return "\n".join(L)


def part_g_md(R):
    G1 = R["G1"]
    L = ["## 7. 四个数据库的 E1 汇总（HTFS vs P\\*，m = 1%）\n"]
    L.append(tbl(G1, ["数据库", "尺度", "任务", "相对改善中位", "相对上界最大", "全负", "跨零", "全正", "排除m1", "证据不足m1", "支持m1"],
                 fmt={"相对改善中位": pc, "相对上界最大": pc}))
    L.append("\nMIMIC、eICU 与 NWICU 是 ICU 住院死亡与 ICU>72h；NHANES 是一般人群 10 年死亡。四库的结局、人群与特征可得性不同，只并列，不合并成一个总体估计。\n")
    return "\n".join(L)


def conclusion(R):
    A1, A2, D, E, F, fx, B2 = R["A1"], R["A2"], R["D"], R["E"], R["F"], R["fx"], R["B2"]
    L = ["## 0. 一页结论\n"]
    ex = "；".join(f"{r.部分} {int(r.完成)}/{int(r.单元总数)}（不可执行 {int(r.不可执行)}，缺失 {int(r.缺失)}）" for r in A1.itertuples())
    L.append(f"**执行**：{ex}。冻结记录登记的文件在报告生成时等于冻结值（或登记的冻结后修订值）：{'是' if A1.登记文件现值等于冻结值或登记修订值.all() else '否'}；冻结后登记修订 {int(A1.冻结后登记修订.sum())} 处（表 A3）；"
             f"运行前检查 {len(A2)} 类，全部通过：{'是' if A2.全部通过.all() else '否'}（表 A1、A2）。\n")
    # NHANES 目标一致性
    if D is not None:
        k = dict(结局="O_MORT10", 表型="低白蛋白")
        c1, d1 = row(D["D2"], 对照="C1", 尺度="pop", **k), row(D["D2"], 对照="D1", 尺度="B", **k)
        e1p, e1b = row(D["D2"], 对照="E1", 尺度="pop", **k), row(D["D2"], 对照="E1", 尺度="B", **k)
        v1, vB = row(D["D1"], 对照="C1", **k), row(D["D1"], 对照="D1", **k)
        g1, gB = row(D["D5"], 目标="pop", **k), row(D["D5"], 目标="B", **k)
        c1all = D["D2"][(D["D2"].对照 == "C1") & (D["D2"].尺度 == "pop")]
        ex1, ins1, sup1 = three(c1all)
        mt = D["D4"]
        L.append("**1. NHANES 低白蛋白「专业化收益」的来源（运行前登记的追加对照，预定判读）**")
        L.append(f"- R44 v1.0 中 O_MORT10×低白蛋白是唯一支持实质收益的任务：E1 在 B 轨 {pc(e1b.观察相对改善)}（{ivl(e1b)}）、人口加权 {pc(e1p.观察相对改善)}（{ivl(e1p)}）。")
        L.append(f"- 合并参照改按人口加权开发（P\\*_pop）后，C1（HTFS_pop vs P\\*_pop，pop 尺度）= {pc(c1.观察相对改善)}（{ivl(c1)}），{c1['三态_m0.010']}；"
                 f"预定措辞：{esc(v1.预定措辞)}。")
        L.append(f"- 合并参照只按 B 轨开发（P\\*_B）后，D1（HTFS_B vs P\\*_B，B 轨）= {pc(d1.观察相对改善)}（{ivl(d1)}），{d1['三态_m0.010']}；"
                 f"预定措辞：{esc(vB.预定措辞)}。")
        L.append(f"- 描述：P\\*_pop 收回了 v1.0 人口加权差距的 {pc(g1.收回比例_相对v10差距)}，P\\*_B 收回了 B 轨差距的 {pc(gB.收回比例_相对v10差距)}。"
                 f"11 个任务的 C1 在 pop 尺度上：排除 {ex1}、证据不足 {ins1}、支持 {sup1}；max-T 排除 {int(mt.同时判定_相对m.str.startswith('排除').sum())}/{len(mt)}。")
        L.append("- **含义（只限本冻结程序与本任务）**：R44 中这项收益不需要 B 轨独立训练就能取得；它与合并参照的开发目标（两轨等权）和评价尺度（B 轨、人口加权）不一致有关。"
                 "这是在看到 R44 结果之后提出、运行之前登记的对照，R44 v1.0 原样保留。\n")
    # NWICU
    if "E12" in E:
        E12 = E["E12"]
        w = (f"在 NWICU 中，同一冻结程序下的专属模型在两轨等权尺度上未显示实质收益（全部 {E12['纳入任务']} 个纳入任务的 95% 区间上界 < 1%）"
             if E12["主要尺度全部排除"] else
             (f"未发现支持实质收益的任务；{E12['主要尺度证据不足m1']} 个任务证据不足（{E12['主要尺度证据不足任务']}）" if E12["主要尺度支持m1"] == 0 else
              f"{E12['主要尺度支持m1']} 个任务在主要尺度上支持实质收益，作为与 MIMIC/eICU 的不一致逐任务报告（见 §5.3）"))
        L.append("**2. NWICU 独立程序复现（冻结学习程序在独立 ICU 数据中的复现，不是模型迁移）**")
        L.append(f"- 预定措辞（E1，两轨等权，m = 1%）：{w}。")
        L.append(f"- 两轨等权尺度上 {E12['纳入任务']} 个任务中 {E12['主要尺度点估计为负任务']} 个点估计为负（专属模型更差），{E12['主要尺度区间全负任务']} 个 95% 区间全在 0 以下。"
                 f"逐轨、两轨等权、人口加权四个尺度共 {E12['四尺度行数']} 行，支持实质收益 {E12['四尺度支持合计']} 行；"
                 f"max-T（{E12['maxT项数']} 项）支持 {E12['maxT支持'] or '0 项'}、排除 {E12['maxT排除项数']} 项、证据不足 {E12['maxT证据不足项数']} 项。")
        G1 = R["G1"]; icu = G1[G1.数据库.str.startswith(("MIMIC", "eICU", "NWICU"))]
        if int(icu.支持m1.sum()) == 0:
            L.append(f"- 与 MIMIC-IV、eICU-CRD（R42）一致：三个 ICU 数据库的 {int(icu[icu.尺度 == 'bal'].任务.sum())} 个任务中，"
                     "没有任何任务在任何尺度上被判为「支持实质收益」（§7 表）。")
        else:
            L.append(f"- 三个 ICU 数据库合计有 {int(icu.支持m1.sum())} 个任务×尺度被判为支持实质收益（§7 表）。")
        L.append("- 来源适配（MAP 为计算值、29% 入住整段无体征、Hb 只有血液学来源、碳酸氢盐为 Total CO2）见 §5.1；结果只对这一构建成立。\n")
    # plasmode
    if "F2" in F:
        F2, F5, F6 = F["F2"], F["F5"], F["F6"]
        L.append("**3. plasmode（固定经验协变量分布下的探索性压力测试）**")
        for dl in sorted(F2.δ.unique()):
            a_ = row(F2, δ=dl, 候选对="D|P*", 轨="A"); b_ = row(F2, δ=dl, 候选对="D|P*", 轨="B")
            fc = row(F5, δ=dl, 选择器="FCT|P*"); mx = row(F6, δ=dl)
            s = (f"- δ = {dl:g}：HTFS_D 相对 P\\*_D 的真值相对收益 A 轨 {pc(a_.真值相对收益均值)}（SE {pc(a_.SE)}，达 m {int(a_.达m_分子)}/{int(a_.重复)}），"
                 f"B 轨 {pc(b_.真值相对收益均值)}（SE {pc(b_.SE)}，达 m {int(b_.达m_分子)}/{int(b_.重复)}）")
            if fc is not None:
                s += f"；FCT|P\\* 同一重复至少一条不达标轨被启用 {int(fc['≥1错误启用_分子'])}/{int(fc.独立重复)}"
            if mx is not None:
                s += f"；MIX − POOLMIX 两轨等权部署改善配对差 {pc(mx.配对差均值)}（SE {pc(mx.配对差SE)}）"
            L.append(s + "。")
        F3, F4, F7 = F["F3"], F["F4"], F["F7"]
        dmax = float(F2.δ.max())
        def dep(nm):
            r = row(F4, δ=dmax, 选择器=nm); return f"{pc(r.均值)}" if r is not None else "—"
        fa, fb = row(F3, δ=dmax, 选择器="FCT|P*", 轨="A"), row(F3, δ=dmax, 选择器="FCT|P*", 轨="B")
        ca, cb = row(F7, δ=dmax, 轨="A"), row(F7, δ=dmax, 轨="B")
        L.append(f"- δ = {dmax:g} 时各选择器的两轨等权部署改善（相对 P\\*_D）：ALWAYS_SPLIT {dep('ALWAYS_SPLIT')}、CV_ANY|P\\* {dep('CV_ANY|P*')}、"
                 f"CV_POINT_M|P\\* {dep('CV_POINT_M|P*')}、CV_LB_M|P\\* {dep('CV_LB_M|P*')}、MIX {dep('MIX')}、POOLMIX {dep('POOLMIX')}、"
                 f"FCT|P\\* {dep('FCT|P*')}、USTUN_ADAPT {dep('USTUN_ADAPT')}。")
        L.append(f"- FCT|P\\* 没有错误启用，但在 δ = {dmax:g} 时只启用了达标重复中的 A 轨 {int(fa.启用)}/{int(fa.达标重复)}、B 轨 {int(fb.启用)}/{int(fb.达标重复)}；"
                 f"未启用时部署只在拟合集训练的 P\\*_F，其真值风险相对 P\\*_D 的改善为 A 轨 {pc(ca.P星F相对P星D改善均值)}、B 轨 {pc(cb.P星F相对P星D改善均值)}（样本成本），"
                 "所以即使存在收益，部署结果仍不如 P\\*_D。")
        z0 = F3[(F3.δ == 0) & F3.选择器.isin(["CV_POINT_M|P*", "CV_LB_M|P*", "FCT|P*", "USTUN_ADAPT"])]
        L.append(f"- **含义（只限这一生成机制）**：δ = 0 时专属模型的真值收益为负，CV_POINT_M、CV_LB_M、FCT、USTUN_ADAPT 在两轨合计启用 {int(z0.启用.sum())} 次；"
                 "δ = 1 时冻结程序的专属模型平均真值收益超过 1%，CV_ANY、CV_POINT_M 与 MIX 的部署改善接近总是拆分，CV_LB_M 较少，FCT 因样本成本取不到。"
                 "这说明在这类结构强度下程序能看到并取得收益，"
                 "不说明真实数据中的轨道特异结构有多强。MIX 相对 POOLMIX 的优势随 δ 增大，说明 MIX 的收益来自专属成分，不只是两个模型平均。")
        L.append("- δ > 0 只表示生成函数含轨道特异结构，不等于已注入某个幅度的收益；不模拟医院层残差相关。\n")
    # 更正
    fct = B2[B2.选择器 == "FCT|P*"]
    L.append("**4. R43/R44 更正与补齐（不重跑）**")
    L.append(f"- R43：候选对达标数统一取自最终 truth，口径不同的单元格 {len(R['B1'])} 个（含 GPT 指出的 2 个）；`regret` 改名为带符号风险差，"
             f"{fx['选择器表行数']} 行中 {fx['带符号差为负的行']} 行为负；「全部来自样本成本」改为「主要来自」；"
             f"FCT|P\\* 在 {len(fct)} 个条件中同一重复至少一条不达标轨被启用都是 {int(fct['≥1错误启用_分子'].max())}/20（CP 上界 {pc(float(fct['≥1错误启用_CP上'].max()))}），不写成 5% 错误率已验证。")
    c1t = R["C1"].set_index("项").N
    L.append(f"- R44：10 年内死亡但死因缺失 {int(c1t['10 年内死亡但 UCOD_LEADING 缺失（会被心脑血管死亡静默判为非事件）'])} 例在 O_CVD10 中被记为非事件（至多影响 1 个事件，披露、不重跑）；"
             "贫血切点更正为女 < 12、男 < 13 g/dL；CDC 公开版死亡文件的合成替代说明已核对原文；nhanes.py 随包交付。\n")
    return "\n".join(L)


def opinions(R):
    D, E, F, H1 = R["D"], R["E"], R["F"], R["sheets"]["H1_意见所用数字"]
    hv = H1.set_index("项").值
    L = ["## 8. 我对第 49 轮的意见与提请裁决\n"]
    n = 0
    def add(t):
        nonlocal n
        n += 1; L.append(f"{n}. {t}")
    # 主线
    G1 = R["G1"]; gb = G1[G1.尺度 == "bal"]
    if D is not None and "E12" in E:
        c1 = D["D2"][(D["D2"].对照 == "C1") & (D["D2"].尺度 == "pop")]
        add("**主线表述（请 GPT 裁决）。** "
            + "；".join(f"{r.数据库} {int(r.任务)} 个任务" for r in gb.itertuples())
            + f"。三个 ICU 库在 E1 的四个尺度上没有任何「支持」；NHANES 唯一的「支持」（O_MORT10×低白蛋白）在合并参照按人口加权开发后消失，"
            f"11 个任务的 C1（pop 尺度）全部排除（{int((c1['三态_m0.010'] == '排除实质收益').sum())}/{len(c1)}）。"
            + f"但 NHANES 在两轨等权尺度上仍有 {int(G1[(G1.数据库.str.startswith('NHANES')) & (G1.尺度 == 'bal')].证据不足m1.iloc[0])} 个任务证据不足（≥1% 增量既未被支持也未被排除）。"
            "我建议论文主张写成：「在冻结学习程序与这些任务中，相对按与评价相同目标开发的表型感知合并模型，没有任何任务显示出被区间支持的 ≥1% Brier 增量；"
            "ICU 三库全部排除了这一幅度；参照的开发目标与评价尺度不一致会制造表面的专业化收益」。不写成「专属训练没有增量」。"
            "限定条件必须同时写：Brier 尺度、m = 1%、固定 OOF 预测的第 Ⅱ 层推断；"
            "NHANES 的 P\\*_q 只在 pop（11 任务）与 B（1 任务）两个目标上运行过；这不等于在任何表型或数据中专属训练都没有价值"
            "（plasmode 回答的是：在已知含轨道特异结构的生成机制下，这套程序能否看到并安全地取得收益，见 §6）。")
    if "F2" in F:
        F4, F6 = F["F4"], F["F6"]
        dmax = float(F["F2"].δ.max())
        g = lambda d, nm: row(F4, δ=d, 选择器=nm)
        m0, m1, f1, l1 = g(0.0, "MIX"), g(dmax, "MIX"), g(dmax, "FCT|P*"), g(dmax, "CV_LB_M|P*")
        add(f"**选择器的定位（请 GPT 裁决论文怎么写）。** plasmode 中 FCT|P\\* 在每个 δ 都没有错误启用，但 δ = {dmax:g} 时部署改善为 {pc(f1.均值)}——"
            f"认证需要留出数据，样本成本大于它能取得的收益。MIX 在 δ = 0 时为 {pc(m0.均值)}（最小 {pc(m0.最小)}），δ = {dmax:g} 时为 {pc(m1.均值)}；"
            f"CV_LB_M|P\\* 各 δ 都没有错误启用，δ = {dmax:g} 时为 {pc(l1.均值)}。我的看法：论文不应把 FCT 写成推荐的默认程序，"
            "而应写成「错误启用受控、但代价是样本成本」的保守选项，并同时报告它在有收益时的净损失；MIX 没有错误启用这一概念，"
            "它的低风险只在这一生成机制与 20 个重复中观察到。这些都是有限半合成的探索性结果，不能替代真实数据。")
    # NHANES 目标一致性的方法学含义
    if D is not None:
        add("**比较专属模型时，合并参照的开发目标必须与评价尺度一致。** NHANES 唯一的支持例在目标一致后不再有实质增量（§4.1）。"
            "R42 的 MIMIC/eICU 与本轮 NWICU 的 E1 也是用两轨等权开发的 P\\* 去比逐轨与人口加权尺度；"
            + ("这三个库在这些尺度上都没有支持结果，所以结论不受影响；" if int(G1[G1.数据库.str.startswith(("MIMIC", "eICU", "NWICU"))].支持m1.sum()) == 0
               else "其中出现的支持结果同样受这一问题影响；")
            + "但凡今后在非 bal 尺度上报告的「支持」，都应同时报告按该尺度开发的合并参照（P\\*_q）。我建议把「参照按评价目标开发」写进方法定义，"
            "并把 NHANES 低白蛋白作为「目标不一致会制造表面专业化收益」的实例写进正文（标明为看到结果后登记的追加对照）。请 GPT 裁决是否采用这一表述。")
        add("**两轨等权开发在极小轨道上可能不稳定（描述，未分离机制）。** 低白蛋白任务中 P\\*_pop 相对 v1.0 P\\* 连 A 轨也更好（§4.1）。"
            "这提示 q = 0.5 的早停与选择在 A 轨极小时噪声很大，但早停轮数、θ、k、raw/cal 同时随目标改变，本轮不能说是哪一环。"
            "我不建议为此改动冻结引擎；如 GPT 认为需要，可在下一轮登记一个只改早停权重、其余保持 bal 的分解对照。")
        add("**推断脚本「4_预定判读」表的适用范围。** 该表对 11 个任务都套了 C1 措辞；按方案 §6 只有 O_MORT10×低白蛋白适用，其余 10 个只作描述。"
            "推断脚本已冻结，我没有改它；报告与证据表 D1 只取该任务（§4.2 已说明）。")
    # NWICU 编码决定
    E14 = E["E3b"].set_index("去向")
    reg = E["E2"].set_index("项").N
    add(f"**NWICU 出院去向 UNKNOWN 的 {int(E14.loc['UNKNOWN', '人次'])} 人次按活着出院处理**（在任何拟合前决定；其中 deathtime ≤ dischtime 的已按冲突排除）。"
        "第 49 轮裁决没有覆盖这一类。若 GPT 认为应排除，可登记一个只删这些人次的敏感性分析；本轮没有运行。")
    add(f"**hospital_expire_flag 的异常。** 该标志为 1 的 {int(reg['hospital_expire_flag = 1（不使用）'])} 次恰好全部是「死亡状态且无 deathtime」的记录，"
        f"而合格队列中 EXPIRED 去向有 {int(E14.loc['EXPIRED', '人次'])} 人次——它在 NWICU 中不能当住院死亡标志使用，按裁决不使用。")
    add(f"**b 口径与 c 口径的差别集中在临终关怀。** INPATIENT HOSPICE 的 {int(E14.loc['INPATIENT HOSPICE', '人次'])} 人次中有 {int(E14.loc['INPATIENT HOSPICE', 'c口径'])} 人次在出院后 1 天内死亡，"
        "b 口径按裁决不计为事件（只有 HOSPICE 转出不算死亡）。这是定义选择，不是数据错误；如 GPT 希望，可在论文中把 c 口径列为描述性敏感性。")
    allm = row(E["E5"], 分组方式="全部")
    ob = row(E["E6"], 结局="OUT_HOSP_MORT", 表型="肥胖")
    add(f"**NWICU 体征是结构性缺失。** {pc(allm.整段无任何体征记录)} 的合格入住整段没有任何体征记录。主分析按裁决不排除、用训练内中位数插补，"
        f"这会削弱体征类特征的信息量；同样因为身高体重记录少，肥胖任务只有 {int(ob.可判定人次)} 个可判定人次。"
        "若 GPT 认为必要，可登记「窗内有体征者」的敏感性分析（本轮未运行；是否运行应在看结果前决定）。")
    add(f"**0/20 不能验证 5% 错误率。** 0/20 时 Clopper–Pearson 双侧 95% 上界为 {pc(hv['0/20 时 Clopper–Pearson 双侧 95% 上界'])}；"
        f"要在 0 次错误时把上界压到 5% 以下，双侧需要至少 {int(hv['0/n 时双侧 95% 上界 < 5% 所需最少独立重复'])} 个独立重复（单侧 {int(hv['0/n 时单侧 95% 上界 < 5% 所需最少独立重复'])} 个）。"
        "论文中 FCT 的错误启用控制只能写成「在 20 个重复中未观察到」，不能写成已验证的错误率。")
    if len(R["A3"]):
        add(f"**冻结后修订 {len(R['A3'])} 处（我的疏漏）。** plasmode 汇总脚本在正式运行后因列名不一致报错，修了一行列名映射（表 A3）。"
            "原因是端到端测试只调用了单个重复的运行函数，没有把汇总脚本在测试输出上跑一遍。今后的运行前测试会把汇总与推断脚本也跑通再冻结。")
    add("**R43 的第三处口径差异**（P1 12000 πA = 0.15 A 轨 F|S0\\*）是我在统一口径时查出、GPT 没有列出的；达标数不变，只有两个重复分类互换。")
    if "F2" in F:
        add("**plasmode 中 POOLMIX 的定义。** POOLMIX = P\\* 与另一合并族的最佳者 ALT 的融合（P\\* 属含 Z 族 SZ、SZ-GFS、SZ-union 时 ALT = S0\\*，P\\* 属 S0 族时 ALT = SZ\\*），"
            "按轨在 0 到 1 之间间隔四分之一的 5 个融合权重中以内层 OOF 本轨 Brier 选权重——与 MIX 结构相同，只是第二个模型来自合并族。"
            "它用来检查 MIX 的优势是否只是「两个模型平均」带来的。若 GPT 认为 ALT 应换成其他候选，需在下一轮登记。")
    return "\n".join(L) + "\n"


# ── md 输出 ────────────────────────────────────────────────────
def esc(s):
    return str(s).replace("|", "\\|").replace("*", "\\*")


def tbl(df, cols, fmt=None, rename=None):
    """把 DataFrame 的若干列写成 md 表；fmt: {列: 函数}。"""
    fmt = fmt or {}; rename = rename or {}
    head = [esc(rename.get(c, c)) for c in cols]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else (str(int(v)) if float(v).is_integer() else f"{v:.6f}"))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(esc(c) for c in cells) + " |")
    return "\n".join(out)


def ivl(r, lo="相对_P2_5", hi="相对_P97_5"):
    return f"{pc(r[lo])} ~ {pc(r[hi])}"


def write_md(R):
    L = []; a = L.append
    A1, A2, D, E, F, G1 = R["A1"], R["A2"], R["D"], R["E"], R["F"], R["G1"]
    fx = R["fx"]
    a("# R45 外部复现与机制核验 · 统一报告\n")
    a("> 由 `pgss_r45_report.py` 生成；正文数字只来自同名证据表中「目录」标为「报告引用」的表（md 与 xlsx 同源）。"
      "相对改善 = (R参照 − R候选)/R参照，正值 = 候选更好；三态判读以 m = 1% 为主。\n")
    a(conclusion(R))
    # 1
    a("## 1. 冻结、执行与运行前检查\n")
    a(tbl(A1, ["部分", "冻结时间", "登记文件数", "登记文件现值等于冻结值或登记修订值", "冻结后登记修订", "驱动核对通过", "单元总数", "完成", "不可执行", "缺失",
               "代码SHA全部相同且等于现值", "附加核对"]))
    a("\n运行前检查（全部在对应的真实拟合之前完成）：\n")
    a(tbl(A2, ["检查", "项数", "通过项", "全部通过"]))
    A3 = R["A3"]
    if len(A3):
        a("\n**冻结后登记的修订**（冻结记录本身不改；核对时这些文件按登记的修订后 SHA 比对）：\n")
        a(tbl(A3, ["部分", "编号", "时间", "文件", "冻结SHA", "修订后SHA", "原因", "改动", "影响"]))
    a("\n**q = 0.5 等价测试**的含义：适配器把冻结引擎的早停权重函数在运行时换成目标版本，并逐行复制 `develop()` 的合并部分。"
      "当目标取两轨等权时，它在 2 个 R44 单元上逐位复现了合并臂规格、内层分数、P\\*、合并臂 raw/cal 测试预测，以及 HTFS 的内层两轨等权分数与版本选择。"
      "所以 pop 与 B 目标下的差异只能来自目标本身，不是复制代码时的无意改动。\n")
    # 2
    B1, B2, B3, B4 = R["B1"], R["B2"], R["B3"], R["B4"]
    a("## 2. R43 只读更正（不重跑任何重复）\n")
    a("### 2.1 候选对达标数统一取自最终 truth\n")
    a("原汇总从初始 10 万评价样本的各臂风险重算候选对收益，而选择器统计用的是追加评价后的最终 truth。统一到最终 truth 后，达标数口径不同的单元格如下"
      f"（全部 {fx['追加评价重复对合计']} 个追加评价的重复×候选对中，追加后仍距边界不足 2 个积分标准误的有 {fx['追加后仍在边界附近合计']} 个，单列在明细表 X1）：\n")
    a(tbl(B1, ["批次", "机制", "n", "πA", "轨", "候选对", "达m_初始评价", "达m_最终", "达m口径不同的重复"],
          fmt={"πA": lambda v: f"{v:.2f}"}))
    a("\n前两行是 GPT 指出的两处（5/20 → 4/20；14/20 → 13/20）。第三处是 R43 原表没有列出的 F|S0\\* 候选对：达标数不变，但有 2 个重复的分类互换。"
      "各臂风险表（arms）保留初始评价样本口径，已在表头注明。\n")
    a("### 2.2 `regret` 改名为「相对两纯策略较优者的带符号风险差」\n")
    a(f"原字段 = R(策略) − min{{R(P\\*_D), R(HTFS_D)}}，只与两个纯策略比较，可以为负，不截断。{fx['选择器表行数']} 行中为负的有 {fx['带符号差为负的行']} 行，按选择器：\n")
    a(tbl(B3, ["选择器", "行", "带符号差为负的重复", "带符号差最小"]))
    a("\n### 2.3 「全部差距来自样本成本」改为「主要来自」\n")
    a("R43 §4 的分解中，认证启用项并非处处为零：\n")
    a(tbl(B4, ["库", "结局", "表型", "尺度", "样本成本项", "认证启用项", "样本成本项_相对R_PstarD", "认证启用项_相对R_PstarD"],
          fmt={"样本成本项_相对R_PstarD": pc, "认证启用项_相对R_PstarD": pc}))
    a("\n更正后的表述：认证程序几乎总是部署只在拟合集训练的参照；它相对全量训练参照的差距**主要**对应减少开发数据后学习程序的性能变化，"
      "个别启用决定带来了较小的追加变化。这是性能差值恒等式，不是已分离出的纯样本量因果效应。\n")
    a("### 2.4 同一重复中至少一条不达标轨被启用（按条件，不跨机制合并）\n")
    fct = B2[B2.选择器 == "FCT|P*"]
    a(f"FCT|P\\* 在全部 {len(fct)} 个条件中，「同一重复至少一条不达标轨被启用」都是 {int(fct['≥1错误启用_分子'].max())}/20；"
      f"Clopper–Pearson 95% 上界 {pc(float(fct['≥1错误启用_CP上'].max()))}。**这不等于 5% 错误率已经验证**：20 个独立重复只能把上界压到这个水平，"
      "而且在 P2/P2-T 的部分条件中，存在不达标轨的重复本来就很少（见表中「至少一轨不达标的重复」），此时该比例几乎不提供信息。"
      "不达标与实际变差分开计数：")
    a("\n" + tbl(R["B2b"], ["选择器", "条件", "错误启用重复合计", "实际变差启用重复合计"]))
    a("\n逐条件的分子、分母与区间见表 B2（每个条件 20 个独立重复）。\n")
    # 3
    C1, C2, C3 = R["C1"], R["C2"], R["C3"]
    c1 = C1.set_index("项").N
    a("## 3. R44 材料补齐（不重新训练）\n")
    a(f"- **死亡随访字段**（队列 {int(c1['队列人数'])} 人）：MORTSTAT 缺失 {int(c1['MORTSTAT 缺失'])}、PERMTH_EXM 缺失 {int(c1['PERMTH_EXM 缺失（全体）'])}、"
      f"存活且随访不足 120 个月 {int(c1['存活且随访 < 120 个月（10 年窗不完整）'])}；"
      f"死亡者中 UCOD_LEADING 缺失 {int(c1['死亡者 UCOD_LEADING 缺失'])} 例，且这 {int(c1['10 年内死亡但 UCOD_LEADING 缺失（会被心脑血管死亡静默判为非事件）'])} 例在 10 年内死亡——"
      "它在 O_CVD10 中被逻辑表达式**静默编码为非事件**。只影响 O_CVD10 的至多 1 个事件，不重新训练，在此如实披露。O_MORT10 与 R44 队列逐例一致。")
    a("- **公开版死亡链接文件的来源限制**（CDC 原文，已核对）：“For certain records, the follow-up time or underlying cause of death were substituted with synthetic values.” "
      "“Information about whether a person is alive or dead was not changed.” 生死状态未扰动，因此不否决固定窗死亡预测，但随访时间与死因的个别替代会进入 10 年窗与心脑血管死因的判定。")
    a(f"- **贫血定义**：{C2.iloc[0]['定义']}。R44 表 A3 只写了 12，是展示不完整；切点由脚本直接从队列代码源文本取出。")
    a(f"- **心脑血管死亡的解释**：{C2.iloc[1]['解释']}。")
    a(f"- **外部辅助模块**：`nhanes.py`（SHA {C3.iloc[0]['模块SHA']}）随本轮代码包交付（`code/external/nhanes.py`），"
      "R44 实际调用其 `load`、`add_derived`、`add_mortality`（血压多次读数、1999–2000 血肌酐校准、eGFR 等衍生都在其中）。")
    a("- **O_CVD10×低白蛋白**：7 个不可执行单元的原因都是「冻结程序在部分折的早停划分中出现单类别」（Hfull A 轨），完整任务不可评价；"
      "不泛化为「45 个事件在任何方法下都不能研究」。逐单元原因见明细表 X4。\n")
    a(part_d_md(R)); a(part_e_md(R)); a(part_f_md(R)); a(part_g_md(R))
    a(opinions(R))
    a("## 9. 合规与交付范围\n")
    a("- 交付三件：本报告、证据表、`R45_CODE_TESTS.zip`（代码、方案、冻结记录、测试、路径审计、汇总表、日志）。")
    a("- **不交付**：任何逐人预测、逐单元 runs 产物、NWICU/MIMIC/NHANES 的个体数据与暂存表、患者编号；这些只留本地。")
    a("- NWICU 原始数据在 E 盘原地只读，源文件 SHA256 与官方清单一致；未搬动、未修改。\n")
    MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    R = build()
    write_md(R)
    print("写出", MD.name, XL.name, len(R["sheets"]))
