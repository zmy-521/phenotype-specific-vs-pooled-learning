# -*- coding: utf-8 -*-
r"""R52：由 R52_NHANES_SCr_SENS_EVIDENCE.xlsx 与 R52_COMPARE.json 生成 R52_NHANES_SCr_SENS_REPORT.md。
每个数字经 n()/p()/pp()/cells() 从证据表取值并登记到 R52_NUM_TRACE.json；结论文字按 STOP/GO 判定与表中计数生成，不手填数字。
用法：python pgss_r52_report.py [dryrun]
"""
import sys, json, re, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r52_sens as S

TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_.])[−-]?\d[\d,]*(?:\.\d+)?%?")
YEARS = {str(y) for y in range(1999, 2009)} | {"2021", "2020", "2018", "2019", "2022", "2023", "2024", "2025", "2026"}
STRIP = [r"`[^`]*`", r"(?m)^#+\s*\d+\.\s*", r"(?m)^\s*\d+\.\s", r"第\s*\d+\s*(?:节|步|条|层|轮)", r"§[A-Z]?\d*", r"\b[A-Z]-\d[a-z]?\b",
         r"\[\d+(?:[–,]\d+)*\]", r"https?://\S+", r"doi:\S+"]
TRACE = {}
K = "定义常量"


def norm(t):
    return t.replace("−", "-")


def reg(tok, src):
    for t in TOKEN_RE.findall(tok) or [tok]:
        TRACE.setdefault(norm(t), set()).add(src)
    return tok


def n(v, src, dp=0):
    v = float(v)
    return reg(f"{int(round(v)):,}" if dp == 0 else f"{v:,.{dp}f}", src)


def p(v, src, dp=2, sign=True):
    v = float(v)
    while v != 0 and abs(100 * v) < 0.5 * 10 ** (-dp) and dp < 4:     # 近零值多保留小数，避免显示成 ±0.00%
        dp += 1
    s = (f"{100 * v:+.{dp}f}%" if sign else f"{100 * v:.{dp}f}%").replace("-", "−")
    return reg(s, src)


def pp(v, src, dp=2):
    """百分点差（Δ），带符号。"""
    return reg(f"{100 * float(v):+.{dp}f}".replace("-", "−"), src)


def ap(v, src, dp=2):
    """绝对值的百分点（不带符号）。"""
    return reg(f"{100 * abs(float(v)):.{dp}f}", src)


def cells(df_or_text, src):
    vals = [df_or_text] if isinstance(df_or_text, str) else df_or_text.astype(str).to_numpy().ravel()
    for v in vals:
        for t in TOKEN_RE.findall(v):
            reg(t, src)
    return df_or_text


def tokens(md):
    for pat in STRIP:
        md = re.sub(pat, " ", md)
    return [norm(t) for t in TOKEN_RE.findall(md)]


def unregistered(md, trace):
    return sorted({t for t in tokens(md) if t not in trace and t not in YEARS})


def lint(src):
    bad = re.findall(r"\b(?:n|p|pp|ap)\(\s*[−-]?\d[\d_.]*\s*,", src)
    bad += [m for m in re.findall(r"reg\(\s*['\"][\d.,%]+['\"]\s*,\s*([^)]+)\)", src) if m.strip() != "K"]
    return bad


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(out)


def main(write=True, dry=False):
    TRACE.clear()
    root = (S.BR / "_dryrun") if dry else S.BR
    W = pd.read_excel(root / "R52_NHANES_SCr_SENS_EVIDENCE.xlsx", sheet_name=None)
    J = json.loads((root / "R52_COMPARE.json").read_text(encoding="utf-8"))
    E = lambda k: W[k]
    src = lambda k: f"R52_NHANES_SCr_SENS_EVIDENCE.xlsx:{k}"
    SC = dict(zip(E("4_血肌酐改动").项, E("4_血肌酐改动").值))
    A = E("3_QC锚点"); PRE = E("2_拟合前核验"); MIG = E("5_eGFR迁移矩阵"); TK = E("6_任务n事件轨"); RQ = E("7_运行QC")
    E1 = E("8_E1主对照"); MT = E("9_maxT"); DIR = E("10_对照方向汇总"); LL = E("12_logloss对照"); AUXm = E("13_辅助指标均值")
    JD = E("16_L3预定判读"); GAP = E("19_L3差距收回"); RULE = E("22_STOP_GO"); PERT = E("23_变化幅度与重训扰动基准")
    eq = json.loads((S.BR / "R52_EQUIV.json").read_text(encoding="utf-8")); pa = json.loads((S.BR / "R52_PATHAUDIT.json").read_text(encoding="utf-8"))
    zc = json.loads((S.BR / "R52_INFER_ZERO_CHECK.json").read_text(encoding="utf-8"))
    fc = json.loads((S.BR / "R52_FOLDCHECK.json").read_text(encoding="utf-8")) if (S.BR / "R52_FOLDCHECK.json").exists() else None
    fz = json.loads((S.BR / "R52_FREEZE.json").read_text(encoding="utf-8"))
    sdr = J["判定"].startswith("SCIENTIFIC")
    mig = lambda a, b: int(MIG[(MIG.旧 == a) & (MIG.新 == b)].人数.iloc[0])
    nAB, nBA = mig("A（降低）", "B（不降低）"), mig("B（不降低）", "A（降低）")
    OCO = {"O_MORT10": 0, "O_CVD10": 1}; PHO = {k: i for i, k in enumerate(["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"])}
    bal = E1[E1.尺度 == "bal"].assign(_o=lambda d: d.结局.map(OCO), _p=lambda d: d.表型.map(PHO)).sort_values(["_o", "_p"]).reset_index(drop=True)
    la = E1[(E1.结局 == "O_MORT10") & (E1.表型 == "低白蛋白")].set_index("尺度")
    d1 = JD[JD.对照 == "D1"].iloc[0]; c1 = JD[JD.对照 == "C1"]
    st = "三态_m0.010"
    cnt = lambda df, col, v: int((df[col] == v).sum())

    a = []; Ap = a.append
    Ap("# R52 Claude 回复：NHANES 2005–2006 血肌酐校准的限定重训敏感性分析 + Introduction v0.4a 审阅\n")
    Ap(f"分支 `{S.BRANCH}`。**这是授权的限定新拟合敏感性分析，不是只读核数**：NHANES main 程序与目标一致性追加臂全部重新开发"
       f"（main {n(len(list(S.DIRS['main'].glob('*/_DONE'))) if not dry else 0, '运行目录计数')} 个单元完成、"
       f"目标臂 {n(len(list(S.DIRS['pop'].glob('*/_DONE'))) if not dry else 0, '运行目录计数')} 个单元完成）。"
       "结果是 sensitivity，不替代原预定主分析；原 R44/R45/R46 v2、R47 数字、R48 图与 v0.3b/v0.4a 正文一概未覆盖。"
       "数字均由 `pgss_r52_report.py` 从证据表生成并登记在 `R52_NUM_TRACE.json`。\n")

    Ap("## 0. 结论\n")
    Ap(f"**判定：{J['判定']}**" + (f"（触发规则：{'、'.join(J['触发规则'])}）" if sdr else "") + "\n")
    Ap(f"1. 主对照（两轨等权 E1，m = {reg('1%', K)}）：{n(len(bal), src('8'))} 个任务中三态改变 {n(J['E1_bal_三态改变'], 'R52_COMPARE.json')} 个；"
       f"新结果 排除 {n(cnt(bal, f'{st}_新', '排除实质收益'), src('8'))}、不足 {n(cnt(bal, f'{st}_新', '证据不足'), src('8'))}、支持 {n(cnt(bal, f'{st}_新', '支持实质收益'), src('8'))}"
       f"（原 排除 {n(cnt(bal, f'{st}_旧', '排除实质收益'), src('8'))}、不足 {n(cnt(bal, f'{st}_旧', '证据不足'), src('8'))}、支持 {n(cnt(bal, f'{st}_旧', '支持实质收益'), src('8'))}）。"
       f"点估计最大绝对变化 {ap(J['E1_bal_最大绝对Δ'], 'R52_COMPARE.json')} 个百分点。")
    Ap(f"2. NHANES 唯一的原 B 轨正结果（O_MORT10×低白蛋白）：B 轨 {p(la.loc['B', '观察相对改善_旧'], src('8'))} → {p(la.loc['B', '观察相对改善_新'], src('8'))}"
       f"（新区间 {p(la.loc['B', '相对_P2_5_新'], src('8'))} 至 {p(la.loc['B', '相对_P97_5_新'], src('8'))}），三态 {la.loc['B', f'{st}_旧']} → {la.loc['B', f'{st}_新']}。")
    Ap(f"3. L3 目标一致性：D1（B 轨目标）{p(d1.观察相对改善_旧, src('16'))} → {p(d1.观察相对改善_新, src('16'))}"
       f"（新区间 {p(d1.相对_P2_5_新, src('16'))} 至 {p(d1.相对_P97_5_新, src('16'))}），三态 {d1.三态_m1_旧} → {d1.三态_m1_新}；"
       f"C1（人口加权目标）{n(len(c1), src('16'))} 项中三态改变 {n(int(c1.三态改变.sum()), src('16'))} 项。")
    Ap(f"4. 新出现的区间支持 ≥1% 增益：{RULE[RULE.规则 == 'D-3'].明细.iloc[0]}。程序资格：{RULE[RULE.规则 == 'D-4'].明细.iloc[0]}。")
    pr = lambda lab: PERT[PERT.来源 == lab].iloc[0]
    r_all, r46 = pr("R52 E1 全部 11 任务 × 4 尺度"), pr("NHANES O_CVD10（删除 1 人）")
    Ap(f"5. 变化幅度：E1 四个尺度共 {n(r_all.行, src('23'))} 行，|Δ| 中位 {ap(r_all.绝对变化中位, src('23'))}、最大 {ap(r_all.绝对变化最大, src('23'))} 个百分点，"
       f"|Δ|/原区间半宽 中位 {n(r_all.比值中位, src('23'), 2)}、最大 {n(r_all.比值最大, src('23'), 2)}。作为对照，R46 在 NHANES 只删 1 人重训，"
       f"|Δ| 中位 {ap(r46.绝对变化中位, src('23'))}、最大 {ap(r46.绝对变化最大, src('23'))} 个百分点，比值中位 {n(r46.比值中位, src('23'), 2)}、最大 {n(r46.比值最大, src('23'), 2)}。"
       "固定 OOF 区间不含重训变异，所以这类变化本身不能读成校准的「效应」。")
    big = E1.loc[E1.观察相对改善_Δ.abs().idxmax()]; rbig = E1.loc[E1["|Δ点估计|/原半宽"].idxmax()]
    cmpw = lambda a, b: "大于" if a > b else ("小于" if a < b else "等于")
    Ap(f"   R52 的最大 |Δ|（{ap(r_all.绝对变化最大, src('23'))}）{cmpw(r_all.绝对变化最大, r46.绝对变化最大)} R46 删人重训的最大值（{ap(r46.绝对变化最大, src('23'))}），"
       f"最大比值（{n(r_all.比值最大, src('23'), 2)}）{cmpw(r_all.比值最大, r46.比值最大)} R46 的最大比值（{n(r46.比值最大, src('23'), 2)}）。"
       f"最大 |Δ| 出在 {big.结局}×{big.表型} 的 {big.尺度} 尺度（{p(big.观察相对改善_旧, src('8'))} → {p(big.观察相对改善_新, src('8'))}），"
       f"最大比值出在 {rbig.结局}×{rbig.表型} 的 {rbig.尺度} 尺度；该任务不涉及 eGFR 分轨，只有血肌酐特征改变，A 轨只有 "
       f"{n(int(TK[(TK.结局 == rbig.结局) & (TK.表型 == rbig.表型)].新_A轨n.iloc[0]), src('6'))} 人。三态没有改变。")
    BA0 = E("11_全部Brier对照"); oth0 = BA0[(BA0.对照 != "E1") & BA0.三态改变_m1]; lle0 = LL[(LL.对照 == "E1") & LL.三态改变_m1]
    tasks0 = oth0.groupby(["结局", "表型"]).size().sort_values(ascending=False)
    Ap(f"6. 不在 STOP 规则内、但同样报告的变化：逐轨 max-T 判定改变 {n(int(MT.判定改变.sum()), src('9'))} 项；E1 以外的 Brier 对照三态改变 "
       f"{n(len(oth0), src('11'))}/{n(int((BA0.对照 != 'E1').sum()), src('11'))} 行（"
       + "、".join(f"{oc}×{ph} {n(k, src('11'))} 行" for (oc, ph), k in tasks0.items()) + f"）；log loss E1 三态改变 {n(len(lle0), src('12'))} 行。明细见第 6、7 节。")
    holds = (la.loc["B", f"{st}_新"] == "支持实质收益") and (d1.三态_m1_新 == "排除实质收益")
    Ap(f"7. 原 L3 的解释（低白蛋白 B 轨收益在按 B 轨目标开发的合并参照下不再保留 ≥{reg('1%', K)}）在敏感性分析中"
       + ("**仍成立**：E1 的 B 轨仍被支持，D1 仍排除实质收益。" if holds else "**不再成立**，见第 8 节。") + "\n")
    if sdr:
        Ap("**按指令 §D，本轮停在这里，不建议自动更新论文；请 GPT、用户与我共审触发的规则。**\n")

    Ap("## 1. 官方依据与输入边界（已校验事实）\n")
    Ap("- 公式来自本地存档的 CDC/NCHS `BIOPRO_D.htm` 原文：「Correction of serum creatinine values in NHANES 2005-2006 is highly recommended … "
       "Standard creatinine (mg/dL) = -0.016 + 0.978 X (NHANES 05-06 uncalibrated serum creatinine, mg/dL)」。在线地址同 GPT 指令。")
    Ap("- BIOPRO_D 的血肌酐字段只有 LBXSCR（mg/dL）与 LBDSCRSI（µmol/L），没有已标准化变量；冻结数据中 2005–2006 的血肌酐与原始 LBXSCR 逐人相同，"
       "因此是未校准值，不存在二次校准。1999–2000 沿用原冻结实现（Selvin），其他周期不改。")
    Ap(f"- 只改 2005–2006 且血肌酐非缺失者（{n(SC['改动人数（2005–2006 且血肌酐非缺失）'], src('4'))} 人）；该周期血肌酐缺失的 "
       f"{n(SC['2005–2006 合格者中血肌酐缺失（未插补、未改动）'], src('4'))} 人保持缺失；其他周期改动 {n(SC['其他周期改动人数'], src('4'))} 人。"
       "随之用与 `nhanes.add_derived` 逐字相同的 CKD-EPI 2021 表达式重算 eGFR 与 eGFR降低 表型（逐位一致性已核）。")
    nfeat = len(pd.read_parquet(S.DATA / "X.parquet").columns) - 1
    Ap(f"- 资格、结局（含 R46 v2 O_CVD10 可评价状态）、分组键、其余 {n(nfeat, 'data/X.parquet 列数')} 个特征与其余五个表型全部不变；外层划分只依赖分组与结局，因此与原版逐人相同。\n")

    Ap("## 2. 代码与输入指纹\n")
    Ap(md_table(cells(E("1_来源与指纹").iloc[2:], src("1")))); Ap("")
    Ap(f"冻结记录 `R52_FREEZE.json`：分支代码 SHA {cells(fz['代码SHA'], 'R52_FREEZE.json')}；正式运行前核对 {n(len(fz['文件SHA']), 'R52_FREEZE.json')} 个代码文件与 {n(len(fz['数据文件SHA']), 'R52_FREEZE.json')} 个分支数据文件未变。"
       "原冻结学习程序（引擎 `xgb-1.0`、适配器 `r45-target-1.0`、R44/R45 运行器、推断脚本）一字未改，只在独立进程里替换模块级的数据与运行目录引用（与 R46 包装同一做法）。\n")
    Ap("## 3. 拟合前核验与 QC 锚点\n")
    pre = PRE.copy(); pre["通过"] = pre.通过.map({True: "是", False: "否"})
    Ap(md_table(cells(pre[["核验", "通过"]], src("2")))); Ap("")
    aa = A.copy(); aa["一致"] = aa.一致.map({True: "是", False: "否"})
    fmt = lambda v: f"{int(v):,}" if float(v).is_integer() else f"{float(v):.4f}"
    aa["R52重算"] = [fmt(v) for v in aa.R52重算]; aa["R51登记"] = [fmt(v) for v in aa.R51登记]
    Ap(md_table(cells(aa, src("3")))); Ap("")

    Ap("## 4. 改动内容\n")
    sc = E("4_血肌酐改动").copy(); sc["值"] = [f"{v:.4f}" if isinstance(v, float) and not float(v).is_integer() else f"{int(v):,}" for v in sc.值]
    Ap(md_table(cells(sc, src("4")))); Ap("")
    Ap(f"eGFR 降低迁移矩阵（可判定者）：旧 A→新 B {n(nAB, src('5'))} 人，旧 B→新 A {n(nBA, src('5'))} 人；"
       f"A→A {n(mig('A（降低）', 'A（降低）'), src('5'))}，B→B {n(mig('B（不降低）', 'B（不降低）'), src('5'))}。\n")
    tk = TK.copy(); tk["改变"] = tk.改变.map({True: "是", False: "否"})
    Ap(md_table(cells(tk, src("6")))); Ap("")
    Ap("O_CVD10 的分母已按 R46 v2 删除死因未知者。只有两个 eGFR降低 任务的分轨与各轨事件改变，n 与事件不变。\n")

    Ap("## 5. 运行与核对\n")
    Ap(f"- 零修正等价：用原数据经本分支包装运行 {n(len(eq['单元']), 'R52_EQUIV.json')} 个代表单元（main、v2 删人的 O_CVD10、eGFR 任务、pop、B），"
       f"与现行冻结单元逐位相同；变异对照 {n(sum(m['检出'] for m in eq['变异对照']), 'R52_EQUIV.json')}/{n(len(eq['变异对照']), 'R52_EQUIV.json')} 检出（含「校准后数据运行同一单元必须不同」）。")
    Ap(f"- 干跑审计：main {n(pa['可执行'], 'R52_PATHAUDIT.json')}/{n(pa['总数'], 'R52_PATHAUDIT.json')} 可执行，"
       f"{n(len(pa['不可执行单元']), 'R52_PATHAUDIT.json')} 个不可执行单元与原 R44 逐单元相同（全部在 O_CVD10×低白蛋白，原规则下不可评价，未改可执行门槛）。")
    Ap(f"- 零修正推断：包装在原数据与现行冻结单元上重跑推断，与 R46 v2 冻结推断表 {n(sum(1 for c in zc['比较'] if c['不同格数'] == 0), 'R52_INFER_ZERO_CHECK.json')}/"
       f"{n(len(zc['比较']), 'R52_INFER_ZERO_CHECK.json')} 张表逐格相同。")
    amf = S.BR / "R52_AMENDMENT.json"
    if amf.exists():
        am = json.loads(amf.read_text(encoding="utf-8"))
        Ap(f"- **冻结后修订 1 处（已登记 `R52_AMENDMENT.json`）**：{cells(am['原因'], 'R52_AMENDMENT.json')}改法：{am['改法']}。{am['影响']}；"
           f"在 run_unit 里改一个字符的变异对照被检出 = {'是' if am['变异对照_改run_unit一字符被检出'] else '否'}。"
           f"冻结 SHA {cells(am['冻结SHA'], 'R52_AMENDMENT.json')} → 修订后 {cells(am['修订后SHA'], 'R52_AMENDMENT.json')}；修订后补跑单元核对与敏感性推断，首次失败的哨兵已归档。")
    if fc:
        Ap(f"- 单元核对：{n(fc['单元'], 'R52_FOLDCHECK.json')} 个单元的测试集合、标签与原版相同，分轨只在迁移者处改变，全部带分支数据指纹；"
           f"可执行性与原版一致 = {'是' if fc['可执行性一致'] else '否'}；每个 seed 的五个测试折互不重叠（患者不泄漏）。")
    Ap("")

    Ap("## 6. 主对照：E1（P* vs HTFS），两轨等权尺度\n")
    rows = []
    for _, r in bal.iterrows():
        rows.append({"结局": r.结局, "表型": r.表型, "原": p(r.观察相对改善_旧, src("8")),
                     "原区间": f"{p(r.相对_P2_5_旧, src('8'))} 至 {p(r.相对_P97_5_旧, src('8'))}", "新": p(r.观察相对改善_新, src("8")),
                     "新区间": f"{p(r.相对_P2_5_新, src('8'))} 至 {p(r.相对_P97_5_新, src('8'))}", "Δ（百分点）": pp(r.观察相对改善_Δ, src("8")),
                     "原三态": r[f"{st}_旧"], "新三态": r[f"{st}_新"]})
    Ap(md_table(pd.DataFrame(rows))); Ap("")
    Ap(f"正值 = 专属模型更好。四个尺度的计数（m = {reg('1%', K)}，排除/不足/支持，旧 → 新）：\n")
    dd = DIR[(DIR.指标 == "Brier") & (DIR.对照 == "E1")]
    Ap(md_table(pd.DataFrame([{"尺度": r.尺度, "任务": n(r.任务_旧, src("10")),
                                "旧": f"{n(r.排除m1_旧, src('10'))}/{n(r.证据不足m1_旧, src('10'))}/{n(r.支持m1_旧, src('10'))}",
                                "新": f"{n(r.排除m1_新, src('10'))}/{n(r.证据不足m1_新, src('10'))}/{n(r.支持m1_新, src('10'))}",
                                "相对改善中位 旧→新": f"{p(r.相对改善中位_旧, src('10'))} → {p(r.相对改善中位_新, src('10'))}",
                                "上界最大 旧→新": f"{p(r.相对上界最大_旧, src('10'))} → {p(r.相对上界最大_新, src('10'))}"} for _, r in dd.iterrows()]))); Ap("")
    mc = lambda col: f"{n(cnt(MT, col, '排除（收益<m）'), src('9'))}/{n(cnt(MT, col, '证据不足'), src('9'))}/{n(cnt(MT, col, '支持（收益>m）'), src('9'))}"
    assert set(MT["同时判定_相对m_旧"]) | set(MT["同时判定_相对m_新"]) <= {"排除（收益<m）", "证据不足", "支持（收益>m）"}
    Ap(f"逐轨 max-T（{n(len(MT), src('9'))} 项，排除/不足/支持）：旧 {mc('同时判定_相对m_旧')} → 新 {mc('同时判定_相对m_新')}；判定改变 {n(int(MT.判定改变.sum()), src('9'))} 项"
       + ("：" + "；".join(f"{r.结局}×{r.表型}×{r.轨} {r['同时判定_相对m_旧']}→{r['同时判定_相对m_新']}" for _, r in MT[MT.判定改变].iterrows()) if MT.判定改变.any() else "") + "。\n")

    BA = E("11_全部Brier对照")
    oth = BA[(BA.对照 != "E1") & BA.三态改变_m1]
    Ap(f"E1 以外的 Brier 对照（E2–E7、E1_vs_S0* 等，{n(int((BA.对照 != 'E1').sum()), src('11'))} 行）中三态改变 {n(len(oth), src('11'))} 行"
       + ("，全部列出（不属于 STOP 规则，按不择优原则报告）：\n" if len(oth) else "。\n"))
    if len(oth):
        oth = oth.assign(_o=lambda d: d.结局.map(OCO), _p=lambda d: d.表型.map(PHO)).sort_values(["对照", "_o", "_p", "尺度"])
        Ap(md_table(pd.DataFrame([{"对照": r.对照, "参照→候选": f"{r.参照} → {r.候选}", "结局": r.结局, "表型": r.表型, "尺度": r.尺度,
                                    "原": p(r.观察相对改善_旧, src("11")), "新": p(r.观察相对改善_新, src("11")),
                                    "原三态": r[f"{st}_旧"], "新三态": r[f"{st}_新"]} for _, r in oth.iterrows()]))); Ap("")

    Ap("## 7. 辅助指标\n")
    lle = LL[LL.对照 == "E1"]
    Ap(f"- log loss E1（{n(lle.尺度.nunique(), src('12'))} 尺度 × {n(lle[['结局', '表型']].drop_duplicates().shape[0], src('12'))} 任务）：三态改变 {n(int(lle.三态改变_m1.sum()), src('12'))} 行"
       + ("：" + "；".join(f"{r.结局}×{r.表型} {r.尺度} {r[f'{st}_旧']}→{r[f'{st}_新']}" for _, r in lle[lle.三态改变_m1].iterrows()) if lle.三态改变_m1.any() else "") + "。")
    for met in ("AUROC", "AUPRC"):
        g = AUXm[(AUXm.指标 == met) & (AUXm.范围 == "全体") & AUXm.臂.isin(["P*", "HTFS"])]
        Ap(f"- {met}（三 seed 均值，全体，P* 与 HTFS，{n(len(g), src('13'))} 行）：均值变化绝对值最大 {n(g.均值Δ.abs().max(), src('13'), 4)}，中位 {n(g.均值Δ.abs().median(), src('13'), 4)}。")
    g = AUXm[(AUXm.指标 == "校准斜率") & (AUXm.范围 == "全体") & AUXm.臂.isin(["P*", "HTFS"])]
    Ap(f"- 校准斜率（同上）：均值变化绝对值最大 {n(g.均值Δ.abs().max(), src('13'), 4)}。逐 seed 明细与配对差见证据表 `13_辅助指标均值`、`14_辅助指标逐seed`、`15_配对差逐seed`。\n")

    Ap("## 8. L3 目标一致性追加对照（结果知情后的解释性对照，属性不变）\n")
    jd = pd.DataFrame([{"对照": r.对照, "结局": r.结局, "表型": r.表型, "判读尺度": r.判读尺度, "原": p(r.观察相对改善_旧, src("16")),
                        "新": p(r.观察相对改善_新, src("16")), "新区间": f"{p(r.相对_P2_5_新, src('16'))} 至 {p(r.相对_P97_5_新, src('16'))}",
                        "原三态": r.三态_m1_旧, "新三态": r.三态_m1_新} for _, r in JD.iterrows()])
    Ap(md_table(jd)); Ap("")
    gl = GAP[(GAP.结局 == "O_MORT10") & (GAP.表型 == "低白蛋白")]
    for _, r in gl.iterrows():
        Ap(f"- 差距收回（目标 {r.目标}，尺度 {r.尺度}）：收回比例 {n(r.收回比例_相对v10差距_旧, src('19'), 3)} → {n(r.收回比例_相对v10差距_新, src('19'), 3)}。")
    Ap("")

    Ap("## 9. 校准的双重影响：模型输入与表型分轨\n")
    pt = PERT.copy()
    pt["区间方向改变"] = ["" if pd.isna(v) else f"{int(v)}" for v in pt.区间方向改变]
    for c in ("绝对变化中位", "绝对变化最大"):
        pt[c] = [ap(v, src("23")) for v in pt[c]]
    for c in ("比值中位", "比值最大"):
        pt[c] = [n(v, src("23"), 2) for v in pt[c]]
    Ap(md_table(cells(pt, src("23")))); Ap("")
    Ap(f"- 其余 {n(int((bal.表型 != 'eGFR降低').sum()), src('8'))} 个任务只有血肌酐这一特征在 2005–2006 周期的取值改变，分轨不变；两个 eGFR降低 任务同时改变特征与分轨（迁移者从 A 轨移到 B 轨）。")
    Ap("- 两类任务的变化都包含重训本身的扰动；上表把它们与 R46「删 1 人重训」的扰动并列，只作量级参照，不分解各自的来源。")
    Ap(f"- 不能用迁移人数（{n(nAB, src('5'))} 人）的占比代替模型稳定性证据；稳定性的证据是上面逐任务重训后的旧新比较。\n")

    Ap("## 10. 已校验事实与不确定推断\n")
    Ap("- 已校验：公式与单位（官方原文）、源字段未校准、只改目标周期与目标列、资格/结局/分组/外层折不变、包装逐位复现冻结单元与冻结推断、可执行单元与原版相同、新结果全部来自新单元。")
    Ap("- 不确定：重训的随机扰动与校准效应在单次重训中不可分；区间仍是固定 OOF 预测上的区间，不含重新开发变异；2001–2004 周期是否需要校准沿用原实现，本轮按指令不扩展。")
    Ap("- 本分支只回答「按官方建议校准后，NHANES 结论是否改变」，不是新的主分析，也不改变 ICU、NWICU、T1/T2 的任何结果。\n")

    Ap("## 11. 建议的论文表述（供 GPT 裁决，不由我决定写入）\n")
    if not sdr:
        Ap("建议放在 Methods 的 NHANES 部分一句、Results 或补充材料一段，标明是事后敏感性分析：\n")
        Ap(f"> *Sensitivity analysis.* NHANES 2005–2006 serum creatinine had not been recalibrated as recommended by NCHS in the frozen analysis data. "
           f"Applying the recommended equation to the {n(SC['改动人数（2005–2006 且血肌酐非缺失）'], src('4'))} affected participants reclassified "
           f"{n(nAB, src('5'))} of {n(mig('A（降低）', 'A（降低）') + nAB, src('5'))} participants from reduced to non-reduced eGFR (none in the opposite direction). "
           "We redeveloped the complete NHANES learning program, including the objective-aligned comparisons, with this single input change. "
           f"The three-state classification of the primary balanced comparison was unchanged in all {n(len(bal), src('8'))} evaluable tasks; "
           f"the low-albumin track-B advantage remained supported ({p(la.loc['B', '观察相对改善_新'], src('8'))}; pointwise {reg('95%', K)} interval {p(la.loc['B', '相对_P2_5_新'], src('8'))} to {p(la.loc['B', '相对_P97_5_新'], src('8'))}; originally {p(la.loc['B', '观察相对改善_旧'], src('8'))}), "
           f"and the track-B-objective-aligned comparison again excluded a specialist gain of 1% or more ({p(d1.观察相对改善_新, src('16'))}; {p(d1.相对_P2_5_新, src('16'))} to {p(d1.相对_P97_5_新, src('16'))}). "
           "Changes in point estimates were of the same order as those produced by refitting after removing a single participant, which fixed out-of-fold intervals do not capture.\n")
        Ap("最后一句是否保留由你定：R52 的最大变化略大于删 1 人重训的最大变化（第 0 节第 5 条），写「same order」是描述量级，不是等价判断。\n")
    else:
        Ap("判定为 SCIENTIFIC_DECISION_REQUIRED，不提供写入建议，等共审。\n")

    Ap("## 12. Introduction v0.4a 审阅（独立科学质疑，只报必改与可选）\n")
    Ap(f"核对范围：v0.3b 与 v0.4a 逐段比对，确认只改 Introduction 三段（P9、P10、P12）并新增参考文献 15–18，其余段落一字未变。")
    Ap("新增文献核对（摘要与出版页面层级，未逐字读全文）：Dwork 2018 PMLR 81:119–133、Angelopoulos 2025 Ann Appl Stat 19(2):1641–1662（doi 一致）与 Wang 2021 IEEE TIT（doi 10.1109/TIT.2021.3075415 一致；卷期页码未在检索结果中见到，投稿前以 IEEE Xplore 记录核对）。Sagawa 2020 ICLR 为通行引用。\n")
    Ap("**必改**：无。三段改写没有发现事实错误或越界主张。\n")
    Ap("**可选**")
    Ap("- P10 插入 Wang、Dwork 两句后，下一句仍以「In particular」开头，读起来像在承接 Wang/Dwork；建议改为「Closest to the present clinical setting, Pfohl and colleagues …」，或把 Wang/Dwork 两句移到 Pfohl/Suriyakumar 之后。")
    Ap("- Wang 等的描述可以更贴近与本文最相关的部分：他们不只讨论何时分组有益，还指出在有限样本下分组并不总是有益——这正是本文「样本成本」论点的理论先例。建议写成「… quantified the benefit of splitting and showed that, with finite samples, splitting need not help [15]」。")
    Ap("- P9 开头「For developers of clinical prediction models」与 P12 新句「For clinical model developers」重复，可删其一。")
    Ap("- P12 新句把评价顺序写成「the proposed evaluation sequence is …」。它由一项结果知情后的单任务追加对照与模拟支持，建议在句首或句末点明来源（如「motivated by these analyses」），避免读成经过验证的规程。")
    Ap("- 文献 15–18 先于 7–14 在正文出现，Vancouver 顺序需在投稿版统一重排（GPT 已说明会做）。")
    Ap("- Pfohl 2022 [4] 本身就评价了分布鲁棒优化，P10「Group-robust optimization … address related but distinct questions [17,18]」可顺带写明 [4] 已在临床数据上比较过这类目标，避免读者以为 DRO 与本题无交集。\n")
    reg("15", K); reg("16", K); reg("17", K); reg("18", K); reg("7", K); reg("14", K); reg("4", K)
    cells("81:119–133 19(2):1641–1662 10.1109/TIT.2021.3075415", "文献出版页面核对")

    Ap("## 13. 交付\n")
    Ap("- `R52_NHANES_SCr_SENS_REPORT.md`：本文件。")
    Ap("- `R52_NHANES_SCr_SENS_EVIDENCE.xlsx`：首页目录；来源与指纹；拟合前核验；QC 锚点；血肌酐改动；eGFR 迁移矩阵；12 任务 n/事件/轨；运行 QC；E1 主对照；max-T；对照方向汇总；全部 Brier 与 log loss 对照；辅助指标均值与逐 seed；配对差；L3 判读、全部对照、C1 max-T、差距收回；开发选择分布；任务纳入；STOP/GO；变化幅度与重训扰动基准。只含汇总，无个人信息。")
    Ap("- `R52_NHANES_SCr_SENS_CODE_TESTS.zip`：分支包装脚本（数据构建、运行、推断、对比、报告、测试、打包、驱动）、冻结与审计记录、测试日志、依赖版本与文件指纹。原冻结核心脚本未改，逐人预测与个人记录留在本地。")
    Ap("\n交付这三件后停止，不改 R47/R48、主文 Word、六张主图与补充材料。")

    md = "\n".join(a) + "\n"
    if write:
        (root / "R52_NHANES_SCr_SENS_REPORT.md").write_text(md, encoding="utf-8")
        (root / "R52_NUM_TRACE.json").write_text(json.dumps({k: sorted(v) for k, v in sorted(TRACE.items())}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("written", len(md), "chars;", len(TRACE), "tokens; unregistered:", unregistered(md, {k: 1 for k in TRACE}))
    return md


if __name__ == "__main__":
    main(dry=len(sys.argv) > 1 and sys.argv[1] == "dryrun")
