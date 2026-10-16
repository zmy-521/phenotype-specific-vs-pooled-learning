# -*- coding: utf-8 -*-
r"""R51：由 R51_Table1_Baseline_Source.xlsx、R51_BASELINE.json 与 R47/R48 冻结工作簿生成 R51_Baseline_QC.md。
正文里每个数字都经 n()/p()/cells() 从结果表取值并登记到 R51_NUM_TRACE.json（token → 来源）；pgss_r51_tests.py 反查 md 中全部数字。
定义常量（24 h、72 h、10 年等）以「定义常量」登记；审阅条目数由条目列表长度算出。
"""
import sys, json, re, pathlib
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
OUT = ROOT / "34_R51_基线只读提数"
X47 = ROOT / "29_R47_写作交接" / "R47_冻结数字包.xlsx"
X48 = ROOT / "30_R48_图稿与补充材料" / "R48_图源与补充材料.xlsx"
TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_.])[−-]?\d[\d,]*(?:\.\d+)?%?")
TRACE = {}


def norm(t):
    return t.replace("−", "-")


def reg(tok, src):
    for t in TOKEN_RE.findall(tok) or [tok]:      # 「+3.23%」在正文里被切成「3.23%」，按同一切法登记
        TRACE.setdefault(norm(t), set()).add(src)
    return tok


def n(v, src, dp=0):
    v = float(v)
    return reg(f"{int(round(v)):,}" if dp == 0 else f"{v:,.{dp}f}", src)


def p(v, src, dp=1, sign=False):
    s = (f"{100 * float(v):+.{dp}f}%" if sign else f"{100 * float(v):.{dp}f}%").replace("-", "−")
    return reg(s, src)


def cells(df_or_text, src):
    vals = [df_or_text] if isinstance(df_or_text, str) else df_or_text.astype(str).to_numpy().ravel()
    for v in vals:
        for t in TOKEN_RE.findall(v):
            reg(t, src)
    return df_or_text


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(out)


def main(write=True):
    TRACE.clear()
    W = pd.read_excel(OUT / "R51_Table1_Baseline_Source.xlsx", sheet_name=None)
    T0, L, T2, T3, T4, T5, T6, V, V2, Q = (W[k] for k in ("T0_Table1_排版", "T1_基线长表", "T2_单位患者与结局对账", "T3_表型可判定覆盖率", "T4_单位核实",
                                                            "T5_补充图S1核对", "T6_NHANES血肌酐校准影响", "A1_版本与指纹", "A1b_NHANES重建闸门", "A2_质控"))
    B3 = pd.read_excel(X47, sheet_name="B3_分层计数"); B4 = pd.read_excel(X47, sheet_name="B4_maxT汇总"); B5 = pd.read_excel(X47, sheet_name="B5_NHANES低白蛋白归因")
    D1 = pd.read_excel(X47, sheet_name="D1_主张证据限制矩阵")
    S54 = pd.read_excel(X48, sheet_name="S54_D1_认证汇总"); S58 = pd.read_excel(X48, sheet_name="S58_C2_认证分解汇总")
    t6 = dict(zip(T6.项, T6.值))
    R = {(r.数据库, r.项): int(r.冻结数据计数) for r in T2.itertuples()}
    S = lambda sh: f"R51_Table1_Baseline_Source.xlsx:{sh}"
    s47, s48, K = "R47_冻结数字包.xlsx", "R48_图源与补充材料.xlsx", "定义常量"
    cells(T4, S("T4"))      # 单位脚注引用的截顶、范围等定义值

    b3 = lambda lib, sc, layer="L1": B3[(B3.库 == lib) & (B3.尺度 == sc) & B3.层.str.startswith(layer) & (B3.对照 == "E1")].iloc[0]
    b4 = lambda lib: B4[B4.库 == lib].iloc[0]
    e1B = B5[(B5.对照 == "E1") & (B5.尺度 == "B")].iloc[0]; d1B = B5[(B5.对照 == "D1") & (B5.尺度 == "B")].iloc[0]
    nh_bal, nh_B, nh_pop = b3("nhanes", "bal", "L2"), b3("nhanes", "B", "L2"), b3("nhanes", "pop", "L2")
    cert_on, cert_n = int(S54.专业化.sum()), int(S54.单元.sum())
    fct_bal = float(S58[S58.尺度 == "bal"].iloc[0].合计相对中位)
    k1lim = D1[D1.编号 == "K1"].iloc[0].限制; k13 = D1[D1.编号 == "K13"].iloc[0].关键数字
    cov = lambda db, oc, ph: T3[(T3.数据库 == db) & (T3.结局 == oc) & (T3.表型 == ph)].iloc[0]
    obs = lambda db, var: L[(L.数据库 == db) & (L.变量 == var)].iloc[0]
    ex = lambda db, step: T5[(T5.面板 == db) & (T5.步骤.astype(str) == step)].iloc[0].来源值
    q_ok, q_n, mut_ok = int(Q.通过.sum()), len(Q), int(Q.变异对照检出.sum())
    v_ok, v2_ok = int((V.实测 == V.登记).sum()), int((V2.实测 == V2.登记).sum())
    t2_ok, t2_n = int(T2.一致.dropna().astype(bool).sum()), int(T2.一致.notna().sum())
    t5_ok, t3_ok = int(T5.一致.sum()), int(T3.与R47_B1一致.sum())
    n_icu24 = int(T3.数据库.isin(["MIMIC-IV", "eICU-CRD"]).sum())
    n_scan = int(T4.NWICU依据.str.contains("全表实扫").sum())
    unk = R[("NWICU", "单位")] - R[("NWICU", "死亡分母")]
    ce = "本报告条目计数"

    # ---- 审阅条目（先列，再计数）----
    S1_must = [
        "**S1-a 末框没有写排除原因。** 三个 ICU 面板的末框只写「Eligible ICU stays」，读者看不出最后排除的 "
        f"{n(ex('MIMIC-IV', '排除 3→4'), S('T5'))}（MIMIC-IV）、{n(ex('eICU-CRD', '排除 3→4'), S('T5'))}（eICU-CRD）、{n(ex('NWICU', '排除 5→6'), S('T5'))}（NWICU）是因为什么。"
        "来源里的原因：MIMIC-IV「landmark 前无已知院内死亡、死亡时点可靠、无死亡—出院时间冲突」；eICU-CRD「landmark 前无已知院内死亡、出院状态非缺失」；NWICU「结局来源可靠」。"
        "流程图每个排除都应写原因，建议写进末框或排除标注。",
        "**S1-b 图注草稿对中间步骤的核实说得过头。** 草稿写「Counts for the intermediate ICU eligibility stages originate from frozen cohort-construction logs, which matched the final eligible flags」。"
        "实际只有第 1 步和最终人数能从冻结数据重算并与运行记录一致；中间两步（仍在 ICU、成年）只有运行记录（R49 C1「仅 R35 运行记录，无法重算」）。建议改为："
        "*Counts at the intermediate MIMIC-IV and eICU-CRD stages are taken from the frozen cohort-construction log; the first and final counts in that log were reproduced exactly from the frozen data, but the intermediate stages could not be recomputed.*",
        "**S1-c 图里有内部版本标签。** 面板 C 的「Hospital death (R46 v2)」和底部「Data sources: R48 S2/S9/S10, R49 C1/C4, R50 NWICU direct-count report」对读者没有意义。"
        "投稿版应删去，溯源放在源数据文件里；「R46 v2」可改为「corrected outcome status」。"]
    S1_opt = [
        "S1-d MIMIC-IV 第 3 步写「Aged ≥18 years」，来源是「Aged ≥18 years with known age」，可与 eICU-CRD 面板统一。",
        "S1-e 面板 C、D 的框挨得太紧，箭头压在框线上（PNG 上可见）；建议拉开间距或减小框高。",
        f"S1-f NHANES 心血管死亡只写了分母，可补事件数 {n(R[('NHANES', 'CVD事件')], S('T2'))}/{n(R[('NHANES', 'CVD分母')], S('T2'))}，与全因死亡格式一致。",
        "S1-g 图题「Cohort assembly and outcome coverage」与图注题「Cohort assembly and outcome eligibility by data source」不一致；图注里的数字缺千分位（"
        f"{reg(str(R[('NWICU', '死亡分母')]), S('T2'))}、{reg(str(R[('NWICU', '死亡患者')]), S('T2'))} 等）。",
        "S1-h 起点不同：NWICU 从全部 ICU 记录起，MIMIC-IV/eICU-CRD 从「每次住院的首段 ICU 住院」起（更早的总数没有保存），可在图注里说一句。"]
    E_must = [
        f"**E-1 L2 行只报了两轨等权尺度，漏了唯一的支持结果。** NHANES 在 B 轨尺度上有 {n(nh_B.支持, s47 + ':B3')} 个任务被支持（低白蛋白；B3 L2 B 行："
        f"排除 {n(nh_B.排除, s47 + ':B3')}、不足 {n(nh_B.证据不足, s47 + ':B3')}、支持 {n(nh_B.支持, s47 + ':B3')}），人口加权尺度也有 {n(nh_pop.支持, s47 + ':B3')} 个；"
        f"逐轨 max-T 为排除 {n(b4('nhanes').排除, s47 + ':B4')}、不足 {n(b4('nhanes').证据不足, s47 + ':B4')}、支持 {n(b4('nhanes').支持, s47 + ':B4')}。"
        f"读者在 L3 行才第一次看到「B 轨原 {p(e1B.观察相对改善, s47 + ':B5', 2, True)}」，却不知道它是 L2 里唯一被支持的结果。建议 L2 行写成："
        f"*balanced scale {n(nh_bal.排除, s47 + ':B3')} excluded / {n(nh_bal.证据不足, s47 + ':B3')} insufficient / {n(nh_bal.支持, s47 + ':B3')} supported; "
        "one track-specific supported result (low albumin, track B), examined in L3.*",
        f"**E-2 NWICU 行要写明住院死亡结局是在看到首轮结果后修正的。** R47 D1 的 K1 限制原文：「{k1lim.split('；')[-1]}」。三角表把 NWICU 作为独立数据源上的复现来呈现，"
        f"这个事后修正应在同一行可见：*in-hospital death status corrected after the first run ({n(unk, S('T2'))} stays with unresolvable discharge status excluded); "
        "total ICU stay >72 h unaffected.*"]
    E_opt = [
        f"E-3 表注写明两类区间不同：真实数据是固定 OOF 预测上的区间（max-T 为逐轨同时区间），不含重新开发的变异；T2 是独立重复的 Monte Carlo 区间。否则读者会把 L1 与 T2 的区间当成同类。",
        f"E-4 加一行真实数据 FCT：{n(cert_on, s48 + ':S54')}/{n(cert_n, s48 + ':S54')} 次可评价轨认证启用了专属模型；在共同可评价记录上，相对 P*_D 的两轨等权相对改善中位 "
        f"{p(fct_bal, s48 + ':S58', 2, True)}（负号表示更差）。这是评价顺序第二、三层（能否选对、部署风险）唯一的真实数据证据，现在三角表里这两层只有 T2。"
        f"须注明这 {n(cert_n, s48 + ':S54')} 个决策互相重叠，不是独立试验，也不是错误启用率。",
        "E-5 三个 seed：若 Results 提到「三个 seed 一致」，在 L1 行或表注写明这是同一批患者上的重复开发（检验算法随机性），不是复现。",
        f"E-6 「{n(b4('mimic').排除 + b4('eicu').排除, s47 + ':B4')}/{n(b4('mimic').项数 + b4('eicu').项数, s47 + ':B4')}」由两个独立的 max-T 族合成"
        f"（MIMIC-IV、eICU-CRD 各 {n(b4('mimic').项数, s47 + ':B4')} 轨，各有临界值），建议写成「{n(b4('mimic').排除, s47 + ':B4')}/{n(b4('mimic').项数, s47 + ':B4')} in each database」，避免读成一个族。",
        f"E-7 T2 行补上设计限定（R47 D1 K13：「{cells(k13, s47 + ':D1')}」），即只对所设条件成立。"]

    a = []; A = a.append
    A("# R51 Claude 回复：四库基线只读提数、补充图 S1 核对、证据三角表审阅\n")
    A("只读：没有拟合、没有重训或再校准、没有改动任何冻结目录。数字均由 `pgss_r51_report.py` 从结果表生成，逐个登记在 `R51_NUM_TRACE.json`。\n")
    A("## 0. 结论\n")
    A(f"1. **七项任务全部完成，没有触发 STOP。** 冻结文件指纹 {n(v_ok, S('A1'))}/{n(len(V), S('A1'))} 与登记一致（含 R46 v2 两份结局状态文件）。"
      f"质控 {n(q_ok, S('A2'))}/{n(q_n, S('A2'))} 通过；每条都配了一个必须失败的变异对照，{n(mut_ok, S('A2'))}/{n(q_n, S('A2'))} 检出。")
    A(f"2. **任务书 §3 的队列表逐值一致**：{n(t2_ok, S('T2'))}/{n(t2_n, S('T2'))} 项。另补一个表里没有的数：NHANES 10 年心血管死亡事件 "
      f"{n(R[('NHANES', 'CVD事件')], S('T2'))}/{n(R[('NHANES', 'CVD分母')], S('T2'))}（R46 v2 死因修正后）。")
    A("3. **Table 1 四库变量齐全。** NHANES 的白蛋白、BMI、eGFR 不在冻结分析文件里（只存了表型标志）。我按 R44 队列脚本的同一路径在内存里重建，"
      f"闸门是重建的 X/P/OK/meta 四个指纹与登记逐位相同（{n(v2_ok, S('A1b'))}/{n(len(V2), S('A1b'))} 通过），且由重建值重算的三个表型与冻结 P/OK 逐个一致。只用于描述。")
    A(f"4. **补充图 S1 的数字全部可溯源**：{n(t5_ok, S('T5'))}/{n(len(T5), S('T5'))} 个数字与来源一致。问题在文字标注，不在数字（第 5 节：必改 {n(len(S1_must), ce)} 项）。")
    A("5. **新发现一处需要披露的问题。** NHANES 2005–2006 周期的血肌酐没有按官方 codebook 的建议校准（冻结数据只校准了 1999–2000）。"
      f"若校准，eGFR 降低表型有 {n(t6['由「降低」改判为「不降低」n'], S('T6'))}/{n(t6['全队列 eGFR 降低（冻结）n'], S('T6'))} 人"
      f"（{p(t6['改判占冻结 eGFR 降低（A 轨）的比例'], S('T6'))}）改判为不降低，反方向 {n(t6['由「不降低」改判为「降低」n'], S('T6'))} 人。"
      "冻结结果不变。我建议只披露、不重跑（第 7 节 D1，需要你裁决）。")
    A(f"6. **证据三角表**：引用的数字全部与 R47/R48 冻结表一致；必改 {n(len(E_must), ce)} 项，可选 {n(len(E_opt), ce)} 项（第 6 节）。\n")

    A("## 1. 数据版本与 STOP 规则\n")
    vv = V.copy(); vv["一致"] = np.where(vv.实测 == vv.登记, "是", "否")
    A(md_table(cells(vv[["数据库", "文件", "实测", "登记", "一致"]], S("A1")))); A("")
    A("指纹口径：MIMIC-IV/eICU-CRD 用文件 SHA-256 前 16 位对 `DATA_VERSION.json`；NWICU/NHANES 用内容指纹（`hash_pandas_object`）对各自的 `COHORT.json`；"
      "两份 R46 v2 结局状态对 `R46_OUTCOME_VERIFY.json` 的「状态SHA」（R46 重跑读取的同一登记）。没有版本冲突、结局不明、单位不可确认或患者去重冲突，四库都没有暂停。\n")
    reg("256", K); reg("16", K)

    A("## 2. 单位、患者与结局分母对账（任务 1）\n")
    t2 = T2.copy()
    t2["一致"] = np.where(t2.一致.isna(), "（表中无此项）", np.where(t2.一致.astype(float) == 1, "是", "否"))   # Excel 读回为 1.0/NaN
    t2["GPT任务书表值"] = t2.GPT任务书表值.astype("Int64").astype(str).replace("<NA>", "—")
    A(md_table(cells(t2, S("T2")))); A("")
    A("计数口径：ICU 三库按合格住院计（每次住院的首段 ICU 住院），患者数是 `group_id` 去重；NWICU 住院死亡用 R46 v2 可评价标志；NHANES 心血管死亡用 R46 v2 死因修正后的可评价标志。\n")

    A("## 3. Table 1（任务 2、3）\n")
    A("每列一个数据源；连续变量为 median [Q1, Q3]，后附实际观测数与观测比例；不插补、不换算单位、不做来源间 P 值比较。完整长表（含 P1/P99、缺失 n 与比例）在 xlsx 的 T1。\n")
    A(md_table(cells(T0.drop(columns=["区块"]).rename(columns={"行": "Characteristic"}), S("T0")))); A("")
    A("脚注（供 GPT 写正式表注）：")
    A(f"- 时点：ICU 三库为每次合格 ICU 住院在入 ICU 后 {reg('24', K)} h landmark 前可得的最后一个值（data v5 窗口规则）；NHANES 为体检时测量。一人多次住院按住院计，独立患者数另报。")
    A(f"- 单位核实（xlsx T4）：NWICU 对 labevents 的 valueuom 做了全表实扫，{n(n_scan, S('T4'))} 项化验只出现同义单位写法（如血小板 K/uL、10^3/µL、k/cu mm；氯 mEq/L 与 mmol/L）；"
      "NHANES 逐周期核对了官方变量标签与取值范围；MIMIC-IV/eICU-CRD 依据 R34 变量映射核验表（两库同单位）以及抽取时的生理范围（SI 量级的值落在范围外），本轮没有连库重扫单位字段。"
      "NHANES 1999–2002 的血小板标签写作「(%) SI」，取值范围与其余周期的 1000 cells/µL 相同，按标签笔误处理。没有任何一项标为 not comparable。")
    A("- 年龄截顶：eICU-CRD「> 89」记为 90；NHANES 1999–2006 截顶 85、2007–2008 截顶 80（本地 NHANES 变量索引）。")
    A("- BMI：ICU 三库限定 10–80 kg/m²（冻结 `phenotypes()`）；NHANES 未设此限制。")
    A("- 医院数：只有 eICU-CRD 的冻结文件带医院标识。MIMIC-IV 与 NWICU 写 NA；正文若写单中心，须引用数据源文献。")
    A(f"- 低覆盖变量：白蛋白在 MIMIC-IV 只有 {p(1 - obs('MIMIC-IV', 'Albumin').缺失比例, S('T1'))} 的住院可得，BMI 在 NWICU 只有 {p(1 - obs('NWICU', 'Body mass index').缺失比例, S('T1'))}。"
      "两者也是低白蛋白、肥胖两个表型的定义输入，所以这两个表型任务只在可判定的住院里比较（第 4 节）。")
    A("- NHANES 血肌酐：1999–2000 已按 Selvin 2007 校准；2005–2006 未做官方推荐的校准（第 7 节 D1）。Table 1 报告的是冻结分析值。\n")

    A("## 4. 六表型可判定覆盖率（任务 4）\n")
    A(f"{n(len(T3), S('T3'))} 个结局×表型组合的可判定 n、不可判定 n 与比例、A/B 轨 n 与事件数全部在 xlsx T3；{n(t3_ok, S('T3'))}/{n(len(T3), S('T3'))} 与 R47 B1 逐值相同，"
      f"MIMIC-IV/eICU-CRD 的 {n(n_icu24, S('T3'))} 行另与 R49 C2 的可判定 n、不可判定 n 逐值相同。不可判定者不计为阴性。下表只列可判定比例：\n")
    rows = []
    for db, oc in (("MIMIC-IV", "OUT_LOS72"), ("eICU-CRD", "OUT_LOS72"), ("NWICU", "OUT_LOS72"), ("NWICU", "OUT_HOSP_MORT"), ("NHANES", "O_MORT10"), ("NHANES", "O_CVD10")):
        r = {"数据源": db, "结局（分母）": f"{oc}（{n(cov(db, oc, '女性').结局可评价分母, S('T3'))}）"}
        for ph in ("女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"):
            r[ph] = p(1 - cov(db, oc, ph).不可判定比例, S("T3"))
        rows.append(r)
    A(md_table(pd.DataFrame(rows))); A("")
    A("MIMIC-IV、eICU-CRD 两个结局的分母相同，覆盖率也相同，表中只列一行。")
    lowA = cov("NHANES", "O_CVD10", "低白蛋白")
    A(f"- 覆盖率最低：MIMIC-IV 低白蛋白 {p(1 - cov('MIMIC-IV', 'OUT_LOS72', '低白蛋白').不可判定比例, S('T3'))}、NWICU 肥胖 {p(1 - cov('NWICU', 'OUT_LOS72', '肥胖').不可判定比例, S('T3'))}、"
      f"MIMIC-IV 肥胖 {p(1 - cov('MIMIC-IV', 'OUT_LOS72', '肥胖').不可判定比例, S('T3'))}。能否判定不是随机的，本轮没有评估其方向。")
    A(f"- NHANES 低白蛋白 A 轨很小：{n(lowA.A轨n, S('T3'))} 人，心血管死亡事件 {n(lowA.A轨事件, S('T3'))} 个；O_CVD10×低白蛋白按 R47 B1 列为不进入分析（外层折内早停留出单类别，不可执行）。")
    A(f"- 女性表型在 eICU-CRD（{n(obs('eICU-CRD', 'Female').缺失n, S('T1'))} 人）和 NWICU（{n(obs('NWICU', 'Female').缺失n, S('T1'))} 人）有性别未知者，"
      "上表四舍五入后仍显示为 100.0%。\n")

    A("## 5. 补充图 S1 核对（任务 5）\n")
    A(f"数字：{n(t5_ok, S('T5'))}/{n(len(T5), S('T5'))} 一致。核对内容：每个面板的步数、每一步人数、每个排除数（用来源的相邻两步相减，不用图中数字自减）、"
      "最终人数与冻结文件直接计数、结局框里的全部患者数、分母与事件数。MIMIC-IV/eICU-CRD 对 R49 C1，NWICU 对 R48 S2（即 R45 E1），NHANES 对 R44 `COHORT.json` 与 R48 S9。\n")
    A("**必改**\n"); [A(f"- {x}") for x in S1_must]; A("")
    A("**可选**\n"); [A(f"- {x}") for x in S1_opt]; A("")

    A("## 6. 证据三角表审阅（任务 7）\n")
    A("数字核对：任务书第 4 节表中的数字全部与冻结表一致——"
      f"MIMIC-IV/eICU-CRD 两轨等权 E1 中位 {p(b3('mimic', 'bal').相对改善中位, s47 + ':B3', 2, True)}/{p(b3('eicu', 'bal').相对改善中位, s47 + ':B3', 2, True)}，"
      f"max-T 排除 {n(b4('mimic').排除, s47 + ':B4')}+{n(b4('eicu').排除, s47 + ':B4')}；"
      f"NWICU 中位 {p(b3('nwicu', 'bal').相对改善中位, s47 + ':B3', 2, True)}，max-T {n(b4('nwicu').排除, s47 + ':B4')}/{n(b4('nwicu').项数, s47 + ':B4')} 排除、{n(b4('nwicu').证据不足, s47 + ':B4')} 项不足；"
      f"NHANES 两轨等权 {n(nh_bal.排除, s47 + ':B3')}/{n(nh_bal.证据不足, s47 + ':B3')}/{n(nh_bal.支持, s47 + ':B3')}（排除/不足/支持）；"
      f"L3 低白蛋白 B 轨 {p(e1B.观察相对改善, s47 + ':B5', 2, True)} → {p(d1B.观察相对改善, s47 + ':B5', 2, True)}"
      f"（{p(d1B.相对_P2_5, s47 + ':B5', 2, True)} 至 {p(d1B.相对_P97_5, s47 + ':B5', 2, True)}）。层级区分（重复开发、模拟、追加对照不当作独立验证）总体正确。\n")
    A("**必改**\n"); [A(f"- {x}") for x in E_must]; A("")
    A("**可选**\n"); [A(f"- {x}") for x in E_opt]; A("")

    A("## 7. 需要 GPT 裁决的事项\n")
    A("- **D1 NHANES 2005–2006 血肌酐校准（建议：只披露、不重跑）。** 依据是本地存档的官方 codebook `BIOPRO_D.htm` 原文："
      "「Correction of serum creatinine values in NHANES 2005-2006 is highly recommended」，公式为 Standard creatinine = -0.016 + 0.978 × 未校准值（mg/dL）。"
      f"冻结数据只按 Selvin 2007 校准了 1999–2000。合格队列中 2005–2006 周期 {n(t6['合格队列中 2005–2006 周期人数'], S('T6'))} 人，其中血肌酐可得 {n(t6['其中血肌酐可得'], S('T6'))} 人；"
      f"该周期血肌酐中位数 {n(t6['2005–2006 血肌酐中位数（冻结，未校准）mg/dL'], S('T6'), 2)} → {n(t6['2005–2006 血肌酐中位数（按 BIOPRO_D 公式校准后）mg/dL'], S('T6'), 2)} mg/dL。"
      f"影响有两处：一是两个 eGFR 降低任务（O_MORT10、O_CVD10）的分轨，{n(t6['由「降低」改判为「不降低」n'], S('T6'))}/{n(t6['全队列 eGFR 降低（冻结）n'], S('T6'))} 人会从 A 轨移到 B 轨；"
      "二是全部 NHANES 任务都用到的血肌酐特征（只涉及 2005–2006 周期的值）。冻结结果不应为此重跑，本轮也没有估计它对性能比较的影响。建议 Methods 与 Limitations 各写一句：\n")
    cells(T6, S("T6"))
    A("  > In NHANES, serum creatinine from 1999–2000 was calibrated with the published equation; the recommended recalibration of 2005–2006 values was not applied in the frozen analysis data. "
      f"Applying it would lower 2005–2006 creatinine values (median {n(t6['2005–2006 血肌酐中位数（冻结，未校准）mg/dL'], S('T6'), 2)} to {n(t6['2005–2006 血肌酐中位数（按 BIOPRO_D 公式校准后）mg/dL'], S('T6'), 2)} mg/dL) "
      f"and reclassify {n(t6['由「降低」改判为「不降低」n'], S('T6'))} of {n(t6['全队列 eGFR 降低（冻结）n'], S('T6'))} participants ({p(t6['改判占冻结 eGFR 降低（A 轨）的比例'], S('T6'))}) "
      "from reduced to non-reduced eGFR, with none in the opposite direction; the frozen analyses were not re-run.\n")
    A("  另：2007–2008 的 codebook（`BIOPRO_E.htm`）写明方法已 IDMS 标准化，不需校准；2001–2004 不校准沿用 `nhanes.py` 的既有做法（依据 Selvin 2007），本轮没有回原文复核这一点。")
    A("- **D2 NHANES 白蛋白、BMI、eGFR 用内存重建值进 Table 1**：闸门已过（第 0 节第 3 条）。若你认为 Table 1 只能用冻结分析文件里的列，这三格改为 NA 加脚注即可，脚本里有现成的回退分支。")
    A("- **D3 MIMIC-IV、NWICU 的医院数**：冻结文件没有医院标识。写「single centre」须引数据源文献，否则保持 NA。\n")

    A("## 8. 交付与复现\n")
    A("- `R51_Table1_Baseline_Source.xlsx`：0_说明；T0 排版用 Table 1；T1 长表；T2 对账；T3 覆盖率；T4 单位核实；T5 图 S1 核对；T6 NHANES 血肌酐校准影响；A1/A1b 版本与重建闸门；A2 质控。只含汇总统计。")
    A("- `R51_Baseline_QC.md`：本文件。")
    A("- `R51_CODE.zip`：`R51_READONLY_BASELINE.py`（即 `pgss_r51_baseline.py`，只读冻结文件，不写任何冻结目录）、`pgss_r51_report.py`、`pgss_r51_tests.py`、测试日志、`R51_NUM_TRACE.json`、`R51_BASELINE.json`。"
      "运行需要本地冻结数据，患者级数据不随包上传。")
    A("- 质控明细（xlsx A2）：\n")
    q = Q.copy(); q["通过"] = q.通过.map({True: "是", False: "否"}); q["变异对照检出"] = q.变异对照检出.map({True: "是", False: "否"})
    A(md_table(cells(q, S("A2")))); A("")
    A("这一批做完即停，等你裁决 D1–D3 和第 5、6 节的修改。")

    md = "\n".join(a) + "\n"
    if write:
        (OUT / "R51_Baseline_QC.md").write_text(md, encoding="utf-8")
        (OUT / "R51_NUM_TRACE.json").write_text(json.dumps({k: sorted(v) for k, v in sorted(TRACE.items())}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("written", len(md), "chars;", len(TRACE), "registered tokens")
    return md


if __name__ == "__main__":
    main()
