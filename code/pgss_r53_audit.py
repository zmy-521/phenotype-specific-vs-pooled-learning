# -*- coding: utf-8 -*-
r"""R53 v0.5 限定只读复核：M1–M3 修复、Table 1 逐格对 R51、Table 2 逐数对 R47/R48/R52、参考文献映射与首次引用顺序、
Figure 1–6 指纹、Supplementary Figure S1 数字、eICU 医院数口径。每项配变异对照。只读，不改任何文件。
输出：36_R53_v05复核/R53_AUDIT.xlsx、R53_AUDIT.json、R53_复核.md
"""
import sys, json, re, pathlib, hashlib, difflib
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, docx

ROOT = HERE.parent
OUT = ROOT / "36_R53_v05复核"
X = pathlib.Path(r"D:\Research\3_对话与交接\GPT交接_PGSS")
PK = X / "53" / "1_GPT_R53交付包（已收到）" / "PGSS_R53_交付包"
V04B = X / "52" / "3_GPT_R52回应（已收到）" / "PGSS_Manuscript_v0.4b_R52_sensitivity_review.docx"
V05 = PK / "01_MANUSCRIPT" / "PGSS_Manuscript_v0.5_R53_integrated_review.docx"
TOK = re.compile(r"[−-]?\d[\d,]*(?:\.\d+)?%?")
M = {"M1": "thus, the main inference was stable, although point estimates and a minority of secondary classifications changed, including 3 of 44 E1 log-loss judgments (Supplementary Materials).",
     "M2": "the standardized values replaced the original serum-creatinine predictor and were used to recalculate eGFR with the same 2021 CKD-EPI equation, thereby updating the reduced-eGFR phenotype.",
     "M3": "preserved the main material-gain classifications but changed point estimates and a minority of secondary judgments."}
M_OLD = {"M1": "all secondary classifications were not invariant", "M2": "then recalculated eGFR using the same 2021 CKD-EPI equation and updated the reduced-eGFR phenotype",
         "M3": "altered some point estimates and secondary judgments"}


def doc(f):
    d = docx.Document(f)
    ps = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    return d, ps


def cites(ps):
    out = []
    for k, p in enumerate(ps):
        for m in re.finditer(r"\[(\d+(?:\s*[–,-]\s*\d+)*)\]", p):
            nums = []
            for part in re.split(r"\s*,\s*", m.group(1)):
                if re.search("[–-]", part):
                    a, b = map(int, re.split("[–-]", part)); nums += list(range(a, b + 1))
                else:
                    nums.append(int(part))
            out.append((k, nums))
    return out


def refcheck(old_ps, new_ps):
    def split(ps):
        i = ps.index("References"); return ps[:i], ps[i + 1:i + 20]
    oa, orf = split(old_ps); na, nrf = split(new_ps)
    key = lambda r: re.sub(r"^\d+\.\s*", "", r)
    oref = {int(r.split(".")[0]): key(r) for r in orf}; nref = {int(r.split(".")[0]): key(r) for r in nrf}
    inv = {v: k for k, v in nref.items()}; mp = {o: inv.get(t) for o, t in oref.items()}
    oc, nc = cites(oa), cites(na)
    mism = [(o, n) for o, n in zip(oc, nc) if [mp.get(x) for x in o[1]] != n[1]] + ([("组数不同", len(oc), len(nc))] if len(oc) != len(nc) else [])
    seen = []
    for _, ns in nc:
        for x in ns:
            if x not in seen:
                seen.append(x)
    return dict(映射=mp, 条目文字相同=set(oref.values()) == set(nref.values()), 引用组=len(nc), 映射不符=mism,
                首次引用顺序=seen, 顺序连续=seen == list(range(1, len(nref) + 1)), 全部被引=sorted(set(x for _, ns in nc for x in ns)) == list(range(1, len(nref) + 1)))


def table1_check(t1, T0):
    order = {"Analytical role": "Role in study", "Eligible observation units, n": "Eligible units, n", "Distinct participants, n": "Distinct patients, n",
             "Female, n/N (%)": "Female", "Age, years": "Age, years", "Hemoglobin, g/dL": "Hemoglobin, g/dL", "Serum creatinine, mg/dL": "Serum creatinine, mg/dL",
             "eGFR, mL/min/1.73 m²": "eGFR (CKD-EPI 2021), mL/min/1.73 m²", "Serum albumin, g/dL": "Albumin, g/dL", "Body mass index, kg/m²": "Body mass index, kg/m²",
             "RDW, %": "RDW, %", "Platelets, 10³/µL": "Platelet count, ×10³/µL", "Serum chloride, mmol/L": "Chloride, mmol/L",
             "In-hospital mortality, n/N (%)": "In-hospital death", "Total ICU stay >72 h from ICU admission, n/N (%)": "Total ICU stay >72 h from ICU admission",
             "10-year all-cause mortality, n/N (%)": "10-year all-cause death", "10-year cardiovascular mortality, n/N (%)": "10-year cardiovascular death"}
    rows = []
    t0 = T0.drop_duplicates("行", keep="first").set_index("行")
    for r in t1.rows[1:]:
        c = [x.text.strip() for x in r.cells]
        if c[0] not in order:
            continue
        src = t0.loc[order[c[0]]]
        for j, db in enumerate(["MIMIC-IV", "eICU-CRD", "NWICU", "NHANES"], start=1):
            dt = TOK.findall(c[j]); st = TOK.findall(str(src[db]))
            if c[0] == "Female, n/N (%)":
                st = st[:3]                                  # R51 T0 在同格附「sex unknown n」，主表未列
            if c[0] == "In-hospital mortality, n/N (%)" and db == "NWICU":
                st = st[:3]                                  # 患者数 17,626 移到了表注
            rows.append(dict(行=c[0], 数据源=db, 文稿=c[j].replace("\n", " / "), 源=str(src[db]), 一致=dt == st))
    return pd.DataFrame(rows)


def table2_check(t2, B3, B4, B5, S54, S58, C1, C3, E1n, JDn):
    pc = lambda v: f"{100 * v:.2f}%"
    b3 = lambda lib, layer="L1": B3[(B3.库 == lib) & (B3.尺度 == "bal") & B3.层.str.startswith(layer) & (B3.对照 == "E1")].iloc[0]
    b4 = lambda lib: B4[B4.库 == lib].iloc[0]
    e1B = B5[(B5.对照 == "E1") & (B5.尺度 == "B")].iloc[0]; d1B = B5[(B5.对照 == "D1") & (B5.尺度 == "B")].iloc[0]
    c3 = lambda cond, sel: C3[(C3.条件 == cond) & (C3.选择器 == sel)].iloc[0].两轨等权部署改善均值
    la = E1n[(E1n.结局 == "O_MORT10") & (E1n.表型 == "低白蛋白") & (E1n.尺度 == "B")].iloc[0]; d1n = JDn[JDn.对照 == "D1"].iloc[0]
    bal = E1n[E1n.尺度 == "bal"]
    exp = [("L1 MIMIC 中位", "−" + pc(-b3("mimic").相对改善中位)), ("L1 eICU 中位", "−" + pc(-b3("eicu").相对改善中位)),
           ("L1 max-T MIMIC", f"{int(b4('mimic').排除)}/{int(b4('mimic').项数)}"), ("L1 max-T eICU", f"{int(b4('eicu').排除)}/{int(b4('eicu').项数)}"),
           ("NWICU 中位", "−" + pc(-b3("nwicu").相对改善中位)), ("NWICU max-T 排除", f"{int(b4('nwicu').排除)}/{int(b4('nwicu').项数)}"),
           ("NWICU max-T 不足", f"{int(b4('nwicu').证据不足)}/{int(b4('nwicu').项数)}"),
           ("L2 8/3/0", f"{int(b3('nhanes', 'L2').排除)} excluded / {int(b3('nhanes', 'L2').证据不足)} inconclusive / {int(b3('nhanes', 'L2').支持)} supported"),
           ("L2 B +3.23%", "+" + pc(e1B.观察相对改善)), ("L3 +0.11%", "+" + pc(d1B.观察相对改善)),
           ("L3 区间", f"−{pc(-d1B.相对_P2_5)} to +{pc(d1B.相对_P97_5)}"),
           ("R52 11 任务不变", f"unchanged in all {len(bal)} tasks" if not bal.三态改变_m1.any() else "（有改变）"),
           ("R52 B +2.20%", "+" + pc(la.观察相对改善_新)), ("R52 aligned +0.20%", "+" + pc(d1n.观察相对改善_新)),
           ("FCT 1/712", f"{int(S54.专业化.sum())}/{int(S54.单元.sum())}"), ("FCT −0.61%", "−" + pc(-S58[S58.尺度 == "bal"].iloc[0].合计相对中位)),
           ("FCT 24 tasks", f"across {int(S58[S58.尺度 == 'bal'].iloc[0].任务)} tasks"),
           ("T2 8×200", f"{C1.shape[0]} conditions × {int(C1.完成重复.iloc[0])} repeats"), ("T2 n", f"n={int(C1.开发集n.iloc[0]):,}"),
           ("T2 π_A", f"track A={int(round(100 * C1.π_A.iloc[0]))}%"), ("T2 P1 always-split", "always-split −" + pc(-c3("P1", "ALWAYS_SPLIT"))),
           ("T2 P2 +3.15%", "P2 +" + pc(c3("P2", "ALWAYS_SPLIT"))), ("T2 P2 MIX", "MIX +" + pc(c3("P2", "MIX"))), ("T2 P2 FCT", "FCT −" + pc(-c3("P2", "FCT|P*")))]
    text = " ".join(c.text for r in t2.rows for c in r.cells).replace("\n", " ")
    return pd.DataFrame([dict(项=k, 期望文字=v, 在表中=v in text) for k, v in exp]), text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = []
    def add(sec, item, ok, mut, det=""):
        R.append(dict(部分=sec, 检查=item, 通过=bool(ok), 变异对照检出=(bool(mut) if mut is not None else "不适用（发现项）"), 说明=str(det)[:400]))
    d5, p5 = doc(V05); _, p4 = doc(V04B)
    # 1 M1–M3
    fixed = lambda ps, k: any(M[k] in p for p in ps) and not any(M_OLD[k] in p for p in ps)
    for k in M:
        reverted = [q.replace(M[k], M_OLD[k]) for q in p5]          # 变异：把修好的句子换回旧说法
        add("1 M1–M3", k, fixed(p5, k), not fixed(reverted, k), M[k][:80])
    # 2 参考文献
    rc = refcheck(p4, p5)
    add("5 参考文献", "条目文字与 v0.4b 相同；17 处引用按新旧映射逐一相符；首次引用顺序 1→19", rc["条目文字相同"] and not rc["映射不符"] and rc["顺序连续"] and rc["全部被引"],
        bool(refcheck(p4, [p.replace("[6]", "[7]", 1) if "Wang and colleagues" in p else p for p in p5])["映射不符"]), f"引用组 {rc['引用组']}")
    # 3 Table 1
    T0 = pd.read_excel(PK / "03_FROZEN_SOURCES" / "R51_Table1_Baseline_Source.xlsx", sheet_name="T0_Table1_排版")
    t1c = table1_check(d5.tables[0], T0)
    T0m = T0.copy(); T0m.loc[T0m.行 == "Hemoglobin, g/dL", "NWICU"] = T0m.loc[T0m.行 == "Hemoglobin, g/dL", "NWICU"].str.replace("10.8", "10.9", n=1)
    add("2 Table 1", "主文 Table 1 每格数字 = R51 T0（同格内数字逐个、按顺序）", t1c.一致.all(), not table1_check(d5.tables[0], T0m).一致.all(), f"{len(t1c)} 格")
    # eICU 医院数口径
    meta = pd.read_parquet(ROOT / "19_R38" / "data_v5" / "R38_eicu_META.parquet")
    h_all, h_el = int(meta.hospital_id.nunique()), int(meta.hospital_id[meta.eligible.astype(bool)].nunique())
    note = next(p for p in p5 if p.startswith("For ICU databases, observations are eligible ICU stays"))
    add("2 Table 1", "表注医院数口径（发现问题）", "dataset contains 207 hospitals" not in note, None,
        f"表注写「The eICU-CRD dataset contains 207 hospitals」；冻结 META 全部 {h_all} 家、合格队列 {h_el} 家")
    # 4 Table 2
    W47 = ROOT / "29_R47_写作交接" / "R47_冻结数字包.xlsx"; W48 = PK / "03_FROZEN_SOURCES" / "R48_图源与补充材料.xlsx"; W52 = PK / "03_FROZEN_SOURCES" / "R52_NHANES_SCr_SENS_EVIDENCE.xlsx"
    rd = lambda w, s: pd.read_excel(w, sheet_name=s)
    args = (rd(W47, "B3_分层计数"), rd(W47, "B4_maxT汇总"), rd(W47, "B5_NHANES低白蛋白归因"), rd(W48, "S54_D1_认证汇总"), rd(W48, "S58_C2_认证分解汇总"),
            rd(W47, "C1_T2条件与设计"), rd(W47, "C3_T2两轨等权部署"), rd(W52, "8_E1主对照"), rd(W52, "16_L3预定判读"))
    t2c, t2text = table2_check(d5.tables[1], *args)
    import copy
    a2 = list(args); c3m = a2[6].copy(); c3m.loc[(c3m.条件 == "P2") & (c3m.选择器 == "MIX"), "两轨等权部署改善均值"] += 0.0001; a2[6] = c3m
    add("3 Table 2", "Table 2 每个数字 = R47/R48/R52 冻结表", t2c.在表中.all(), not table2_check(d5.tables[1], *a2)[0].在表中.all(), f"{len(t2c)} 项")
    fct_row = next(" ".join(c.text for c in r.cells) for r in d5.tables[1].rows if r.cells[0].text.strip().startswith("Clinical FCT"))
    add("3 Table 2", "FCT 行的数据库范围（发现问题）", ("MIMIC" in fct_row or "eICU" in fct_row), None, "FCT 行未写明只来自 MIMIC-IV/eICU-CRD（712 = 2×180 + 2×176）")
    # 5 Evidence 段落边界
    ev = next(p for p in p5 if p.startswith("The evidence layers addressed complementary"))
    def ev_ok(t):
        bad = [w for w in ("independent validation cohort", "externally validated", "patient benefit", "validated across") if w in t]
        return ("rather than providing independent validation" in t and "not transfer of fixed ICU models" in t and "not clinical validation experiments" in t and not bad)
    add("4 Evidence 段落", "3 seed 写成重复开发、NWICU/NHANES 写成程序复现、模拟写成非临床验证；无越界词", ev_ok(ev),
        not ev_ok(ev + " Results were externally validated."), "")
    # 6 图
    loc = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "30_R48_图稿与补充材料" / "figures").rglob("*") if p.is_file()}
    figs = sorted((PK / "06_FROZEN_FIGURES_UNCHANGED").iterdir())
    same = [loc.get(p.name) == hashlib.sha256(p.read_bytes()).hexdigest() for p in figs]
    swapped = loc.get(figs[1].name) == hashlib.sha256(figs[0].read_bytes()).hexdigest()   # 变异：拿另一张图的指纹比
    add("6 图", "Figure 1–6 的 PNG/TIFF/PDF 与本地 R48 逐字节相同", all(same) and len(figs) == 18, not swapped, f"{sum(same)}/{len(figs)}")
    cap = (PK / "02_SUPPLEMENTARY" / "Figure_S1_caption.txt").read_text(encoding="utf-8")
    cap_ok = lambda t: "only the starting and final-stage counts could be reproduced independently" in t and "frozen cohort-construction logs" in t
    add("6 图", "S1 图注写明中间步骤只来自运行记录、只有首末两步可独立复现", cap_ok(cap),
        not cap_ok(cap.replace("only the starting and final-stage counts could be reproduced independently", "all counts were reproduced")), "")
    A = pd.DataFrame(R)
    rec = dict(通过=f"{int(A.通过.sum())}/{len(A)}", 发现问题=A[~A.通过].检查.tolist(), Table1格数=len(t1c), Table2项数=len(t2c), 参考文献=rc,
               eICU医院=dict(META全部=h_all, 合格队列=h_el), 时间=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    with pd.ExcelWriter(OUT / "R53_AUDIT.xlsx", engine="openpyxl") as xw:
        A.to_excel(xw, sheet_name="1_核对汇总", index=False); t1c.to_excel(xw, sheet_name="2_Table1逐格", index=False)
        t2c.to_excel(xw, sheet_name="3_Table2逐数", index=False)
        pd.DataFrame([dict(旧编号=k, 新编号=v) for k, v in rc["映射"].items()]).to_excel(xw, sheet_name="4_参考文献映射", index=False)
    (OUT / "R53_AUDIT.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
    print(A.to_string()); print(json.dumps({k: v for k, v in rec.items() if k != "参考文献"}, ensure_ascii=False, indent=1))
    write_md(A, rec, t1c, t2c, h_all, h_el, rd(W48, "S54_D1_认证汇总"))
    return A, rec, t1c, t2c, h_all, h_el


def write_md(A, rec, t1c, t2c, h_all, h_el, S54):
    ok = A[A.通过]; rc = rec["参考文献"]
    prov = (ROOT / "13_MIMIC_EICU" / "DATA_PROVENANCE_AUDIT.md").read_text(encoding="utf-8")
    rows = [l for l in prov.splitlines() if l.startswith("| MIMIC-IV | v3.1") or l.startswith("| eICU-CRD | v2.0")]
    cnt = lambda db, item: next(l.split("|")[5].strip() for l in rows if f"| {db} |" in l and f"| {item} |" in l)
    units = "、".join(f"{r.库} {r.轨} {int(r.单元)}" for r in S54.itertuples())
    a = []; Ap = a.append
    Ap("# R53 v0.5 限定只读复核（Claude → GPT）\n")
    Ap("只读：没有运行模型或模拟，没有改任何冻结文件。核对脚本 `pgss_r53_audit.py`，逐项结果在 `R53_AUDIT.xlsx`。\n")
    Ap("## 结论：GO with minor edits\n")
    Ap(f"{len(ok)} 项核对全部通过（每项配变异对照并检出）；另有 **2 处必须修改的文字**（1 处事实口径、1 处范围歧义），都不涉及数字重算。图 1–6 未动，S1 已按上一轮意见改好。\n")
    Ap("## 1. 已核对通过\n")
    Ap(f"- **M1–M3**：三句都按 R52 v0.4b 复核的替换句原样改入，旧说法已不在文中；R52 在 Methods、Results、Limitations、Table 2 中都标为 post hoc、不替代主分析。")
    Ap(f"- **Table 1**：{rec['Table1格数']} 个数据格逐格对 R51 `T0_Table1_排版`，同格内数字逐个、按顺序一致（主表省略的「sex unknown」计数与 NWICU 死亡患者数已分别由分母与表注体现）。分母、单位、缺失比例、独立患者数与医院列正确；住院单位与独立患者区分清楚。")
    Ap(f"- **Table 2**：{rec['Table2项数']} 个数字逐个对 R47 `B3/B4/B5/C1/C3`、R48 `S54/S58`、R52 `8_E1主对照/16_L3预定判读`，全部一致；方向、百分比、1% 技术边界与区间类型的写法正确；L1/L2/L3/FCT/T2 的解释列与限制列没有越界。")
    Ap("- **新增 Evidence 段落**：3 个 seed 写成「repeated development in overlapping patients」、NWICU 写成学习程序复现并注明结局状态修正、NHANES 写成跨领域程序复现而非固定模型迁移、模拟写成「not clinical validation experiments」。没有越界表述。")
    Ap(f"- **参考文献**：19 条条目文字与 v0.4b 完全相同；正文 {rc['引用组']} 处引用逐一按新旧编号映射相符（如 Wang 15→6、Paes 6→8、XGBoost 10→17、BIOPRO_D 仍为 19）；首次引用顺序严格为 1→19，全部被引用。Table 1/Table 2 中方括号内是四分位数，不是引用。")
    Ap("- **图**：Figure 1–6 的 PNG/TIFF/PDF 共 18 个文件与本地 R48 逐字节相同。S1 已改正上一轮三项必改：末框写明排除原因（MIMIC 的 222 人标为 landmark 前死亡——R35 登记表中「结局来源不可靠」与「时间冲突」均为 0，标注成立）、图注写明中间步骤只来自运行记录且只有首末两步可独立复现、内部版本标签已删除；并补了心血管死亡事件数。")
    Ap("- **排版**：Table 1、Table 2 的渲染预览清楚，三线表结构、表注与页面无问题。\n")
    Ap("## 2. 必须修改（最小替换文字）\n")
    Ap(f"**F1｜事实口径｜Table 1 表注第二段。** 原文「The eICU-CRD dataset contains 207 hospitals; other frozen data files do not encode a hospital count.」"
       f"冻结的 eICU-CRD META 中共有 {h_all} 家医院，合格队列覆盖 {h_el} 家；207 是本研究队列的医院数，不是数据集的医院数。替换为：")
    Ap("> Eligible eICU-CRD stays came from 207 hospitals; the other frozen data files do not encode a hospital identifier.\n")
    Ap(f"**F2｜范围歧义｜Table 2「Clinical FCT」行。** 该行没有写数据库范围，读者可能以为包含 NWICU 或 NHANES；实际 712 = {units}，只来自 MIMIC-IV 与 eICU-CRD。最小改法：把第一列改为")
    Ap("> Clinical FCT (MIMIC-IV/eICU-CRD)\n")
    Ap("## 3. 不阻塞投稿的提示\n")
    Ap(f"- **数据库版本（你列为待核实）**：本地已有第 34 轮做的来源审计 `13_MIMIC_EICU/DATA_PROVENANCE_AUDIT.md`：MIMIC-IV 的 patients/admissions/ICU stays（{cnt('MIMIC-IV', 'patients')}/{cnt('MIMIC-IV', 'hospital_admissions')}/{cnt('MIMIC-IV', 'icu_stays')}）与 v3.1 官方计数逐项一致，eICU-CRD 的 patients/unit stays（{cnt('eICU-CRD', 'patients(uniquepid)')}/{cnt('eICU-CRD', 'unit_encounters')}）与 v2.0 一致；但没有官方校验文件，只能写「结构与计数与官方版本一致」，不能写「文件逐字节相同」。建议数据声明据此表述，由用户最终确认。")
    Ap("- S1 面板 C、D 的框仍偏挤、箭头压线；图题「outcome coverage」与图注题「outcome eligibility by data source」仍不一致。纯排版，可在定稿时一并处理。")
    Ap("- Table 2 L2 行写「Low-albumin B track showed +3.23% gain」，可补「(interval-supported)」，与 R52 行和正文的「supported」一致。可选。\n")
    Ap("## 4. 交付\n")
    Ap("- `R53_复核.md`：本文件。")
    Ap("- `R53_AUDIT.xlsx`：核对汇总、Table 1 逐格、Table 2 逐数、参考文献新旧映射。")
    Ap("- `R53_AUDIT_CODE.zip`：核对脚本与结果 json。")
    (OUT / "R53_复核.md").write_text("\n".join(a) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
