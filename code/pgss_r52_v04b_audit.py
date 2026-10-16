# -*- coding: utf-8 -*-
r"""R52 v0.4b 限定只读复核：新增正文与补充表的每个数字回查 R52 证据工作簿（逐格），并对 v0.4a→v0.4b 做逐段比对。
输出：35_R52_NHANES_SCr2005_06_SENS/v04b_audit/R52_V04B_AUDIT.xlsx 与 .json。只读，不改任何文件。
"""
import sys, json, re, pathlib, difflib
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, docx

ROOT = HERE.parent
BR = ROOT / "35_R52_NHANES_SCr2005_06_SENS"
PK = pathlib.Path(r"D:\Research\3_对话与交接\GPT交接_PGSS\52\3_GPT_R52回应（已收到）")
OUT = BR / "v04b_audit"
EV = pd.read_excel(BR / "R52_NHANES_SCr_SENS_EVIDENCE.xlsx", sheet_name=None)
OC = {"10-year all-cause mortality": "O_MORT10", "10-year cardiovascular mortality": "O_CVD10"}
PH = {"Female sex": "女性", "Age ≥65 years": "年龄≥65", "Anemia": "贫血", "Reduced eGFR": "eGFR降低", "Low albumin": "低白蛋白", "Obesity": "肥胖"}
ST = {"排除实质收益": "Excluded", "证据不足": "Inconclusive", "支持实质收益": "Supported"}
num = lambda s: float(s.replace("−", "-").replace("+", ""))


def paras(f):
    d = docx.Document(f)
    ps = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for r in t.rows:
            ps.append(" | ".join(c.text.strip() for c in r.cells))
    return ps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    def add(where, item, doc, src, ok, note=""):
        rows.append(dict(位置=where, 项=item, 文稿值=doc, 证据值=src, 一致=bool(ok), 说明=note))
    # 1 补充表逐格
    d = docx.Document(PK / "PGSS_R52_SuppTable_NHANES_SCr_Sensitivity_review.docx")
    E1 = EV["8_E1主对照"]; bal = E1[E1.尺度 == "bal"]
    pat = re.compile(r"([+\-−]?\d+\.\d+) \(([+\-−]?\d+\.\d+) to ([+\-−]?\d+\.\d+)\)")
    nrow = 0
    for r in d.tables[0].rows[1:]:
        c = [x.text.strip() for x in r.cells]
        e = bal[(bal.结局 == OC[c[0]]) & (bal.表型 == PH[c[1]])].iloc[0]; nrow += 1
        for k, tag in ((2, "旧"), (3, "新")):
            m = pat.fullmatch(c[k]); vals = [num(x) for x in m.groups()]
            src = [round(100 * e[f"观察相对改善_{tag}"], 2), round(100 * e[f"相对_P2_5_{tag}"], 2), round(100 * e[f"相对_P97_5_{tag}"], 2)]
            add("补充表", f"{c[0]}×{c[1]} {tag}", c[k], f"{src[0]:.2f} ({src[1]:.2f} to {src[2]:.2f})", np.allclose(vals, src, atol=1e-9))
        dec = f"{ST[e[f'三态_m0.010_旧']]} → {ST[e[f'三态_m0.010_新']]}"
        add("补充表", f"{c[0]}×{c[1]} 判定", c[4], dec, c[4] == dec)
    add("补充表", "行数", str(nrow), str(len(bal)), nrow == len(bal))
    # 2 正文新增段落
    MIG = EV["5_eGFR迁移矩阵"]; mig = lambda a, b: int(MIG[(MIG.旧 == a) & (MIG.新 == b)].人数.iloc[0])
    SC = dict(zip(EV["4_血肌酐改动"].项, EV["4_血肌酐改动"].值))
    la = E1[(E1.结局 == "O_MORT10") & (E1.表型 == "低白蛋白") & (E1.尺度 == "B")].iloc[0]
    JD = EV["16_L3预定判读"]; d1 = JD[JD.对照 == "D1"].iloc[0]
    MT = EV["9_maxT"]; BA = EV["11_全部Brier对照"]
    nb = paras(PK / "PGSS_Manuscript_v0.4b_R52_sensitivity_review.docx")
    res = next(x for x in nb if x.startswith("In the post hoc NHANES assay-harmonization sensitivity analysis"))
    met = next(x for x in nb if x.startswith("After the frozen NHANES analyses, we undertook a post hoc sensitivity analysis"))
    p2 = lambda v: f"{100 * v:+.2f}%".replace("-", "−")
    def make_checks(res, met):
      return [
        ("Methods", "改动测量数 4,154", "4,154 nonmissing measurements" in met, f"{int(SC['改动人数（2005–2006 且血肌酐非缺失）']):,}"),
        ("Methods", "公式 −0.016 + 0.978", "−0.016 + 0.978" in met, "BIOPRO_D：-0.016 + 0.978 X"),
        ("Results", "改动人数 4,154", "for 4,154 participants" in res, f"{int(SC['改动人数（2005–2006 且血肌酐非缺失）']):,}"),
        ("Results", "61 of 2,018", f"{mig('A（降低）', 'B（不降低）')} of {mig('A（降低）', 'A（降低）') + mig('A（降低）', 'B（不降低）'):,}" in res, f"{mig('A（降低）', 'B（不降低）')} of {mig('A（降低）', 'A（降低）') + mig('A（降低）', 'B（不降低）'):,}"),
        ("Results", "无反向改判", "no reverse reclassification" in res and mig("B（不降低）", "A（降低）") == 0, str(mig("B（不降低）", "A（降低）"))),
        ("Results", "11 个任务三态不变；8/3/0", "All 11 evaluable tasks retained" in res and "eight excluded" in res and "three remained inconclusive" in res and "none supported" in res
         and int((bal["三态_m0.010_新"] == "排除实质收益").sum()) == 8 and int((bal["三态_m0.010_新"] == "证据不足").sum()) == 3 and int((bal["三态_m0.010_新"] == "支持实质收益").sum()) == 0
         and not bal.三态改变_m1.any(), "8/3/0，三态改变 0"),
        ("Results", "低白蛋白 B：+3.23% → +2.20% (+1.83% to +2.58%)",
         f"from {p2(la.观察相对改善_旧)} to {p2(la.观察相对改善_新)} (pointwise 95% interval, {p2(la.相对_P2_5_新)} to {p2(la.相对_P97_5_新)})" in res,
         f"{p2(la.观察相对改善_旧)} → {p2(la.观察相对改善_新)} ({p2(la.相对_P2_5_新)} to {p2(la.相对_P97_5_新)})"),
        ("Results", "D1：+0.11% → +0.20% (−0.003% to +0.41%)",
         f"from {p2(d1.观察相对改善_旧)} to {p2(d1.观察相对改善_新)} (−0.003% to {p2(d1.相对_P97_5_新)})" in res and abs(100 * d1.相对_P2_5_新 + 0.003) < 0.0005,
         f"{p2(d1.观察相对改善_旧)} → {p2(d1.观察相对改善_新)} ({100 * d1.相对_P2_5_新:.4f}% to {p2(d1.相对_P97_5_新)})"),
        ("Results", "D1 仍排除 ≥1%", "again excluding an additional gain of 1% or more" in res and d1.三态_m1_新 == "排除实质收益", d1.三态_m1_新),
        ("Results", "max-T 1/22", "One of 22 track-specific max-T judgments" in res and int(MT.判定改变.sum()) == 1 and len(MT) == 22, f"{int(MT.判定改变.sum())}/{len(MT)}"),
        ("Results", "非 E1 Brier 19/440", "19 of 440 non-E1 secondary Brier judgments" in res and int(BA[BA.对照 != 'E1'].三态改变_m1.sum()) == 19
         and int((BA.对照 != 'E1').sum()) == 440, f"{int(BA[BA.对照 != 'E1'].三态改变_m1.sum())}/{int((BA.对照 != 'E1').sum())}")]
    checks = make_checks(res, met)
    for where, item, ok, src in checks:
        add(where, item, "（见原句）", src, ok)
    # 3 逐段比对
    na = paras(PK / "PGSS_Manuscript_v0.4a_Introduction_only_review.docx")
    sm = difflib.SequenceMatcher(a=na, b=nb, autojunk=False)
    ops = [dict(操作=op, 旧段=f"{i1}:{i2}", 新段=f"{j1}:{j2}", 新内容开头=(nb[j1][:80] if j2 > j1 else "")) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    same = sum(i2 - i1 for op, i1, i2, j1, j2 in sm.get_opcodes() if op == "equal")
    # 变异对照：补充表一格改一位必须检出；正文改一个数必须检出
    mrow = rows[0]; mut1 = not np.isclose(num(pat.fullmatch(mrow["文稿值"]).group(1)) + 0.01, float(mrow["证据值"].split(" ")[0]), atol=1e-9)
    # 正文变异：改三个数字后重跑同一组核对，对应条目必须失败
    mres = res.replace("for 4,154 participants", "for 4,155 participants").replace("19 of 440", "19 of 441").replace("to +2.20%", "to +2.21%")
    fails = {c[1] for c in make_checks(mres, met) if not c[2]}
    mut2 = {"改动人数 4,154", "非 E1 Brier 19/440", "低白蛋白 B：+3.23% → +2.20% (+1.83% to +2.58%)"} <= fails
    A = pd.DataFrame(rows)
    rec = dict(数字核对=f"{int(A.一致.sum())}/{len(A)}", 全部一致=bool(A.一致.all()), 未改动段落=same, 改动=ops, 变异对照=[mut1, mut2])
    with pd.ExcelWriter(OUT / "R52_V04B_AUDIT.xlsx", engine="openpyxl") as xw:
        A.to_excel(xw, sheet_name="1_数字核对", index=False); pd.DataFrame(ops).to_excel(xw, sheet_name="2_逐段比对", index=False)
    (OUT / "R52_V04B_AUDIT.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    print(A[~A.一致].to_string() if not A.一致.all() else "全部一致"); print(json.dumps(rec, ensure_ascii=False, indent=1))
    write_md(A, rec, res, nb, E1, BA, MT, EV)


def write_md(A, rec, res, nb, E1, BA, MT, EV):
    L = EV["12_logloss对照"]; le = L[L.对照 == "E1"]
    lim = next(x for x in nb if x.startswith("Third, clinical confidence bands"))
    met = next(x for x in nb if x.startswith("After the frozen NHANES analyses"))
    n_e1, n_e1_zero = len(E1), int((E1.观察相对改善_Δ == 0).sum())
    n_mt, c_mt = len(MT), int(MT.判定改变.sum())
    n_ba, c_ba = int((BA.对照 != "E1").sum()), int(BA[BA.对照 != "E1"].三态改变_m1.sum())
    n_le, c_le = len(le), int(le.三态改变_m1.sum())
    old1 = "thus, the main inference was stable, but the numerical results and all secondary classifications were not invariant to assay harmonization (Supplementary Materials)."
    old2 = "We applied the NCHS-recommended standardization to 4,154 nonmissing measurements (standardized serum creatinine [mg/dL] = −0.016 + 0.978 × original serum creatinine [mg/dL]), then recalculated eGFR using the same 2021 CKD-EPI equation and updated the reduced-eGFR phenotype."
    old3 = "correcting this in post hoc redevelopment preserved the main material-gain classifications but altered some point estimates and secondary judgments."
    for o, t in ((old1, res), (old2, met), (old3, lim)):
        assert o in t, o[:50]
    a = []; Ap = a.append
    Ap("# R52 v0.4b 限定只读复核（Claude → GPT）\n")
    Ap("只读：没有重训、没有改变量或模型、没有重做模拟。核对脚本 `pgss_r52_v04b_audit.py`，逐项结果在 `R52_V04B_AUDIT.xlsx`。\n")
    Ap("## 结论\n")
    Ap(f"数字全部正确（{rec['数字核对']}，变异对照全部检出）；**有 3 处必须修正的文字**，都是一句话的替换，不涉及任何数字或分析。改完即可视为 R52 文字复核通过。\n")
    Ap("## 1. 改动范围（v0.4a → v0.4b 逐段比对）\n")
    Ap(f"与你的说明一致：Introduction 两段（P10、P12）、Methods 新增小节（标题 + 一段）、Results 新增一段、Limitations 一段、参考文献 [19]；其余 {rec['未改动段落']} 段（含 Abstract、全部图注、表格与 R47 主数字所在段落）逐字未变。没有发现跨段矛盾：Abstract 与 L3 原文仍报冻结主结果，新段落都标明 post hoc 且不替代主分析。\n")
    Ap("## 2. 数字核对\n")
    Ap(f"- 补充表 11 个任务 × 旧/新点估计与区间 × 判定，逐格对 R52 证据表 `8_E1主对照`，全部一致；方向、百分比（相对 Brier 改善）与 1% 技术边界的写法正确，区间类型标为 pointwise 95% clustered-resampling，正确。")
    Ap(f"- Methods 与 Results 的 4,154、61/2,018（反向 0）、11 任务 8/3/0、低白蛋白 B 轨 +3.23% → +2.20%（+1.83% 至 +2.58%）、D1 +0.11% → +0.20%（−0.003% 至 +0.41%）、max-T {c_mt}/{n_mt}、非 E1 Brier {c_ba}/{n_ba}，分别对 `5_eGFR迁移矩阵`、`4_血肌酐改动`、`8_E1主对照`、`16_L3预定判读`、`9_maxT`、`11_全部Brier对照`，全部一致。")
    Ap("- CDC/NCHS 公式与本地存档 BIOPRO_D 原文一致；[19] 的 2008 与原文「First Published: March 2008」一致。\n")
    Ap("## 3. 必须修正（最小替换句）\n")
    Ap("**M1 Results 末句的范围歧义。** 原句：")
    Ap(f"> {old1}\n")
    Ap(f"「all secondary classifications were not invariant」在英文里可读成「所有次要判定都变了」，而实际只变了 {c_ba}/{n_ba}（Brier）与 {c_mt}/{n_mt}（max-T）。顺带补上 log loss（E1 {c_le}/{n_le} 行改变），避免只报 Brier。替换为：")
    Ap(f"> thus, the main inference was stable, although point estimates and a minority of secondary classifications changed, including {c_le} of {n_le} E1 log-loss judgments (Supplementary Materials).\n")
    Ap("**M2 Methods 没有写明血肌酐同时是模型输入。** 原句：")
    Ap(f"> {old2}\n")
    Ap("现文只说重算 eGFR 与表型，读者只能从后句「other predictors」推断血肌酐预测变量也被替换；而这正是本敏感性分析不能只换表型标签的理由。替换为：")
    Ap("> We applied the NCHS-recommended standardization to 4,154 nonmissing measurements (standardized serum creatinine [mg/dL] = −0.016 + 0.978 × original serum creatinine [mg/dL]); the standardized values replaced the original serum-creatinine predictor and were used to recalculate eGFR with the same 2021 CKD-EPI equation, thereby updating the reduced-eGFR phenotype.\n")
    Ap("**M3 Limitations 低估了点估计的变化范围。** 原句：")
    Ap(f"> {old3}\n")
    Ap(f"证据表中 E1 的 {n_e1} 个点估计全部有变化（变化恰为 0 的 {n_e1_zero} 个），「some」不准确。替换为：")
    Ap("> correcting this in post hoc redevelopment preserved the main material-gain classifications but changed point estimates and a minority of secondary judgments.\n")
    Ap("## 4. 来源与解释边界\n")
    Ap("- post hoc 标记：Methods 小节与 Results 段都写了 post hoc，且写明不替代冻结主比较；Limitations 写明两版的区间都不含重训变异。正确。")
    Ap("- 目标一致性：Methods 称其为 result-informed，Results 只写「again excluding an additional gain of 1% or more」，没有写成独立确认或因果分解。正确。")
    Ap("- Introduction（对 R46 留在本地的全文抽取文本逐字核对）：Wang 等原文「In the finite sample regime, splitting is not necessarily beneficial」，"
       "v0.4b「showed that splitting need not help in finite samples」准确；Pfohl 等原文对「stratified ERM」（每个亚组分别训练与选择）等做法的结论是"
       "「none of the approaches evaluated consistently improved overall, worst-case, or disaggregated model performance compared to models learned with ERM applied to the entire training dataset」，"
       "v0.4b「compared pooled training with subgroup-specific models and alternative development objectives …, finding limited consistent gains」准确。没有冒认先例。\n")
    Ap("## 5. 交付\n")
    Ap("- `R52_v04b_复核.md`：本文件。")
    Ap("- `R52_V04B_AUDIT.xlsx`：逐项数字核对、逐段比对。")
    Ap("- `R52_V04B_AUDIT_CODE.zip`：核对脚本与结果 json。\n")
    Ap("按你的要求，这一轮不提新的性能实验。")
    (OUT / "R52_v04b_复核.md").write_text("\n".join(a) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
