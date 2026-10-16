# -*- coding: utf-8 -*-
r"""R56 测试：对已生成的 R56_ICU_Predictor_Missingness.xlsx 做独立复核（不调用 pgss_r56_missingness 的计数函数）。

T1 用 pyarrow 另走一条路径（不经 pandas）按 stay 标识对齐 X 与 META 后重数 46 格，与 1_S5A_long 逐格比对（差异检测）。
T2 表内恒等式与百分比；打印宽表的字符串由长表数字反解后一致。
T3 打印表变量顺序与 Supplementary Table S4（v0.2 Word，独立解析）一致。
T4 输出只含聚合行，无个人标识列。
T5 运行后 data v5 文件指纹仍等于 DATA_VERSION.json 登记（证明只读）。
T6 R56 脚本与报告不含控制字符（heredoc 反斜杠事故的防线）。
每项都配一个必须被检出的变异对照。输出 R56_TESTS.json。
"""
import sys, json, re, hashlib, pathlib
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = HERE.parent
V5 = ROOT / "19_R38" / "data_v5"
OUT = ROOT / "38_R56_ICU预测变量缺失率"
XL = OUT / "R56_ICU_Predictor_Missingness.xlsx"
S4DOC = pathlib.Path(r"D:\Research\3_对话与交接\GPT交接_PGSS\53\5_GPT_R55交付包（已收到）\02_SUPPLEMENTARY\PGSS_Supplementary_v0.2_numbered_review.docx")
DBS = {"MIMIC-IV": ("mimic", "stay_id"), "eICU-CRD": ("eicu", "sid")}
SEX = "性别F"
ID_COLS = ("stay_id", "subject_id", "hadm_id", "patientunitstayid", "uniquepid", "group_id", "hospital_id", "sid")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def arrow_counts(tag, key, cols, drop_one=None):
    X = pq.read_table(V5 / f"R38_{tag}_X.parquet", columns=[c for c in cols if c != SEX] + [key])
    M = pq.read_table(V5 / f"R38_{tag}_META.parquet", columns=["eligible", "sex", key])
    assert X.column(key).equals(M.column(key)), "X 与 META 的 stay 标识顺序不同"
    elig = M.column("eligible").combine_chunks()
    out = {}
    for c in cols:
        if c == SEX:
            v = pc.is_in(M.column("sex"), value_set=__import__("pyarrow").array(["F", "M"])).combine_chunks()
            v = pc.fill_null(v, False)
        else:
            v = pc.invert(pc.is_null(X.column(c), nan_is_null=True)).combine_chunks()
        both = pc.and_(v, elig)
        if drop_one == c:                        # 变异：把第一个合格且有值的格子当作缺失
            arr = np.asarray(both.to_numpy(zero_copy_only=False)).copy(); arr[np.flatnonzero(arr)[0]] = False
            out[c] = int(arr.sum())
        else:
            out[c] = int(pc.sum(both).as_py() or 0)
    return out, int(pc.sum(elig).as_py())


def t1(L, drop=None):
    diffs = []
    for db, (tag, key) in DBS.items():
        sub = L[L.Database == db]
        cols = list(sub["Frozen column (data v5)"])
        cnt, N = arrow_counts(tag, key, cols, drop_one=(drop if db == "MIMIC-IV" else None))
        for _, r in sub.iterrows():
            v = r["Frozen column (data v5)"]; got = cnt[v]
            if got != r["Observed n"] or N != r["Eligible ICU stays (N)"]:
                diffs.append((db, v, int(r["Observed n"]), got))
    return diffs


def t2(L, P):
    ok = len(L) == 46 and bool(((L["Observed n"] + L["Missing n"]) == L["Eligible ICU stays (N)"]).all())
    ok &= bool((abs(L["Missing %"] - (100 * L["Missing n"] / L["Eligible ICU stays (N)"]).round(4)) < 1e-9).all())
    bad = []
    for r in P.itertuples(index=False):
        for db in DBS:
            x = L[(L.Database == db) & (L.Predictor == r.Predictor)].iloc[0]
            obs = int(str(P.loc[P.Predictor == r.Predictor, f"{db} observed n"].iloc[0]).replace(",", ""))
            ms = P.loc[P.Predictor == r.Predictor, f"{db} missing n (%)"].iloc[0]
            m = re.fullmatch(r"([\d,]+) \((<0\.1|\d+\.\d)\)", ms)
            pct = 100 * x["Missing n"] / x["Eligible ICU stays (N)"]
            exp = "0.0" if x["Missing n"] == 0 else ("<0.1" if pct < 0.05 else f"{pct:.1f}")
            if not m or obs != x["Observed n"] or int(m.group(1).replace(",", "")) != x["Missing n"] or m.group(2) != exp:
                bad.append((db, r.Predictor, ms))
    return ok and not bad, bad


def s4_labels():
    import docx
    d = docx.Document(str(S4DOC))
    body = d.element.body; cur = None
    for el in body.iterchildren():                # 按文档顺序：表题段落之后的第一张表
        tagname = el.tag.split("}")[1]
        if tagname == "p":
            txt = "".join(t.text or "" for t in el.iter() if t.tag.endswith("}t"))
            if txt.startswith("Supplementary Table S4."): cur = "S4"
        elif tagname == "tbl" and cur == "S4":
            rows = []
            for tr in el.iter():
                if tr.tag.endswith("}tr"):
                    tc = [c for c in tr if c.tag.endswith("}tc")]
                    rows.append("".join(t.text or "" for t in tc[0].iter() if t.tag.endswith("}t")))
            return rows[1:]
    raise SystemExit("找不到 S4")


def t4(sheets):
    bad = []
    for n, d in sheets.items():
        if len(d) > 200: bad.append((n, "行数", len(d)))
        hit = [c for c in d.columns if str(c).lower() in ID_COLS]
        if hit: bad.append((n, "标识列", hit))
    return bad


def t5(flip=False):
    man = json.loads((V5 / "DATA_VERSION.json").read_text(encoding="utf-8"))["v5_SHA"]
    bad = []
    for k, reg in man.items():
        db, t = k.split("_")
        b = bytearray((V5 / f"R38_{db}_{t}.parquet").read_bytes())
        if flip and k == "eicu_META": b[100] ^= 0x01
        if hashlib.sha256(bytes(b)).hexdigest()[:16] != reg: bad.append(k)
    return bad


def t6(extra=None):
    files = [HERE / f for f in ("pgss_r56_missingness.py", "pgss_r56_tests.py", "pgss_r56_report.py", "pgss_r56_package.py") if (HERE / f).exists()]
    md = OUT / "R56_READONLY_QC.md"
    if md.exists(): files.append(md)
    bad = [f.name for f in files if CTRL.search(f.read_text(encoding="utf-8"))]
    if extra is not None and CTRL.search(extra): bad.append("变异字符串")
    return bad, len(files)


def main():
    sheets = pd.read_excel(XL, sheet_name=None)
    L, P = sheets["1_S5A_long"], sheets["2_S5A_print"]
    R = []
    d1 = t1(L); d1m = t1(L, drop="MCV")
    R.append(dict(测试="T1 pyarrow 独立重数 46 格 = 1_S5A_long（差异检测）", 通过=not d1, 变异对照检出=bool(d1m), 说明=f"差异 {len(d1)} 格；变异后差异 {len(d1m)} 格"))
    ok2, bad2 = t2(L, P)
    P2 = P.copy(); P2.loc[0, "MIMIC-IV missing n (%)"] = "1,600 (2.4)"
    ok2m, _ = t2(L, P2)
    R.append(dict(测试="T2 恒等式、百分比与打印字符串反解一致", 通过=ok2, 变异对照检出=not ok2m, 说明=f"不一致 {bad2}"))
    s4 = s4_labels()
    R.append(dict(测试="T3 打印表变量顺序 = Supplementary Table S4（v0.2，按文档顺序独立解析）", 通过=list(P.Predictor) == s4, 变异对照检出=list(P.Predictor)[::-1] != s4, 说明=f"{len(s4)} 行"))
    b4 = t4(sheets); b4m = t4({**sheets, "变异": L.assign(stay_id=1)})
    R.append(dict(测试="T4 输出只含聚合行、无个人标识列", 通过=not b4, 变异对照检出=bool(b4m), 说明=f"{len(sheets)} 张工作表，最多 {max(len(d) for d in sheets.values())} 行"))
    b5 = t5(); b5m = t5(flip=True)
    R.append(dict(测试="T5 运行后 data v5 八个文件指纹仍 = 登记（只读）", 通过=not b5, 变异对照检出=bool(b5m), 说明=f"不符 {b5}"))
    b6, n6 = t6(); b6m, _ = t6("x\x03y")
    R.append(dict(测试="T6 R56 脚本与报告无控制字符", 通过=not b6, 变异对照检出=bool(b6m), 说明=f"{n6} 个文件；含控制字符 {b6}"))
    res = dict(测试=f"{sum(r['通过'] for r in R)}/{len(R)}", 变异对照=f"{sum(bool(r['变异对照检出']) for r in R)}/{len(R)}", 明细=R)
    (OUT / "R56_TESTS.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in R: print(r["通过"], r["变异对照检出"], r["测试"], "|", r["说明"])
    print("测试", res["测试"], "变异对照", res["变异对照"])
    if not all(r["通过"] and r["变异对照检出"] for r in R):
        raise SystemExit("R56 测试未全过")


if __name__ == "__main__":
    main()
