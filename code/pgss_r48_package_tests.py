# -*- coding: utf-8 -*-
r"""R48 包测试：每项检查都先用于真实交付物（必须通过），再用于一个变异对照（必须被检出）。结果写 30_R48_图稿与补充材料/R48_PACKAGE_TESTS.json。

  1 自包含：zip 解压后隔离解释器导入 code/ 全部模块；变异 = 删 pgss_r48_sources.py 后 pgss_r48_report 导入必须失败
  2 图可由图源重现：解压出的代码读交付的 xlsx 重画六张图，PNG、TIFF、PDF 与包内逐字节相同；变异 = 改 F5 一个值后 Fig5 必须不同
  3 TIFF：300 dpi、RGB、LZW、像素与 PNG 相同；变异 = 150 dpi 的 TIFF 必须被判不合格
  4 图文字清单：任何图上没有 population / as in real data；Fig6 有 CV_LB_M 图例；Fig2–Fig6 印有 F0 的图内说明；Fig6 说明含 FCT 与 b≠c 两句；
    变异 = 从 F0 删去 Fig6 的 FCT 句、或在 Fig2 说明里加 Population-weighted，重画后必须被检出
  5 图源与 R47：F1、F2、F3b、F4、F5 与 R47 逐值相同；F3a 只改文字列；F6 中 R47 原有的行逐值相同，CV_LB_M|P* 的行与 R47 C3/C4/C6b 逐值相同；
    变异 = 改 F2 一个值、改 F6a 中 CV_LB_M 一行的值，必须被检出
  6 接回的 S 表与源表逐值相同；变异 = 改一个单元格
  7 索引完整：本包表名都存在、处理取值合法、接回都有建议编号、已被取代都写明存在的取代者、包含声明逐行成立；变异 = 删一张 S 表、清空一个取代者
  8 O1 独立复算：用另一份实现从 zip 内逐位保真的 CSV 重算 O1；变异 = 表中一个值乘 1 + 1e-9
  9 冻结脚本未改：闭包中 R47 登记过的脚本 SHA 与登记值相同；变异 = 临时副本多一个空格
 10 requirements.txt：覆盖 code/ 中全部第三方导入，且版本等于运行环境；变异 = 删去 xgboost 一行
 11 PATHS：独立扫描 code/ 中全部绝对路径字面量，每处都在 E2 中有说明；变异 = 加一个含新绝对路径的文件
 12 数字溯源与隐私：真实 zip 与报告两层通过；编造/改动数字、逐人预测文件、本地专用目录、标识列/键、口令必须被检出
用法：python pgss_r48_package_tests.py
"""
import sys, io, re, ast, json, zipfile, pathlib, tempfile, subprocess, shutil, os, time, importlib.metadata as IM
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from PIL import Image
import pgss_r48_package as P
import pgss_r48_sources as SRC
from pgss_r47_package_tests import import_all
from pgss_r46_package_tests import bump, pick
from pgss_r45_package_tests import mkzip, pqb

PY = sys.executable
ENV = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)} | {"PYTHONIOENCODING": "utf-8"}
PGVAR = "PG" + "PASSWORD"
NOTE_FIGS = ["Fig2", "Fig3", "Fig4", "Fig5", "Fig6"]
STEM = dict(zip(["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6"], P.FIG_STEMS))


def run(args, cwd, timeout=1800):
    t0 = time.time()
    r = subprocess.run([PY, "-I", "-X", "utf8"] + args, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV,
                       timeout=timeout)
    return r.returncode, r.stdout[-3000:], r.stderr[-2000:], round(time.time() - t0, 1)


def render(code_dir, xl, out, figs):
    """在隔离解释器中用解压出的图脚本只画指定的图，返回该次的图文字清单。"""
    src = (f"import sys, json; sys.argv=['x', {str(xl)!r}, {str(out)!r}]; sys.path.insert(0, {str(code_dir)!r})\n"
           "import pgss_r48_figures as F\n"
           f"for k in {figs!r}: F.FIGS[k]()\n"
           "print('@@' + json.dumps(F.TEXTS, ensure_ascii=False))\n")
    r = subprocess.run([PY, "-I", "-X", "utf8", "-c", src], capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    line = [l for l in r.stdout.splitlines() if l.startswith("@@")]
    return json.loads(line[0][2:]) if line else {"_err": r.stderr[-800:]}


def write_xl(sheets, path):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)


# ── 各项检查函数（真实对象与变异对象用同一函数）───────────────────
def chk_tif(png, tif):
    with Image.open(tif) as t, Image.open(png) as p:
        dpi = tuple(round(float(x)) for x in t.info.get("dpi", (0, 0)))
        ok = dpi == (300, 300) and t.mode == "RGB" and t.info.get("compression") == "tiff_lzw" and \
            np.array_equal(np.asarray(t), np.asarray(p.convert("RGB")))
        return ok, f"dpi {dpi}，{t.mode}，{t.info.get('compression')}"


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def chk_texts(texts, f0):
    probs = []
    for fig, ts in texts.items():
        j = " ".join(ts)
        if re.search(r"(?i)population[- ]weighted", j) or any(re.match(r"\s*Population\b", t) for t in ts):   # 尺度名；"general population" 是人群描述，不算
            probs.append(f"{fig} 含 population 尺度名")
        if "as in real data" in j:
            probs.append(f"{fig} 含 as in real data")
    n = f0.set_index("图")["图内说明"]
    for k in NOTE_FIGS:
        st = STEM[k]
        if st in texts and norm(n[k]) not in norm(" ".join(texts[st])):
            probs.append(f"{k} 图上没有 F0 的图内说明")
    t6 = norm(" ".join(texts.get(STEM["Fig6"], [])))
    if STEM["Fig6"] in texts:
        for must in ("CV lower bound > m", "FCT deploys models fitted on F", "it is not the false-enabling rate in c"):
            if must not in t6:
                probs.append(f"Fig6 缺「{must}」")
    return probs


VAL6 = {"F6a_T2部署两轨等权": (["条件", "选择器"], "C3_T2两轨等权部署"), "F6b_T2逐轨实际变差": (["条件", "选择器", "轨"], "C4_T2逐轨部署与变差"),
        "F6c_T2启用错误": (["条件", "选择器"], "C6b_T2重复级启用错误")}


def same_frame(a, b):
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False, check_exact=True)
        return True
    except AssertionError:
        return False


def chk_sources(x48, x47):
    probs = []
    for k in ("F1_研究设计", "F2_ICU复现", "F3b_NHANES归因_绝对", "F4_学习曲线", "F5_T2机制"):
        if not same_frame(x48[k], x47[k]):
            probs.append(f"{k} 与 R47 不同")
    a, b = x48["F3a_NHANES归因_相对"], x47["F3a_NHANES归因_相对"]
    keep = [c for c in b.columns if c not in ("Scale", "Comparison")]
    if not same_frame(a[keep], b[keep]):
        probs.append("F3a 文字列以外有改动")
    for k, (key, src) in VAL6.items():
        new, old = x48[k], x47[k]
        m = old.merge(new, on=key, how="left", suffixes=("_47", "_48"), indicator=True)
        vc = [c for c in old.columns if c not in key]
        if (m._merge != "both").any() or not all(np.array_equal(m[f"{c}_47"].to_numpy(), m[f"{c}_48"].to_numpy()) for c in vc):
            probs.append(f"{k}：R47 原有行不一致")
        lb = new[new.选择器 == "CV_LB_M|P*"]
        ref = x47[src]; ref = ref[ref.选择器 == "CV_LB_M|P*"][new.columns]
        mm = lb.merge(ref, on=key, suffixes=("", "_src"))
        if len(lb) == 0 or len(mm) != len(lb) or not all(np.array_equal(mm[c].to_numpy(), mm[f"{c}_src"].to_numpy()) for c in vc):
            probs.append(f"{k}：CV_LB_M|P* 行与 R47 {src} 不同")
    return probs


def chk_copies(x48, idx):
    probs = []
    for r in idx[idx.处理 == "接回"].itertuples():
        if r.本包表名 not in x48:
            probs.append(f"缺 {r.本包表名}"); continue
        if not same_frame(x48[r.本包表名], SRC.rd(r.来源, r.工作表)):
            probs.append(f"{r.本包表名} 与源表 {r.来源}∷{r.工作表} 不同")
    return probs


def chk_index(idx, sheetnames, a3):
    probs = []
    ok_how = {"R47包内", "接回", "本轮新增", "仅索引", "已被取代", "缺口"}
    if not set(idx.处理) <= ok_how:
        probs.append(f"处理取值不合法：{set(idx.处理) - ok_how}")
    for r in idx.itertuples():
        if isinstance(r.本包表名, str) and r.本包表名 and r.本包表名 not in sheetnames:
            probs.append(f"本包表名不存在：{r.本包表名}")
        if r.处理 == "接回" and not (isinstance(r.建议编号, str) and r.建议编号.startswith("S")):
            probs.append(f"接回无编号：{r.工作表}")
        if r.处理 == "已被取代":
            if not (isinstance(r.取代者, str) and "∷" in r.取代者):
                probs.append(f"已被取代但无取代者：{r.来源}∷{r.工作表}"); continue
            sk, ss = r.取代者.split("∷")
            if sk not in SRC.INP or ss not in pd.ExcelFile(SRC.INP[sk][0]).sheet_names:
                probs.append(f"取代者不存在：{r.取代者}")
    if not a3.通过.all():
        probs.append("包含声明不成立")
    return probs


def oracle_indep(arms_csv):
    """另一份实现：只用 numpy，从逐位保真 CSV 计算 O1 的主要列。"""
    a = pd.read_csv(arms_csv, float_precision="round_trip")
    reps = sorted(a.重复.unique())
    out = {}
    for nm, cs in SRC.ORACLE_SETS.items():
        for c in SRC.CONDS:
            rs = [r for r in reps if r.split("__")[0] == c]
            bal, A, B, ha, hb = [], [], [], 0, 0
            for r in rs:
                g = a[a.重复 == r].set_index("臂")
                pa, pb = g.loc["P*_D", "R_A"], g.loc["P*_D", "R_B"]
                ca, cb = g.loc[cs, "R_A"], g.loc[cs, "R_B"]
                ra, rb = ca.min(), cb.min()
                bal.append(((pa + pb) / 2 - (ra + rb) / 2) / ((pa + pb) / 2)); A.append((pa - ra) / pa); B.append((pb - rb) / pb)
                ha += ca.idxmin() == "HTFS_D"; hb += cb.idxmin() == "HTFS_D"
            bal = np.array(bal)
            out[(nm, c)] = dict(两轨等权改善均值=bal.mean(), SE=bal.std(ddof=1) / np.sqrt(len(bal)), A轨改善均值=np.mean(A), B轨改善均值=np.mean(B),
                                A轨取HTFS_D的重复=ha, B轨取HTFS_D的重复=hb)
    return out


def chk_oracle(o1, indep, tol=1e-14):
    worst, probs = 0.0, []
    for r in o1.itertuples():
        d = indep[(r.候选集, r.条件)]
        for k, v in d.items():
            x = getattr(r, k)
            if k.endswith("重复"):
                if int(x) != int(v):
                    probs.append(f"{r.候选集} {r.条件} {k}")
            else:
                worst = max(worst, abs(float(x) - float(v)))
    if worst > tol:
        probs.append(f"最大差 {worst:.2e} > {tol}")
    return probs, worst


def frozen_cmp(paths, registered):
    return [f for f, p in paths.items() if f in registered and SRC.sha16(pathlib.Path(p).read_bytes()) != registered[f]]


def third_party(code_dir):
    found = set()
    files = list(code_dir.glob("*.py")) + list((code_dir / "external").glob("*.py"))
    local = {p.stem for p in files}
    for f in files:
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                found |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                found.add(n.module.split(".")[0])
    return sorted(m for m in found if m not in sys.stdlib_module_names and m not in local)


def chk_req(req_text, code_dir):
    pins = dict(re.findall(r"(?m)^([A-Za-z0-9_.-]+)==([^\s#]+)", req_text))
    probs = []
    for m in third_party(code_dir):
        p = SRC.PIP.get(m, m)
        if p not in pins:
            probs.append(f"未固定：{p}")
        elif pins[p] != IM.version(p):
            probs.append(f"{p} {pins[p]} ≠ 环境 {IM.version(p)}")
    return probs, pins


ABS = re.compile(r'["\']([A-Za-z]:[/\\][^"\']*)["\']')


def chk_paths(e2, code_dir):
    have = {(r.文件, int(r.行)) for r in e2[e2.类型 == "绝对路径"].itertuples()}
    own = set(SRC.ABS_ROLE)
    miss = []
    for f in list(code_dir.glob("*.py")) + list((code_dir / "external").glob("*.py")):
        name = f.name if f.parent.name != "external" else f"external/{f.name}"
        for i, l in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in ABS.finditer(l):
                lit = m.group(1).replace("\\\\", "/").replace("\\", "/")
                if name == "pgss_r48_sources.py" and lit in own and (name, i) not in have:
                    continue                                   # 路径说明字典自身的键
                if (name, i) not in have:
                    miss.append(f"{name}:{i} {lit}")
    return miss


def main():
    rows = []
    add = lambda 检查, 通过, 说明="": rows.append(dict(检查=检查, 通过=bool(通过), 说明=说明))
    zb, code = P.build_zip(with_tests=False)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="r48pkg_"))
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        z.extractall(tmp)
    cd = tmp / "code"
    x48 = pd.read_excel(P.XL, sheet_name=None)
    x47 = pd.read_excel(SRC.INP["R47数字包"][0], sheet_name=None)
    idx, f0 = x48["S0_补充材料索引"], x48["F0_图注要点"]

    # 1 自包含
    mods = sorted(p.stem for p in cd.glob("*.py"))
    res = import_all(cd, mods + ["nhanes"], {PGVAR: "placeholder-not-a-password"})
    bad = {k: v for k, v in res.items() if v}
    add("自包含：隔离解释器导入 code/ 全部模块", res and not bad, f"{len(mods)} 个模块 + external/nhanes" + (f"；失败 {bad}" if bad else ""))

    # 2 图重现
    fo = tmp / "fig_re"
    rc, out, err, sec = run([str(cd / "pgss_r48_figures.py"), str(P.XL), str(fo)], tmp)
    same = {e: sum((fo / f"{s}{e}").exists() and (fo / f"{s}{e}").read_bytes() == (tmp / "figures" / f"{s}{e}").read_bytes() for s in P.FIG_STEMS)
            for e in (".png", ".tif", ".pdf")}
    add("图可由图源重现：六张图的 PNG、TIFF、PDF 逐字节相同", rc == 0 and all(v == 6 for v in same.values()),
        "；".join(f"{e} {v}/6" for e, v in same.items()) + f"；{sec} 秒" + ("" if rc == 0 else f"；{err[-300:]}"))
    xm = {k: v.copy() for k, v in x48.items()}
    xm["F5_T2机制"].loc[0, "真值相对收益均值"] += 0.005
    write_xl(xm, tmp / "mut_f5.xlsx")
    render(cd, tmp / "mut_f5.xlsx", tmp / "fig_mut", [5])
    p5 = f"{STEM['Fig5']}.png"
    add("图变异对照：改动 F5 图源一个值后 Fig5 必须不同", (tmp / "fig_mut" / p5).exists() and
        (tmp / "fig_mut" / p5).read_bytes() != (tmp / "figures" / p5).read_bytes())

    # 3 TIFF
    tif = [chk_tif(tmp / "figures" / f"{s}.png", tmp / "figures" / f"{s}.tif") for s in P.FIG_STEMS]
    add("TIFF：300 dpi、RGB、LZW，像素与 PNG 相同（六张）", all(t[0] for t in tif), "；".join(t[1] for t in tif[:1]) + f"；{sum(t[0] for t in tif)}/6")
    with Image.open(tmp / "figures" / f"{P.FIG_STEMS[0]}.png") as im:
        im.convert("RGB").save(tmp / "lowdpi.tif", format="TIFF", dpi=(150, 150), compression="tiff_lzw")
    add("TIFF 变异对照：150 dpi 的 TIFF 必须被判不合格", not chk_tif(tmp / "figures" / f"{P.FIG_STEMS[0]}.png", tmp / "lowdpi.tif")[0])

    # 4 图文字
    texts = json.loads((tmp / "figures" / "R48_FIGURE_TEXT.json").read_text(encoding="utf-8"))
    t_re = json.loads((fo / "R48_FIGURE_TEXT.json").read_text(encoding="utf-8")) if (fo / "R48_FIGURE_TEXT.json").exists() else {}
    pr = chk_texts(texts, f0)
    add("图文字清单：无 population / as in real data；Fig6 有 CV_LB_M；Fig2–Fig6 印有 F0 图内说明；重画后清单相同", not pr and texts == t_re,
        "；".join(pr) if pr else f"{sum(len(v) for v in texts.values())} 条文字")
    xm = {k: v.copy() for k, v in x48.items()}
    i6 = xm["F0_图注要点"].图 == "Fig6"
    xm["F0_图注要点"].loc[i6, "图内说明"] = xm["F0_图注要点"].loc[i6, "图内说明"].str.replace(r"FCT deploys.*$", "", regex=True)
    write_xl(xm, tmp / "mut_f0a.xlsx"); ta = render(cd, tmp / "mut_f0a.xlsx", tmp / "fig_m6", [6])
    add("图文字变异：删去 Fig6 说明中的 FCT 句必须被检出", bool(chk_texts(ta, f0)) and "_err" not in ta)
    xm = {k: v.copy() for k, v in x48.items()}
    i2 = xm["F0_图注要点"].图 == "Fig2"
    xm["F0_图注要点"].loc[i2, "图内说明"] = xm["F0_图注要点"].loc[i2, "图内说明"] + " Population-weighted scale."
    write_xl(xm, tmp / "mut_f0b.xlsx"); tb = render(cd, tmp / "mut_f0b.xlsx", tmp / "fig_m2", [2])
    add("图文字变异：图上出现 Population-weighted 必须被检出", any("population" in p for p in chk_texts(tb, xm["F0_图注要点"])) and "_err" not in tb)

    # 5 图源与 R47
    pr = chk_sources(x48, x47)
    add("图源：F1、F2、F3b、F4、F5 与 R47 逐值相同；F3a 只改文字列；F6 原有行不变，CV_LB_M|P* 行与 R47 C3/C4/C6b 相同", not pr, "；".join(pr))
    xm = {k: v.copy() for k, v in x48.items()}; xm["F2_ICU复现"].loc[3, "观察相对改善"] *= 1 + 1e-12
    add("图源变异：F2 一个值乘 1 + 1e-12 必须被检出", bool(chk_sources(xm, x47)))
    xm = {k: v.copy() for k, v in x48.items()}; j = xm["F6a_T2部署两轨等权"].index[xm["F6a_T2部署两轨等权"].选择器 == "CV_LB_M|P*"][3]
    xm["F6a_T2部署两轨等权"].loc[j, "上"] += 1e-9
    add("图源变异：F6a 中 CV_LB_M|P* 一行改动必须被检出", bool(chk_sources(xm, x47)))

    # 6 接回的 S 表
    pr = chk_copies(x48, idx)
    add("接回的 S 表与源表逐值相同", not pr, "；".join(pr[:5]) if pr else f"{int((idx.处理 == '接回').sum())} 张")
    xm = dict(x48)
    s1 = next(s for s in idx[idx.处理 == "接回"].本包表名 if any(x48[s][c].dtype.kind == "f" for c in x48[s].columns))   # 第一张含浮点列的 S 表
    t = xm[s1].copy(); num = [c for c in t.columns if t[c].dtype.kind == "f"]
    c0 = next(c for c in num if (np.isfinite(t[c]) & (t[c] != 0)).any())
    i0 = t.index[np.isfinite(t[c0]) & (t[c0] != 0)][0]
    t.loc[i0, c0] = t.loc[i0, c0] * (1 + 1e-12)                       # 有限非零值（NaN 乘任何数仍是 NaN，会让变异落空）
    xm[s1] = t
    add("S 表变异：一个单元格改动必须被检出", bool(chk_copies(xm, idx)), f"{s1} 列 {c0} 第 {i0} 行")

    # 7 索引完整
    pr = chk_index(idx, set(x48), x48["A3_取代声明核对"])
    add("索引完整：表名存在、处理合法、接回有编号、取代者存在、包含声明成立", not pr, "；".join(pr[:5]) if pr else f"{len(idx)} 条")
    add("索引变异：删一张 S 表必须被检出", bool(chk_index(idx, set(x48) - {s1}, x48["A3_取代声明核对"])))
    im2 = idx.copy(); k = im2.index[im2.处理 == "已被取代"][0]; im2.loc[k, "取代者"] = ""
    add("索引变异：清空一个取代者必须被检出", bool(chk_index(im2, set(x48), x48["A3_取代声明核对"])))

    # 8 O1 独立复算
    ind = oracle_indep(tmp / "t2_records" / "arms.csv.gz")
    pr, worst = chk_oracle(x48["O1_有限D候选硬选择oracle"], ind)
    add("O1 独立复算（另一份实现，读逐位保真 CSV）", not pr, f"最大绝对差 {worst:.1e}" + (f"；{pr}" if pr else ""))
    om = x48["O1_有限D候选硬选择oracle"].copy(); om.loc[2, "两轨等权改善均值"] *= 1 + 1e-9
    add("O1 变异：一个值乘 1 + 1e-9 必须被检出", bool(chk_oracle(om, ind)[0]))

    # 9 冻结脚本
    reg = json.loads((SRC.R47D / "R47_ENV.json").read_text(encoding="utf-8"))["code_sha"]
    paths = {f: cd / f for f in reg if (cd / f).exists()}
    bad = frozen_cmp(paths, reg)
    add("冻结脚本未改：闭包中 R47 登记过的脚本 SHA 与登记值相同", not bad and len(paths) > 40, f"{len(paths)} 个" + (f"；不同 {bad}" if bad else ""))
    mf = tmp / "mut_frozen.py"; mf.write_bytes((cd / "pgss_xgb_engine.py").read_bytes() + b" ")
    add("冻结脚本变异：副本多一个空格必须被检出", bool(frozen_cmp({"pgss_xgb_engine.py": mf}, reg)))

    # 10 requirements
    req = (tmp / "requirements.txt").read_text(encoding="utf-8")
    pr, pins = chk_req(req, cd)
    add("requirements.txt 覆盖全部第三方导入且版本等于运行环境", not pr, f"{len(pins)} 个固定版本" + (f"；{pr}" if pr else ""))
    add("requirements 变异：删去 xgboost 一行必须被检出", bool(chk_req(re.sub(r"(?m)^xgboost==.*\n", "", req), cd)[0]))

    # 11 PATHS
    e2 = x48["E2_路径说明"]
    miss = chk_paths(e2, cd)
    add("PATHS：code/ 中每处绝对路径字面量都在 E2 中有说明", not miss and "## 绝对路径" in (tmp / "PATHS.md").read_text(encoding="utf-8"),
        "；".join(miss[:5]) if miss else f"{int((e2.类型 == '绝对路径').sum())} 处")
    (cd / "zz_new.py").write_text('X = r"' + "D" + ':/somewhere/new"\n', encoding="utf-8")     # 运行时拼出，避免本文件含该字面量
    add("PATHS 变异：新增一个含绝对路径的文件必须被检出", bool(chk_paths(e2, cd)))
    (cd / "zz_new.py").unlink()

    # 12 数字溯源与隐私
    md0 = P.MD.read_text(encoding="utf-8")
    probs, n, na = P.audit_all(zb, P.MD, P.XL)
    add("数字溯源基线：真实 zip 与当前报告两层均通过", not probs, f"报告数字 {n} 个；随机命中率 {na['随机命中率']}" + (f"；{probs}" if probs else ""))
    p2 = pick(md0, r"(?<![\d.])-?\d+\.\d{2}%")
    rz = io.BytesIO(zb)
    cases = {"报告插入编造的六位小数": (zb, md0 + "\n编造 0.123457\n", ""),
             "报告插入编造的百分数": (zb, md0 + "\n编造 -0.9876%\n", ""),
             "报告改动一个两位小数百分数": (zb, md0.replace(p2, bump(p2), 1), f"{p2} → {bump(p2)}"),
             "zip 含逐人预测文件": (mkzip(rz, {"x/runs/u/preds_test.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0, ""),
             "zip 含本地专用目录": (mkzip(rz, {"x/_local_only/a.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0, ""),
             "parquet 含 stay_id 列": (mkzip(rz, {"x/y.parquet": pqb(pd.DataFrame({"stay_id": [1]}))}), md0, ""),
             "json 含 subject_id 键": (mkzip(rz, {"x/y.json": b'{"subject_id": 3}'}), md0, ""),
             "py 含真实口令": (mkzip(rz, {"code/b.py": ("PGPASS" + "WORD=" + "hunt" + "er2\n").encode()}), md0, "")}
    for nm, (z_, md, det) in cases.items():
        f = tmp / "r.md"; f.write_text(md, encoding="utf-8")
        a, _, _ = P.selfaudit(z_, f, P.XL); b = P.trace_audit(f)
        add(f"变异：{nm}", bool(a or b), (f"{det}；" if det else "") + f"第①层{'检出' if a else '未检出'}；第②层{'检出' if b else '未检出'}")

    # 1 的变异放最后（删文件）
    (cd / "pgss_r48_sources.py").unlink()
    mut = import_all(cd, ["pgss_r48_report"])
    add("自包含变异：删 pgss_r48_sources.py 后 pgss_r48_report 导入必须失败", bool(mut.get("pgss_r48_report")))
    shutil.rmtree(tmp, ignore_errors=True)
    out = dict(全部通过=all(r["通过"] for r in rows), 检查=rows, 代码文件=len(code))
    (P.S / "R48_PACKAGE_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return out


if __name__ == "__main__":
    sys.exit(0 if main()["全部通过"] else 1)
