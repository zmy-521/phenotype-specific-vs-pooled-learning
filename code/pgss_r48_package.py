# -*- coding: utf-8 -*-
r"""R48 打包与交付前自检 → 30_R48_图稿与补充材料/R48_FIGURES_AND_CODE.zip。

内容：figures/（六张图 PNG、300 dpi TIFF、PDF，与 R48_FIGURE_TEXT.json）；figure_sources/（xlsx 中 F 开头各表的 CSV）；
      code/（本轮脚本 + 全部既往脚本的 import 闭包；冻结脚本原样）+ code/external/nhanes.py；requirements.txt；PATHS.md；README.md；
      t2_records/（O1 的输入：arms、sel 的 parquet 与逐位保真 CSV，合成数据）；R48_ENV.json；R48_NUM_TRACE.json；tests/R48_PACKAGE_TESTS.json。
不打包：任何患者级预测、runs/、_local_only/、文献 PDF、受限数据。xlsx 与 md 单独交付，不在 zip 内重复。
自检：pgss_r43_package.selfaudit（zip 内容 + 报告数字舍入溯源）+ 逐值追踪与报告 SHA（trace_audit）+ 图文件齐全 + 包测试全部通过。
用法：python pgss_r48_package.py
"""
import sys, json, pathlib, platform, hashlib, zipfile, io, os
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from pgss_r43_package import selfaudit
import pgss_r48_sources as SRC

S = SRC.OUT
MD = S / "R48_图稿与补充材料索引.md"; XL = S / "R48_图源与补充材料.xlsx"; ZIP = S / "R48_FIGURES_AND_CODE.zip"
FIGDIR = S / "figures"
FIG_STEMS = ["R48_Fig1_study_design", "R48_Fig2_ICU_benchmark", "R48_Fig3_NHANES_attribution", "R48_Fig4_learning_curves",
             "R48_Fig5_T2_mechanisms", "R48_Fig6_T2_selection"]
RECS = SRC.R47D / "t2_records"


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


README = """# PGSS — R48 figures, figure sources, code and environment

Accompanies the R48 hand-off (GPT round 52: presentation and delivery only; no new fitting, no new simulation, no recomputed model performance).
Contains **no patient-level data**.

## Layout
- `figures/` — six main figures, each as PNG (300 dpi), TIFF (300 dpi, RGB, LZW; same pixels as the PNG) and PDF.
  `R48_FIGURE_TEXT.json` lists every text string printed on each figure.
- `figure_sources/` — CSV copies of the `F*` sheets of `R48_图源与补充材料.xlsx` (delivered separately). Every plotted value and every in-figure
  note comes from these sheets (`F0_图注要点` holds the notes).
- `code/` — all analysis scripts (frozen versions unchanged) and their import closure; `code/external/nhanes.py` is the NHANES helper.
- `requirements.txt` — exact package versions used for all frozen runs and for these figures.
- `PATHS.md` — every hard-coded absolute path, project directory and environment variable found by scanning the code, with what to change.
- `t2_records/` — inputs of the post hoc oracle table O1 (T2 per-replicate `arms` and `sel`, synthetic): parquet and gzip CSV written with `%.17g`
  (read with `pandas.read_csv(..., float_precision="round_trip")`).

## Re-drawing the figures from the delivered workbook
```
python code/pgss_r48_figures.py R48_图源与补充材料.xlsx figures_out
```
PNG output is byte-identical on the recorded environment (`R48_ENV.json`, `requirements.txt`).

## What the other scripts need
`pgss_r48_sources.py` and `pgss_r48_report.py` read earlier delivered result workbooks from the project layout (see `PATHS.md`); the real-data
pipeline needs credentialed PhysioNet access (MIMIC-IV, eICU-CRD, NWICU) and the public NHANES files. The synthetic T2 example from R47
(`code/pgss_r47_example.py`) needs nothing else.
"""


def env_record(code):
    e1 = SRC.e1_env()
    return dict(python=sys.version, platform=platform.platform(), logical_cpus=os.cpu_count(),
                packages={r.包: r.版本 for r in e1.itertuples()},
                code_sha={f: sha((HERE / f).read_bytes()) for f in code}, nhanes_module=dict(sha=sha(SRC.closure()[1].read_bytes())),
                依赖闭包=dict(入口=SRC.ENTRY, 文件数=len(code)))


def fig_sources(xl=XL):
    x = pd.read_excel(xl, sheet_name=None)
    return {f"figure_sources/{k}.csv": v.to_csv(index=False).encode("utf-8-sig") for k, v in x.items() if k.startswith("F")}


def build_zip(with_tests=True):
    code, nh = SRC.closure()
    env = env_record(code)
    (S / "R48_ENV.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in code:
            z.write(HERE / f, f"code/{f}")
        z.write(nh, "code/external/nhanes.py")
        for stem in FIG_STEMS:
            for ext in (".png", ".tif", ".pdf"):
                z.write(FIGDIR / f"{stem}{ext}", f"figures/{stem}{ext}")
        z.write(FIGDIR / "R48_FIGURE_TEXT.json", "figures/R48_FIGURE_TEXT.json")
        for k, b in fig_sources().items():
            z.writestr(k, b)
        for t in ("arms", "sel"):
            for ext in (".parquet", ".csv.gz"):
                z.write(RECS / f"{t}{ext}", f"t2_records/{t}{ext}")
        z.writestr("README.md", README.encode("utf-8"))
        for fn in ("requirements.txt", "PATHS.md", "R48_ENV.json", "R48_NUM_TRACE.json"):
            z.write(S / fn, fn)
        if with_tests and (S / "R48_PACKAGE_TESTS.json").exists():
            z.write(S / "R48_PACKAGE_TESTS.json", "tests/R48_PACKAGE_TESTS.json")
    return buf.getvalue(), code


def trace_audit(md_path):
    tj = S / "R48_NUM_TRACE.json"
    if not tj.exists():
        return ["缺逐值追踪记录 R48_NUM_TRACE.json"]
    t = json.loads(tj.read_text(encoding="utf-8")); probs = []
    if t["找不到"]:
        probs.append(f"逐值追踪：{len(t['找不到'])} 个原值不在报告引用表中")
    if t["变异对照被放行"]:
        probs.append(f"逐值追踪的变异对照被放行 {t['变异对照被放行']} 个")
    if sha(pathlib.Path(md_path).read_bytes()) != t.get("报告sha"):
        probs.append("报告与逐值追踪登记的 SHA 不符（生成后被改动）")
    return probs


def audit_all(zb, md_path, xl_path):
    probs, nn, na = selfaudit(zb, pathlib.Path(md_path), pathlib.Path(xl_path))
    return probs + trace_audit(md_path), nn, na


def main():
    zb, code = build_zip()
    probs, nn, na = audit_all(zb, MD, XL)
    names = zipfile.ZipFile(io.BytesIO(zb)).namelist()
    miss = [f"figures/{s}{e}" for s in FIG_STEMS for e in (".png", ".tif", ".pdf") if f"figures/{s}{e}" not in names]
    if miss:
        probs.append(f"缺图文件：{miss}")
    if not (S / "R48_PACKAGE_TESTS.json").exists():
        probs.append("包测试尚未运行（R48_PACKAGE_TESTS.json）")
    elif not json.loads((S / "R48_PACKAGE_TESTS.json").read_text(encoding="utf-8")).get("全部通过"):
        probs.append("包测试未全部通过")
    print(f"zip {len(zb)/1024/1024:.2f} MB，sha256 {sha(zb)}；代码 {len(code)} 个；条目 {len(names)}；报告数字 {nn} 个；随机命中率 {na['随机命中率']}")
    if probs:
        print("自检未通过：\n  " + "\n  ".join(probs)); return False
    ZIP.write_bytes(zb)
    (S / "R48_PACKAGE_AUDIT.json").write_text(json.dumps(dict(zip_sha=sha(zb), zip_mb=round(len(zb) / 2**20, 2), 代码文件=len(code), 条目=len(names),
                                                              report_numbers=nn, numaudit=na, problems=[]), ensure_ascii=False, indent=1), encoding="utf-8")
    print("自检通过，已写 R48_FIGURES_AND_CODE.zip")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
