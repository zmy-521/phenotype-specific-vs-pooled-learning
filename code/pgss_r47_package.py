# -*- coding: utf-8 -*-
r"""R47 打包与交付前自检 → 29_R47_写作交接/R47_CODE_AND_FIGURES.zip。

内容：code/（本轮脚本 + 真实数据流水线、T1、plasmode、T2 与 R46 全部脚本的 import 闭包；冻结脚本原样）+ code/external/nhanes.py；
      figures/（六张主图 PNG 与 PDF）；figure_sources/（数字包 F 开头各表的 CSV）；t2_records/（T2 逐重复记录 parquet + CSV + rep.jsonl，合成数据）；
      README.md；R47_ENV.json；R47_NUM_TRACE.json；tests/R47_PACKAGE_TESTS.json 与 tests/example_*.json（若已运行包测试）。
不打包：任何患者级预测、runs/ 下逐单元产物、_local_only/、文献 PDF 与抽取文本、受限数据。
自检：pgss_r43_package.selfaudit（zip 内容 + 报告数字舍入溯源）+ 本轮逐值追踪与报告 SHA（trace_audit）。
用法：python pgss_r47_package.py
"""
import sys, json, pathlib, platform, hashlib, zipfile, io, os
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from pgss_r43_package import selfaudit
import pgss_r46_package as P46

ROOT = HERE.parent
S = ROOT / "29_R47_写作交接"
MD = S / "R47_写作交接与图表索引.md"; XL = S / "R47_冻结数字包.xlsx"; ZIP = S / "R47_CODE_AND_FIGURES.zip"
NHMOD = P46.NHMOD
ENTRY = ["pgss_r47_numbers", "pgss_r47_figures", "pgss_r47_report", "pgss_r47_example", "pgss_r47_package", "pgss_r47_package_tests",
         "pgss_r383_build_v5", "pgss_xgb_run", "pgss_xgb_engine", "pgss_r42_inference",
         "pgss_r44_nhanes_cohort", "pgss_r44_nhanes_run", "pgss_r44_nhanes_inference",
         "pgss_r45_nwicu_build", "pgss_r45_nwicu_run", "pgss_r45_nwicu_inference", "pgss_r45_target", "pgss_r45_target_inference",
         "pgss_r45_plasmode", "pgss_r45_plasmode_summary", "pgss_r43_t1", "pgss_r43_t1_summary"] + P46.ENTRY


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


README = """# PGSS — phenotype-specific training vs phenotype-aware pooled learning: code, figures and simulation records

This archive accompanies the R47 writing hand-off. It contains **no patient-level data**. Real-data analyses need credentialed access
to MIMIC-IV, eICU-CRD, NWICU (PhysioNet) and the public NHANES files; the simulation (T2) and the example below need nothing else.

## Layout
- `code/` — all analysis scripts (frozen versions, unchanged) plus their local import closure; `code/external/nhanes.py` is the NHANES helper.
- `figures/` — six main figures (PNG 300 dpi and PDF). Every plotted value comes from `figure_sources/`.
- `figure_sources/` — CSV copies of the `F*` sheets of `R47_冻结数字包.xlsx` (delivered separately).
- `t2_records/` — per-replicate records of the T2 simulation (8 conditions × 200 replicates; synthetic data): `arms`, `pairs`, `sel`
  as parquet and as gzip CSV written with `%.17g` (read with `pandas.read_csv(..., float_precision="round_trip")` to recover the exact doubles), and `rep.jsonl`.
- `tests/` — package tests and the example results.

## Synthetic example (no restricted data)
```
python code/pgss_r47_example.py reproduce      # re-runs the frozen T2 replicate P2 #0 and compares it bit-for-bit with t2_records/ (~5–8 min)
python code/pgss_r47_example.py quick          # reduced-grid installation check (~2 min); numbers are NOT comparable to the paper
```
Bit-identical reproduction assumes the recorded package versions (`R47_ENV.json`); on other platforms the script reports the largest absolute difference.

## Real-data pipeline (entry scripts; directory layout as in the project, see table D7 of the number package)
| Step | Script(s) |
|---|---|
| MIMIC-IV / eICU-CRD data v5 | `pgss_r383_build_v5.py` (PostgreSQL; password only via the `PGPASSWORD` environment variable) |
| ICU main experiment, certification branch, learning curves | `pgss_xgb_run.py` with the frozen engine `pgss_xgb_engine.py` (xgb-1.0) |
| ICU inference (cluster bootstrap, max-T) | `pgss_r42_inference.py` |
| NWICU cohort, runs, inference (+ outcome correction) | `pgss_r45_nwicu_build.py`, `pgss_r45_nwicu_run.py`, `pgss_r45_nwicu_inference.py`, `pgss_r46_outcome_verify.py`, `pgss_r46_rerun.py`, `pgss_r46_inference.py` |
| NHANES cohort, runs, inference | `pgss_r44_nhanes_cohort.py`, `pgss_r44_nhanes_run.py`, `pgss_r44_nhanes_inference.py` |
| NHANES objective-alignment control | `pgss_r45_target.py`, `pgss_r45_target_inference.py` |
| T1 (exploratory simulation) | `pgss_r43_t1.py`, `pgss_r43_t1_summary.py` |
| Plasmode (exploratory; real MIMIC covariates) | `pgss_r45_plasmode.py`, `pgss_r45_plasmode_summary.py` |
| T2 (independent simulation) | `pgss_r46_t2.py`, `pgss_r46_t2_summary.py` |
| Number package, figures, report | `pgss_r47_numbers.py`, `pgss_r47_figures.py`, `pgss_r47_report.py` |

The frozen scripts keep their original project-relative paths (changing them would change the code fingerprints recorded in earlier rounds).
Definitions of every arm, selector, estimand and margin are in sheet `D3_方法定义` of the number package.
"""


def env_record(code):
    import xgboost, sklearn, scipy, matplotlib
    return dict(python=sys.version, platform=platform.platform(), logical_cpus=os.cpu_count(), numpy=np.__version__, pandas=pd.__version__,
                xgboost=xgboost.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                code_sha={f: sha((HERE / f).read_bytes()) for f in code}, nhanes_module=dict(sha=sha(NHMOD.read_bytes())),
                依赖闭包=dict(入口=ENTRY, 文件数=len(code)))


def fig_sources():
    xl = pd.read_excel(XL, sheet_name=None)
    return {f"figure_sources/{k}.csv": v.to_csv(index=False).encode("utf-8-sig") for k, v in xl.items() if k.startswith("F")}


def build_zip(with_tests=True):
    code = P46.closure(ENTRY)
    env = env_record(code)
    (S / "R47_ENV.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in code:
            z.write(HERE / f, f"code/{f}")
        z.write(NHMOD, "code/external/nhanes.py")
        for p in sorted((S / "figures").glob("R47_Fig*")):
            z.write(p, f"figures/{p.name}")
        for k, b in fig_sources().items():
            z.writestr(k, b)
        for p in sorted((S / "t2_records").iterdir()):
            z.write(p, f"t2_records/{p.name}")
        z.writestr("README.md", README.encode("utf-8"))
        z.write(S / "R47_ENV.json", "R47_ENV.json")
        z.write(S / "R47_NUM_TRACE.json", "R47_NUM_TRACE.json")
        if with_tests:
            for p in [S / "R47_PACKAGE_TESTS.json"] + sorted((S / "tests").glob("example_*.json")):
                if p.exists():
                    z.write(p, f"tests/{p.name}")
    return buf.getvalue(), code


def trace_audit(md_path):
    tj = S / "R47_NUM_TRACE.json"
    if not tj.exists():
        return ["缺逐值追踪记录 R47_NUM_TRACE.json"]
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
    need = [f"figures/R47_Fig{i}_" for i in range(1, 7)]
    miss = [n for n in need if not any(x.startswith(n) and x.endswith(".png") for x in names)]
    if miss:
        probs.append(f"缺主图：{miss}")
    if not (S / "R47_PACKAGE_TESTS.json").exists():
        probs.append("包测试尚未运行（R47_PACKAGE_TESTS.json）")
    else:
        t = json.loads((S / "R47_PACKAGE_TESTS.json").read_text(encoding="utf-8"))
        if not t.get("全部通过"):
            probs.append("包测试未全部通过")
    print(f"zip {len(zb)/1024/1024:.2f} MB，sha256 {sha(zb)}；代码 {len(code)} 个；条目 {len(names)}；报告数字 {nn} 个；随机命中率 {na['随机命中率']}")
    if probs:
        print("自检未通过：\n  " + "\n  ".join(probs)); return False
    ZIP.write_bytes(zb)
    (S / "R47_PACKAGE_AUDIT.json").write_text(json.dumps(dict(zip_sha=sha(zb), zip_mb=round(len(zb) / 2**20, 2), 代码文件=len(code), 条目=len(names),
                                                              report_numbers=nn, numaudit=na, problems=[]), ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print("自检通过，已写 R47_CODE_AND_FIGURES.zip")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
