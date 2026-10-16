# -*- coding: utf-8 -*-
r"""R49 打包与交付前自检 → 31_R49_只读补齐/R49_CODE_TESTS.zip。

内容：code/（R49 脚本及其 import 闭包；另含被按源码读取的冻结定义 pgss_r35_cohort.py、pgss_r35_qc.py；冻结脚本原样）；
      R49_ENV.json；R49_NUM_TRACE.json；tests/R49_PACKAGE_TESTS.json。
不打包：任何受限临床数据、逐人预测、损失或重抽矩阵；data v5 与 R35 队列 parquet 只在本地读取。
自检：pgss_r43_package.selfaudit（zip 内容 + 报告数字舍入溯源）+ 逐值追踪与报告 SHA + 包测试全部通过。
"""
import sys, json, pathlib, platform, hashlib, zipfile, io, os
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from pgss_r43_package import selfaudit
import pgss_r46_package as P46
import pgss_r49_report as R

S = R.OUT
MD, XL, ZIP = R.MD, R.XL, S / "R49_CODE_TESTS.zip"
ENTRY = ["pgss_r49_cohort", "pgss_r49_aux", "pgss_r49_report", "pgss_r49_package", "pgss_r49_tests"]
READ_AS_SOURCE = ["pgss_r35_cohort.py", "pgss_r35_qc.py"]


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def code_files():
    return sorted(set(P46.closure(ENTRY + [f[:-3] for f in READ_AS_SOURCE])))       # 按源码读取的冻结脚本连同其导入闭包一起打包


def build_zip(with_tests=True):
    code = code_files()
    import importlib.metadata as IM
    env = dict(python=sys.version, platform=platform.platform(), packages={p: IM.version(p) for p in ("numpy", "pandas", "openpyxl", "pyarrow")},
               code_sha={f: sha((HERE / f).read_bytes()) for f in code}, 入口=ENTRY, 按源码读取=READ_AS_SOURCE)
    (S / "R49_ENV.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in code:
            z.write(HERE / f, f"code/{f}")
        for fn in ("R49_ENV.json", "R49_NUM_TRACE.json"):
            z.write(S / fn, fn)
        if with_tests and (S / "R49_PACKAGE_TESTS.json").exists():
            z.write(S / "R49_PACKAGE_TESTS.json", "tests/R49_PACKAGE_TESTS.json")
        z.writestr("README.md", ("# PGSS R49 — read-only completion (cohort flow; cross-seed auxiliary metrics)\n\n"
                                 "No patient-level data. `code/pgss_r49_cohort.py` and `code/pgss_r49_aux.py` read the frozen data v5 summary files and "
                                 "previously delivered result workbooks from the project layout; `pgss_r49_report.py` writes the report and workbook from the same tables; "
                                 "`pgss_r49_tests.py` runs the checks (each paired with a mutation that must be detected).\n").encode("utf-8"))
    return buf.getvalue(), code


def trace_audit(md_path):
    tj = S / "R49_NUM_TRACE.json"
    if not tj.exists():
        return ["缺逐值追踪记录"]
    t = json.loads(tj.read_text(encoding="utf-8")); probs = []
    if t["找不到"]:
        probs.append(f"逐值追踪：{len(t['找不到'])} 个原值不在报告引用表中")
    if t["变异对照被放行"]:
        probs.append("逐值追踪的变异对照被放行")
    if sha(pathlib.Path(md_path).read_bytes()) != t.get("报告sha"):
        probs.append("报告与逐值追踪登记的 SHA 不符")
    return probs


def audit_all(zb, md_path, xl_path):
    probs, nn, na = selfaudit(zb, pathlib.Path(md_path), pathlib.Path(xl_path))
    return probs + trace_audit(md_path), nn, na


def main():
    zb, code = build_zip()
    probs, nn, na = audit_all(zb, MD, XL)
    if not (S / "R49_PACKAGE_TESTS.json").exists():
        probs.append("包测试尚未运行")
    elif not json.loads((S / "R49_PACKAGE_TESTS.json").read_text(encoding="utf-8")).get("全部通过"):
        probs.append("包测试未全部通过")
    print(f"zip {len(zb)/1024:.0f} KB，sha256 {sha(zb)}；代码 {len(code)} 个；报告数字 {nn} 个")
    if probs:
        print("自检未通过：\n  " + "\n  ".join(probs)); return False
    ZIP.write_bytes(zb)
    (S / "R49_PACKAGE_AUDIT.json").write_text(json.dumps(dict(zip_sha=sha(zb), 代码文件=len(code), report_numbers=nn, numaudit=na, problems=[]),
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print("自检通过，已写 R49_CODE_TESTS.zip")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
