# -*- coding: utf-8 -*-
r"""R51 打包：md、xlsx 复制到交接文件夹 → 带交付路径跑包测试并写日志 → 代码 zip（含 SHA256 清单）。"""
import sys, json, hashlib, pathlib, shutil, subprocess, zipfile
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = HERE.parent
OUT = ROOT / "34_R51_基线只读提数"
DST = pathlib.Path(r"D:\Research\3_对话与交接\GPT交接_PGSS\51\2_Claude_R51回复（待上传）")
PY = sys.executable


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for f in ("R51_Baseline_QC.md", "R51_Table1_Baseline_Source.xlsx"):
        shutil.copy2(OUT / f, DST / f)
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "pgss_r51_tests.py"), str(DST)], capture_output=True, text=True, encoding="utf-8")
    (OUT / "tests.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    print(r.stdout[-600:])
    if r.returncode != 0:
        raise SystemExit("包测试未通过，停止打包")
    files = {"R51_READONLY_BASELINE.py": HERE / "pgss_r51_baseline.py", "pgss_r51_report.py": HERE / "pgss_r51_report.py",
             "pgss_r51_tests.py": HERE / "pgss_r51_tests.py", "pgss_r51_package.py": HERE / "pgss_r51_package.py",
             "deps/pgss_r44_nhanes_cohort.py": HERE / "pgss_r44_nhanes_cohort.py",
             "logs/baseline_run.log": OUT / "baseline_run.log", "logs/tests.log": OUT / "tests.log",
             "R51_NUM_TRACE.json": OUT / "R51_NUM_TRACE.json", "R51_BASELINE.json": OUT / "R51_BASELINE.json"}
    readme = ("R51 代码包（只读）。\n"
              "R51_READONLY_BASELINE.py = 工作目录中的 pgss_r51_baseline.py；读 data v5、R45 NWICU、R44 NHANES 冻结文件与 R46 v2 结局状态，"
              "输出 R51_Table1_Baseline_Source.xlsx 与 R51_BASELINE.json，不写任何冻结目录。\n"
              "NHANES 白蛋白/BMI/eGFR 的内存重建需要 deps/pgss_r44_nhanes_cohort.py（R44 冻结代码副本，只取其中常量）与本地 NHANES 加载库 nhanes.py（不在包内）。\n"
              "pgss_r51_report.py 由 xlsx/json 与 R47/R48 冻结工作簿生成 R51_Baseline_QC.md 并登记每个数字；pgss_r51_tests.py 为包测试（7+1 项，各配变异对照）。\n"
              "患者级数据不在包内，运行需要本地冻结数据。\n")
    man = {k: sha(v) for k, v in files.items()}
    man.update({f"交付/{f}": sha(DST / f) for f in ("R51_Baseline_QC.md", "R51_Table1_Baseline_Source.xlsx")})
    z = DST / "R51_CODE.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in files.items():
            zf.write(v, k)
        zf.writestr("README.txt", readme)
        zf.writestr("SHA256_MANIFEST.json", json.dumps(man, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(z) as zf:
        bad = zf.testzip(); names = zf.namelist()
        same = all(zf.read(k) == pathlib.Path(v).read_bytes() for k, v in files.items())
    print("zip", z.name, len(names), "files; testzip", bad, "; 与源文件逐字节相同", same)
    print(sorted(p.name for p in DST.iterdir()))


if __name__ == "__main__":
    main()
