# -*- coding: utf-8 -*-
r"""R52 打包：报告与证据表复制到交接文件夹 → 带交付路径跑测试并写日志 → 代码/测试 zip（不含任何逐人文件）。"""
import sys, json, hashlib, pathlib, shutil, subprocess, zipfile, platform
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pgss_r52_sens as S

DST = pathlib.Path(r"D:\Research\3_对话与交接\GPT交接_PGSS\52\2_Claude_R52回复（待上传）")
DELIV = ("R52_NHANES_SCr_SENS_REPORT.md", "R52_NHANES_SCr_SENS_EVIDENCE.xlsx")
SCRIPTS = ["pgss_r52_sens.py", "pgss_r52_infer.py", "pgss_r52_compare.py", "pgss_r52_report.py", "pgss_r52_tests.py", "pgss_r52_package.py", "pgss_r52_noeco.py"]
RECORDS = ["R52_BUILD.json", "R52_BUILD.xlsx", "R52_EQUIV.json", "R52_PATHAUDIT.json", "R52_INFER_ZERO_CHECK.json", "R52_FOLDCHECK.json", "R52_FOLDCHECK.csv",
           "R52_LEAKCHECK.csv", "R52_FREEZE.json", "R52_COMPARE.json", "R52_NUM_TRACE.json", "_INFER_new.json", "zero_inference/_INFER_zero.json", "data/COHORT.json",
           "driver.log", "verify_freeze.log", "run_main.log", "run_target.log", "foldcheck.log", "infer_new.log", "equiv.log", "pathaudit.log", "tests.log",
           "R52_AMENDMENT.json", "amendment/pgss_r52_sens.diff", "amendment/pgss_r52_sens_frozen.py", "amendment/_R52_FAILED_首次运行_单元核对代码缺陷.txt"]
FORBID_TOKENS = (b"preds_test.parquet\x00", )


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def versions():
    from importlib.metadata import version
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pk in ("numpy", "pandas", "xgboost", "scikit-learn", "scipy", "pyarrow", "openpyxl", "psutil"):
        try:
            out[pk] = version(pk)
        except Exception:
            out[pk] = "未安装"
    return out


CONFIG_DIFF = """R52 配置差异（相对冻结的 R44 main / R45 目标臂 / R46 v2 程序）
唯一改动的预处理输入：NHANES 2005–2006 且血肌酐非缺失者 standardized = -0.016 + 0.978 × 未校准值（CDC/NCHS BIOPRO_D）；
随之重算 CKD-EPI 2021（与 nhanes.add_derived 同一表达式）与 eGFR降低 表型。
运行时替换（不改任何冻结文件）：pgss_r44_nhanes_run.DATA → 35_R52_NHANES_SCr2005_06_SENS/data；RUNS → 分支运行目录；
目标臂 HTFS 冻结规格读本分支同一单元的 main specs；推断脚本的 task_arrays 在 O_CVD10 上删除 R46 v2 不可评价者（同 R46）。
不变：引擎 xgb-1.0 全部配置与网格、适配器 r45-target-1.0、候选族、早停、调参、面板选择、交叉拟合 Platt、P* 选择、外层 3 seed × 5 折、
簇重抽 B=2000 SeedSequence(20441)、m=1% 三态、max-T、任务纳入与可执行门槛。
"""


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for f in DELIV:
        shutil.copy2(S.BR / f, DST / f)
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "pgss_r52_tests.py"), str(DST)], capture_output=True, text=True, encoding="utf-8")
    (S.BR / "tests.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    print(r.stdout[-1500:])
    if r.returncode != 0:
        raise SystemExit("测试未通过，停止打包")
    files = {f"code/{f}": HERE / f for f in SCRIPTS}
    files["code/_r52_driver.py"] = S.BR / "_r52_driver.py"
    files.update({f"records/{f}": S.BR / f for f in RECORDS if (S.BR / f).exists()})
    missing = [f for f in RECORDS if not (S.BR / f).exists()]
    man = {k: sha(v) for k, v in files.items()}
    man.update({f"交付/{f}": sha(DST / f) for f in DELIV})
    man.update({f"冻结原程序（未改，只登记）/{f}": sha(HERE / f) for f in S.FROZEN_FILES})
    readme = ("R52_NHANES_SCr2005_06_SENS 代码与测试包。\n"
              "code/：分支包装脚本（数据构建与拟合前核验 sens；推断包装 infer；旧新对比与 STOP/GO compare；报告 report；测试 tests；打包 package；EcoQoS 调度 noeco；驱动 _r52_driver）。\n"
              "records/：冻结、等价、干跑审计、零修正推断核对、单元核对、患者泄漏核对、对比判定、数字溯源、日志。\n"
              "原冻结核心脚本未改，只登记 SHA（SHA256_MANIFEST.json）。逐人预测、个人记录、分支数据 parquet 不在包内。\n"
              f"records 中缺少的文件：{missing or '无'}\n")
    z = DST / "R52_NHANES_SCr_SENS_CODE_TESTS.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in files.items():
            zf.write(v, k)
        zf.writestr("README.txt", readme)
        zf.writestr("CONFIG_DIFF.txt", CONFIG_DIFF)
        zf.writestr("requirements_versions.json", json.dumps(versions(), ensure_ascii=False, indent=1))
        zf.writestr("SHA256_MANIFEST.json", json.dumps(man, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist(); bad = zf.testzip()
        same = all(zf.read(k) == pathlib.Path(v).read_bytes() for k, v in files.items())
        priv = [n for n in names if n.endswith(".parquet") or "preds_test" in n or "_local_only" in n]
    print("zip", z.name, len(names), "files; testzip", bad, "; 逐字节相同", same, "; 逐人文件", priv, "; 缺失记录", missing)
    if priv or not same or bad:
        raise SystemExit("打包自检未通过")
    print(sorted(p.name for p in DST.iterdir()))


if __name__ == "__main__":
    main()
