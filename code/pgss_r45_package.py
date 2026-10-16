# -*- coding: utf-8 -*-
r"""R45 打包与交付前自检 → 27_R45_外部复现与机制核验/R45_CODE_TESTS.zip。

内容：代码（R45 脚本 + 冻结依赖 + 外部辅助模块 nhanes.py）、三份方案与冻结记录、运行前测试、路径审计、构建与推断汇总、驱动与日志、环境记录、包自检变异对照。
**不打包**：任何 runs/ 逐单元产物、data/ 个体数据、_local_only/ 暂存、逐人预测、患者编号。
自检沿用 R43（pgss_r43_package.selfaudit）：zip 内容检查 + 报告数字溯源（数值池 = 证据表「目录」中声明为报告引用的表），任一失败即不写 zip。
用法：python pgss_r45_package.py
"""
import sys, json, pathlib, platform, hashlib, zipfile, io, os
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from pgss_r43_package import selfaudit

ROOT = HERE.parent
S = ROOT / "27_R45_外部复现与机制核验"
MD = S / "R45_外部复现与机制核验_统一报告.md"; XL = S / "R45_外部复现与机制核验_证据表.xlsx"
NHMOD = pathlib.Path(r"D:/NAH公共数据库/NHANES/2_代码/nhanes.py")
CODE_FILES = ["pgss_r45_r43fix.py", "pgss_r45_r44supp.py", "pgss_r45_target.py", "pgss_r45_target_tests.py", "pgss_r45_target_inference.py",
              "pgss_r45_nwicu_stage.py", "pgss_r45_nwicu_build.py", "pgss_r45_nwicu_run.py", "pgss_r45_nwicu_inference.py",
              "pgss_r45_plasmode.py", "pgss_r45_plasmode_tests.py", "pgss_r45_plasmode_summary.py", "pgss_r45_report.py",
              "pgss_r45_package.py", "pgss_r45_package_tests.py",
              # 冻结依赖（只读）
              "pgss_xgb_engine.py", "pgss_r36_engine.py", "pgss_xgb_sim0.py", "pgss_r42_inference.py", "pgss_r43_readonly.py",
              "pgss_r43_t1.py", "pgss_r43_t1_summary.py", "pgss_r43_package.py", "pgss_r43_numaudit.py", "pgss_r44_nhanes_cohort.py",
              "pgss_r44_nhanes_run.py", "pgss_r44_nhanes_inference.py", "pgss_r38_bench.py", "pgss_threads.py"]
FILES = ["_r45_chain_driver.py", "chain_driver.log",
         "r43fix/R45_R43FIX.xlsx", "r43fix/R45_R43FIX.json", "r44supp/R45_R44SUPP.xlsx", "r44supp/R45_R44SUPP.json",
         "target/R45_NHANES_TARGET_PROTOCOL_v1.0.md", "target/R45_TARGET_FREEZE.json", "target/INCLUDED_TASKS.json",
         "target/R45_TARGET_TESTS.json", "target/R45_TARGET_EQUIV.json", "target/_r45_target_driver.py", "target/target_driver.log",
         "target/R45_TARGET_INFERENCE.xlsx", "target/R45_TARGET_INFERENCE.json", "target/_STATUS.json",
         "nwicu/R45_NWICU_PROTOCOL_v1.0.md", "nwicu/R45_NWICU_FREEZE.json", "nwicu/R45_NWICU_STAGE.json", "nwicu/R45_NWICU_UNITS.xlsx",
         "nwicu/R45_NWICU_BUILD.xlsx", "nwicu/data/COHORT.json", "nwicu/R45_NWICU_PATHAUDIT.json", "nwicu/R45_NWICU_PATHAUDIT.csv",
         "nwicu/inference/R45_NWICU_INFERENCE.xlsx", "nwicu/inference/R45_NWICU_INFERENCE.json", "nwicu/_STATUS.json",
         "plasmode/R45_PLASMODE_PROTOCOL_v1.0.md", "plasmode/R45_PLASMODE_FREEZE.json", "plasmode/R45_PLASMODE_TESTS.json",
         "plasmode/R45_PLASMODE_GENERATOR.json", "plasmode/R45_PLASMODE_GENERATOR.csv", "plasmode/R45_PLASMODE_SUMMARY.xlsx",
         "plasmode/R45_PLASMODE_SUMMARY.json", "plasmode/_STATUS.json",
         # 冻结后修订记录与各阶段日志
         "plasmode/R45_PLASMODE_AMENDMENT.json", "plasmode/chain_run.log", "plasmode/chain_summary.log", "plasmode/tests.log",
         "nwicu/build.log", "nwicu/pathaudit.log", "nwicu/chain_run.log", "nwicu/chain_inference.log",
         "target/equiv.log", "target/run_stdout.log", "target/post_inference.log"]


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def env_record():
    import xgboost, sklearn, scipy
    return dict(python=sys.version, platform=platform.platform(), logical_cpus=os.cpu_count(), numpy=np.__version__,
                pandas=pd.__version__, xgboost=xgboost.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
                code_sha={f: sha((HERE / f).read_bytes()) for f in CODE_FILES}, nhanes_module=dict(路径=str(NHMOD), sha=sha(NHMOD.read_bytes())))


def build_zip():
    env = env_record()
    (S / "R45_ENV.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE_FILES:
            z.write(HERE / f, f"code/{f}")
        z.write(NHMOD, "code/external/nhanes.py")
        for rel in FILES:
            if (S / rel).exists():
                z.write(S / rel, f"r45/{rel}")
        z.write(S / "R45_ENV.json", "R45_ENV.json")
        if (S / "R45_PACKAGE_TESTS.json").exists():
            z.write(S / "R45_PACKAGE_TESTS.json", "tests/R45_PACKAGE_TESTS.json")
    return buf.getvalue()


def main():
    missing = [rel for rel in FILES if not (S / rel).exists()]
    zb = build_zip()
    probs, nn, na = selfaudit(zb, MD, XL)
    if missing:
        probs.append(f"应交付文件缺失：{missing}")
    print(f"zip {len(zb)/1024/1024:.2f} MB，sha256 {sha(zb)}；报告数字 {nn} 个；随机命中率 {na['随机命中率']}")
    if probs:
        print("自检未通过：\n  " + "\n  ".join(probs)); return False
    (S / "R45_CODE_TESTS.zip").write_bytes(zb)
    (S / "R45_PACKAGE_AUDIT.json").write_text(json.dumps(dict(zip_sha=sha(zb), zip_mb=round(len(zb) / 2**20, 2),
                                                              report_numbers=nn, numaudit=na, problems=[]),
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print("自检通过，已写 R45_CODE_TESTS.zip")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
