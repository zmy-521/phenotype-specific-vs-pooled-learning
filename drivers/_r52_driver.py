# -*- coding: utf-8 -*-
"""R52 敏感性分支驱动：冻结核对 → main 180 单元 → 可执行性核对 → 目标臂（pop 165 + B 15）→ 单元核对 → 敏感性推断。
终态哨兵：_R52_DONE / _R52_FAILED（含一行摘要）。只写本分支目录。"""
import subprocess, sys, json, time, pathlib, os
PY = r"D:\Python\python.exe"
ROOT = pathlib.Path(r"D:\Research\2_课题\PGSS_表型引导选择性专业化")
CODE = ROOT / "99_代码"
BR = ROOT / "35_R52_NHANES_SCr2005_06_SENS"
LOG = BR / "driver.log"
JOBS = 14


def log(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + s + "\n")


def fail(s):
    log("FAILED: " + s); (BR / "_R52_FAILED").write_text(s, encoding="utf-8"); sys.exit(1)


def step(name, args, logname):
    log(f"开始 {name}")
    with open(BR / logname, "w", encoding="utf-8") as f:
        r = subprocess.run([PY, "-X", "utf8"] + args, cwd=CODE, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    log(f"结束 {name} rc={r.returncode}")
    return r.returncode


def main():
    if step("冻结核对", ["pgss_r52_sens.py", "verify_freeze"], "verify_freeze.log"):
        fail("冻结文件被改动（见 verify_freeze.log）")
    step("main 180 单元", ["pgss_r52_sens.py", "run", "main", str(JOBS)], "run_main.log")
    pa = json.loads((BR / "R52_PATHAUDIT.json").read_text(encoding="utf-8"))
    exp = {tuple(x) for x in pa["原R44不可执行单元"]}
    import importlib.util
    done = [p.name for p in (BR / "runs44").iterdir() if (p / "_DONE").exists()]
    failed = [p.name for p in (BR / "runs44").iterdir() if (p / "_FAILED").exists()]
    log(f"main 完成 {len(done)}、不可执行 {len(failed)}（原版不可执行 {len(exp)}）")
    if len(done) + len(failed) != 180 or len(failed) != len(exp):
        fail(f"main 可执行性与原版不同或有单元未运行：完成 {len(done)} 不可执行 {len(failed)}")
    step("目标臂 pop+B", ["pgss_r52_sens.py", "run", "target", str(JOBS)], "run_target.log")
    tdone = [p.name for p in (BR / "runs_target").iterdir() if (p / "_DONE").exists()]
    log(f"目标臂完成 {len(tdone)}/180")
    if len(tdone) != 180:
        fail(f"目标臂未全部完成：{len(tdone)}/180")
    if step("单元核对", ["pgss_r52_sens.py", "foldcheck"], "foldcheck.log"):
        fail("单元核对未通过（见 foldcheck.log）")
    if step("敏感性推断", ["pgss_r52_infer.py", "new"], "infer_new.log"):
        fail("敏感性推断失败（见 infer_new.log）")
    (BR / "_R52_DONE").write_text("R52 分支完成：main、目标臂、单元核对、敏感性推断全部通过", encoding="utf-8")
    log("完成")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        fail(f"驱动异常 {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
