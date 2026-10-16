# -*- coding: utf-8 -*-
r"""R45 串行驱动：等 NHANES 目标一致性运行结束（CPU 让出）→ NWICU（核对冻结 → 180 单元 → 推断）→ plasmode（核对冻结 → 60 重复 → 汇总）。
每段写哨兵：nwicu/_NWICU_DONE | _NWICU_FAILED；plasmode/_PLASMODE_DONE | _PLASMODE_FAILED；全部结束写 _CHAIN_DONE。日志 chain_driver.log。
冻结后任何登记文件变化 ⇒ 该段拒绝运行（不自动更新冻结记录）；NWICU 失败不阻止 plasmode（两者独立）。
"""
import sys, json, pathlib, time, subprocess, os, hashlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "99_代码"
PY = sys.executable
LOG = HERE / "chain_driver.log"
JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def log(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + s + "\n")


def frozen_ok(fz_path):
    fz = json.loads(fz_path.read_text(encoding="utf-8"))
    bad = [rel for rel, s in fz["文件SHA"].items()
           if not (ROOT / rel).exists() or hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16] != s]
    return bad, len(fz["文件SHA"])


def step(name, d, fz, cmds, done, failed):
    if (d / done).exists():
        log(f"{name} 已完成，跳过"); return
    bad, n = frozen_ok(d / fz)
    if bad:
        log(f"{name} 冻结后文件被改动：{bad}，拒绝运行"); (d / failed).write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8"); return
    log(f"{name} 冻结指纹核对通过（{n} 项）")
    for label, cmd in cmds:
        r = subprocess.run([PY] + cmd, cwd=str(CODE), env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (d / f"chain_{label}.log").write_text(f"退出码 {r.returncode}\n{r.stdout[-20000:]}\n{r.stderr[-20000:]}", encoding="utf-8")
        log(f"{name} {label} 退出码 {r.returncode}")
        if r.returncode != 0:
            (d / failed).write_text(label, encoding="utf-8"); return
    (d / done).write_text(time.strftime("%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"{name} 完成")


def main():
    T = HERE / "target"
    log("链式驱动启动，等待 NHANES 目标一致性运行结束")
    while not ((T / "_TARGET_DONE").exists() or (T / "_TARGET_FAILED").exists()):
        time.sleep(60)
    log("目标一致性运行已结束（" + ("DONE" if (T / "_TARGET_DONE").exists() else "FAILED") + "），开始 NWICU")
    NWD = HERE / "nwicu"
    step("NWICU", NWD, "R45_NWICU_FREEZE.json",
         [("run", [str(CODE / "pgss_r45_nwicu_run.py"), "run", str(JOBS)]),
          ("inference", [str(CODE / "pgss_r45_nwicu_inference.py")])], "_NWICU_DONE", "_NWICU_FAILED")
    PLD = HERE / "plasmode"
    step("plasmode", PLD, "R45_PLASMODE_FREEZE.json",
         [("run", [str(CODE / "pgss_r45_plasmode.py"), "run", str(JOBS)]),
          ("summary", [str(CODE / "pgss_r45_plasmode_summary.py")])], "_PLASMODE_DONE", "_PLASMODE_FAILED")
    (HERE / "_CHAIN_DONE").write_text(time.strftime("%m-%d %H:%M:%S"), encoding="utf-8")
    log("链式驱动结束")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("驱动异常 " + traceback.format_exc()); (HERE / "_CHAIN_FAILED").write_text("exception", encoding="utf-8")
