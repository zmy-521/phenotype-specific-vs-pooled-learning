# -*- coding: utf-8 -*-
r"""R44 NHANES 跨领域复现驱动：等 R43 的 T1 扩展结束（CPU 空出）→ 核对冻结指纹（方案、代码、数据都未在冻结后改动）→
正式运行 180 个单元（15 进程）→ 推断。哨兵 _R44_DONE | _R44_FAILED；日志 r44_driver.log。
冻结后若任何登记文件指纹变化，驱动拒绝运行（写 _R44_FAILED），不自动更新冻结记录。
"""
import sys, json, pathlib, time, subprocess, os, hashlib
HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent / "99_代码"
T1 = HERE.parent / "24_R43_结果补齐与T1" / "t1"
PY = sys.executable
LOG = HERE / "r44_driver.log"
STOP = ("_FAILED_main", "_FAILED_driver", "_COST_STOP", "_FAILED_ext_wait", "_FAILED_ext_tests", "_FAILED_ext_sha",
        "_FAILED_ext_driver", "_FAILED_ext_cost", "_FAILED_ext_all", "_EXT_COST_STOP")


def log(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + s + "\n")


def sha16(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def main():
    log("驱动启动，等待 R43 T1 扩展结束")
    while not (T1 / "_DONE_ext_all").exists():
        if any((T1 / s).exists() for s in STOP):
            log("R43 T1 未正常结束；CPU 情况不明，仍按顺序继续（只影响开始时间，不影响定义）"); break
        time.sleep(60)
    fz = json.loads((HERE / "R44_FREEZE.json").read_text(encoding="utf-8"))
    bad = []
    for rel, s in fz["文件SHA"].items():
        p = (HERE.parent / rel)
        if not p.exists() or sha16(p) != s:
            bad.append(rel)
    if bad:
        log(f"冻结后文件被改动：{bad}，拒绝运行"); (HERE / "_R44_FAILED").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        return
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log("冻结指纹核对通过，开始正式运行")
    r = subprocess.run([PY, str(CODE / "pgss_r44_nhanes_run.py"), "run", "15"], cwd=str(CODE), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log(f"运行结束 退出码 {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    r = subprocess.run([PY, str(CODE / "pgss_r44_nhanes_inference.py")], cwd=str(CODE), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    (HERE / "post_inference.log").write_text(f"退出码 {r.returncode}\n{r.stdout}\n{r.stderr}", encoding="utf-8")
    if r.returncode != 0:
        log("推断失败"); (HERE / "_R44_FAILED").write_text("inference", encoding="utf-8"); return
    (HERE / "_R44_DONE").write_text(time.strftime("%m-%d %H:%M:%S"), encoding="utf-8")
    log("全部完成")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("驱动异常 " + traceback.format_exc()); (HERE / "_R44_FAILED").write_text("exception", encoding="utf-8")
