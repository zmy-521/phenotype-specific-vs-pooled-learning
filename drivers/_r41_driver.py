# -*- coding: utf-8 -*-
r"""R41 Stage 0 长任务驱动：等 0A 选出代表任务 → 并行跑 0C（≤4 任务 × main/cert）与 0B 小模拟 → 写哨兵。

哨兵：_R41_RUN_DONE（全部成功）/ _R41_RUN_FAILED（任一失败，附摘要）。日志：driver.log 与各子任务日志。
"""
import json, pathlib, subprocess, sys, time

R = pathlib.Path(__file__).resolve().parent
CODE = R.parent / "99_代码"
PY = r"D:\Python\python.exe"
LOG = R / "driver.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {m}\n")


def main():
    for f in ("_R41_RUN_DONE", "_R41_RUN_FAILED"):
        (R / f).unlink(missing_ok=True)
    js = R / "stage0a" / "R41_STAGE0A.json"
    t0 = time.time()
    while not js.exists():
        if time.time() - t0 > 3 * 3600:
            (R / "_R41_RUN_FAILED").write_text("等待 0A 超时", encoding="utf-8"); return
        time.sleep(20)
    time.sleep(5)
    sel = json.loads(js.read_text(encoding="utf-8"))["代表任务"]
    log(f"0A 代表任务：{[(s['库'], s['结局'], s['表型'], s['角色']) for s in sel]}")
    procs = []
    (R / "logs").mkdir(exist_ok=True)
    for s in sel:
        for br in ("main", "cert"):
            tag = f"{s['库']}__{s['结局']}__{s['表型']}__{br}"
            out = open(R / "logs" / f"0c_{tag}.log", "w", encoding="utf-8")
            p = subprocess.Popen([PY, "-X", "utf8", str(CODE / "pgss_xgb_stage0c.py"), "--db", s["库"], "--oc", s["结局"],
                                  "--ph", s["表型"], "--branch", br], cwd=str(CODE), stdout=out, stderr=subprocess.STDOUT)
            procs.append((tag, p, out)); log(f"启动 0C {tag} pid={p.pid}")
    out = open(R / "logs" / "sim0.log", "w", encoding="utf-8")
    ps = subprocess.Popen([PY, "-X", "utf8", str(CODE / "pgss_xgb_sim0.py"), "8"], cwd=str(CODE), stdout=out,
                          stderr=subprocess.STDOUT)
    procs.append(("sim0", ps, out)); log(f"启动 sim0 pid={ps.pid}")
    status = {}
    for tag, p, f in procs:
        rc = p.wait(); f.close(); status[tag] = rc
        log(f"结束 {tag} rc={rc}  已用 {time.time()-t0:.0f}s")
    bad = {k: v for k, v in status.items() if v != 0}
    summ = json.dumps(dict(status=status, 用时秒=round(time.time() - t0)), ensure_ascii=False)
    (R / ("_R41_RUN_FAILED" if bad else "_R41_RUN_DONE")).write_text(summ, encoding="utf-8")
    log(f"全部结束：{summ}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        (R / "_R41_RUN_FAILED").write_text(f"driver 异常 {type(e).__name__}: {e}", encoding="utf-8")
        raise
