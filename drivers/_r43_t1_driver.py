# -*- coding: utf-8 -*-
r"""R43 T1 驱动（GPT 第 48 轮 §6 的顺序）：锁定配置 → 测 8 个完整重复的实际成本 → 外推 →
预计墙钟 ≤ 48 小时则连续完成 320 个重复；超出则停止并写明原因（**不删场景、不减竞争者、不改边界**）。

哨兵：t1/_COST.json、t1/_DONE_cost、t1/_DONE_main | t1/_FAILED_* | t1/_COST_STOP；日志 t1_driver.log。
"""
import sys, json, pathlib, time, subprocess, os
HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent / "99_代码"
sys.path.insert(0, str(CODE))
PY = sys.executable
T1DIR = HERE / "t1"
JOBS = 15
BUDGET_H = 48.0
LOG = HERE / "t1_driver.log"


def log(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + s + "\n")


def step(args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY, str(CODE / "pgss_r43_t1.py")] + args, cwd=str(CODE), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log(f"{args} 退出码 {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.returncode


def main():
    import pgss_r43_t1 as T1
    log("驱动启动")
    if step(["config"]) != 0:
        (T1DIR / "_FAILED_driver").write_text("config", encoding="utf-8"); return
    if step(["run", "8", "--only", "cost"]) != 0 or not (T1DIR / "_DONE_cost").exists():
        (T1DIR / "_FAILED_driver").write_text("cost", encoding="utf-8"); return
    rows = []
    for a in T1.job_list("cost"):
        m = json.loads((T1.RUNS / T1.tag_of(*a) / "rep.json").read_text(encoding="utf-8"))
        rows.append(dict(scen=a[0], n=a[1], pi=a[2], sec=m["sec_total"], fits=m["n_fits"]))
    per = {}
    for r in rows:
        per.setdefault((r["n"], r["pi"]), []).append(r["sec"])
    cpu_s = sum(sum(v) / len(v) * T1.REPS * len(T1.SCEN_MAIN) for v in per.values())
    done_s = sum(r["sec"] for r in rows)
    wall_h = (cpu_s - done_s) / JOBS / 3600 * 1.15          # 15 进程并行时的争用余量 15%
    cost = dict(实测重复=rows, 按条件平均秒={f"n{k[0]}_pi{k[1]}": sum(v) / len(v) for k, v in per.items()},
                外推累计秒=cpu_s, 剩余预计墙钟小时=round(wall_h, 2), 并发=JOBS, 预算小时=BUDGET_H,
                说明="外推只用于决定是否在预算内；不据此删场景、减竞争者或改边界")
    (T1DIR / "_COST.json").write_text(json.dumps(cost, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"成本外推：剩余预计墙钟 {wall_h:.2f} 小时")
    if wall_h > BUDGET_H:
        (T1DIR / "_COST_STOP").write_text(json.dumps(cost, ensure_ascii=False), encoding="utf-8")
        log("超出预算，停止，等待报告与裁决"); return
    rc = step(["run", str(JOBS), "--only", "main"])
    log(f"main 结束 退出码 {rc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log("驱动异常 " + traceback.format_exc())
        (T1DIR / "_FAILED_driver").write_text(str(e), encoding="utf-8")
