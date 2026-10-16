# -*- coding: utf-8 -*-
r"""R43 T1 扩展驱动：等原 320 个重复完成（t1/_DONE_main）→ 核对扩展配置与端到端测试已通过 →
8 个代表性完整重复测成本（含全部选择器与认证程序）→ 外推墙钟（只报告，不据此删重复、删比较策略）→ 完成全部 160 个。
超过 48 小时预计墙钟则停止并写 _EXT_COST_STOP 等裁决。日志 t1_ext_driver.log；哨兵 t1/_DONE_ext_all | _FAILED_ext_* | _EXT_COST_STOP。
"""
import sys, json, pathlib, time, subprocess, os
HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent / "99_代码"
sys.path.insert(0, str(CODE))
PY = sys.executable
T1DIR = HERE / "t1"
JOBS, BUDGET_H = 15, 48.0
LOG = HERE / "t1_ext_driver.log"


def log(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + s + "\n")


def step(args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY, str(CODE / "pgss_r43_t1_ext.py")] + args, cwd=str(CODE), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    log(f"{args} 退出码 {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.returncode


def main():
    log("扩展驱动启动，等待原 320 完成")
    while not (T1DIR / "_DONE_main").exists():
        if list(T1DIR.glob("_FAILED_main")) or (T1DIR / "_COST_STOP").exists() or (T1DIR / "_FAILED_driver").exists():
            log("原 320 未正常完成，扩展不启动"); (T1DIR / "_FAILED_ext_wait").write_text("main", encoding="utf-8"); return
        time.sleep(60)
    tests = json.loads((T1DIR / "R43_T1_EXT_E2E_TESTS.json").read_text(encoding="utf-8"))
    if not tests.get("全部通过"):
        log("扩展端到端测试未全部通过，不运行"); (T1DIR / "_FAILED_ext_tests").write_text("tests", encoding="utf-8"); return
    import pgss_r43_t1_ext as TX
    cfg = json.loads((T1DIR / "R43_T1_EXT_CONFIG.json").read_text(encoding="utf-8"))
    if cfg["扩展代码指纹"] != TX.ext_sha():
        log("扩展代码在配置之后被修改，不运行"); (T1DIR / "_FAILED_ext_sha").write_text("sha", encoding="utf-8"); return
    log("原 320 已完成；测试与配置指纹核对通过；开始成本测量")
    if step(["run", str(JOBS), "--only", "cost"]) != 0 or not (T1DIR / "_DONE_ext_cost").exists():
        (T1DIR / "_FAILED_ext_driver").write_text("cost", encoding="utf-8"); return
    rows = []
    for a in TX.job_list("cost"):
        m = json.loads((TX.RUNS_EXT / TX.T1.tag_of(*a) / "rep.json").read_text(encoding="utf-8"))
        rows.append(dict(scen=a[0], n=a[1], pi=a[2], sec=m["sec_total"], fits=m["n_fits"]))
    per = {}
    for r in rows:
        per.setdefault((r["scen"], r["n"], r["pi"]), []).append(r["sec"])
    cpu_s = sum(sum(v) / len(v) * TX.T1.REPS for v in per.values())
    done_s = sum(r["sec"] for r in rows)
    wall_h = (cpu_s - done_s) / JOBS / 3600 * 1.15
    cost = dict(实测重复=rows, 外推累计秒=cpu_s, 剩余预计墙钟小时=round(wall_h, 2), 并发=JOBS, 预算小时=BUDGET_H,
                说明="代表性重复包含完整选择器与认证程序；外推只用于判断是否在预算内，不据此减少重复或删去比较策略")
    (T1DIR / "_EXT_COST.json").write_text(json.dumps(cost, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"成本外推：剩余预计墙钟 {wall_h:.2f} 小时")
    if wall_h > BUDGET_H:
        (T1DIR / "_EXT_COST_STOP").write_text(json.dumps(cost, ensure_ascii=False), encoding="utf-8"); return
    rc = step(["run", str(JOBS), "--only", "all"])
    log(f"扩展全部结束 退出码 {rc}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("驱动异常 " + traceback.format_exc())
        (T1DIR / "_FAILED_ext_driver").write_text("exception", encoding="utf-8")
