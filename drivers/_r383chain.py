# -*- coding: utf-8 -*-
r"""R383 全链。GPT 第 42 轮 §7 规定的执行顺序（报告在 chaincheck 之前）：

    数据冻结 → benchmark → inference → 初版工作簿 → **报告生成**
    → FINAL chaincheck（C8 必须读真实报告）→ provenance → 打包

硬闸门（任一不过立即停）
    · v5 可复现 ≠ True
    · QC 的窗口两侧未闭合
    · 缺实测删除计数（不许缺省 0）
    · 前置检查 24 个任务不全成立
    · FINAL chaincheck 不过 ⇒ 不出 provenance、不打包
"""
import json, os, pathlib, subprocess, sys, time

R = pathlib.Path(r"D:\Research\2_课题\PGSS_表型引导选择性专业化")
O = R / "19_R38"; CODE = R / "99_代码"
PY = r"D:\Python\python.exe"
HO = pathlib.Path(r"D:\Research\3_对话与交接\GPT交接_PGSS\38.3")
PINS = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
            PGSS_HANDOFF=str(HO), PGSS_REPORT="R383_FINAL_LR_REPORT.md")
ENV = {**os.environ, **PINS}
JOBS = {"eicu": "5", "mimic": "6"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sh(args, tag, hard=True, must=None):
    log("运行 " + " ".join(args))
    lf = O / f"r383_{tag}.log"
    with open(lf, "w", encoding="utf-8", errors="replace", buffering=1) as fh:
        r = subprocess.run([PY] + args, cwd=str(CODE), env=ENV, stdout=fh,
                           stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", errors="replace")
    body = lf.read_text(encoding="utf-8", errors="replace")
    for l in [x for x in body.splitlines() if x.strip()][-10:]:
        print("    " + l, flush=True)
    if hard and r.returncode != 0:
        raise RuntimeError(f"{tag} 退出码 {r.returncode}")
    if hard and must and must not in body:
        raise RuntimeError(f"{tag} 缺标记 {must!r}")
    return r.returncode


def wait_file(p, what, timeout_s=14400):
    t0 = time.time(); p = pathlib.Path(p)
    while not p.exists():
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"{what} 超时")
        time.sleep(30)
    log(f"{what} 就绪")


if __name__ == "__main__":
    for s in ("_R383_DONE", "_R383_FAILED"):
        f = O / s
        if f.exists(): f.unlink()
    try:
        # 0 等 v5 构建 + 可复现性验证
        wait_file(O / "data_v5" / "DATA_VERSION.json", "v5 构建")
        log("启动 v5 可复现性验证（清暂存表重扫）")
        sh(["pgss_r383_build_v5.py", "--verify"], "v5verify")
        rep = json.loads((O / "R383_V5_REPRODUCIBILITY.json").read_text(encoding="utf-8"))
        if rep.get("可复现") is not True:
            raise RuntimeError(f"v5 不可复现：{rep.get('明细')}")
        log(f"v5 可复现 = True（比对 {rep.get('文件数')} 个受影响文件）")

        # 1 QC（窗口证明 + 新口径改变计数；脚本内不过会非零退出）
        sh(["pgss_r383_v5_qc.py"], "qc")

        # 2 切活动数据版本到 v5
        dv = CODE / "pgss_dataver.py"
        s = dv.read_text(encoding="utf-8")
        if 'ACTIVE = "v5"' not in s:
            import re
            s = re.sub(r'ACTIVE = "v\d+"', 'ACTIVE = "v5"', s, count=1)
            dv.write_text(s, encoding="utf-8")
        log("活动数据版本已切到 v5")

        # 3 前置检查
        sh(["pgss_r381_precheck.py"], "precheck", must="24 个任务中成立 24 个")

        # 4 72 单元全量重跑（缓存键含 data_version ⇒ v5 使全部单元失效）
        for db in ("eicu", "mimic"):
            log(f"bench {db} 开始（并行 {JOBS[db]}）")
            sh(["pgss_r38_bench.py", "--db", db, "--jobs", JOBS[db]], f"bench_{db}")

        # 5 验收（全部在 v5 生产加载器上）
        sh(["pgss_core_tests.py"], "coretests")
        sh(["pgss_r36_tests.py"], "r36tests")
        sh(["pgss_r381_tests.py"], "r381tests")
        sh(["pgss_r36_e2e.py"], "e2e")

        # 6 推断
        sh(["pgss_r381_joint_inference.py", "--B", "2000"], "joint")

        # 7 初版工作簿 + **报告生成**（GPT §7：报告必须先于 chaincheck）
        sh(["pgss_r383_final.py"], "final_pre", hard=False)

        # 8 FINAL chaincheck —— C8 现在读真实报告，缺失即失败
        if not (HO / "R383_FINAL_LR_REPORT.md").exists():
            raise RuntimeError("报告未生成，不得进入 FINAL chaincheck")
        sh(["pgss_r381_chaincheck.py"], "chaincheck_final")

        # 9 provenance + 打包（只有 chaincheck 真通过才走到这里）
        sh(["pgss_r383_final.py"], "final", hard=False)
        sh(["pgss_selfaudit.py", "--pkg", str(HO),
            "--rows", str(O / "R383_ALL_ROWS.parquet")], "selfaudit", hard=False)

        (O / "_R383_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        log("=== R383 全链完成 ===")
    except Exception as e:
        import traceback
        traceback.print_exc()
        (O / "_R383_FAILED").write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
        sys.exit(1)
