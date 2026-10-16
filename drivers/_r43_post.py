# -*- coding: utf-8 -*-
r"""R43 收尾链（第48轮补充裁决后的版本）：等 T1 扩展结束（t1/_DONE_ext_all），或任一失败/超预算哨兵 →
原 T1 汇总 → 扩展汇总 → 本地完整复算（重抽矩阵只留本地）→ 图 F1–F6 → 报告 → 交付自检的变异对照 → 打包自检。
每步日志 post_<步骤>.log；成功哨兵 _post_<步骤>_ok；全部完成 _R43_POST_DONE，任一步失败 _R43_POST_FAILED（并停止）。
复制到 GPT交接_PGSS\43\ 不在此链内：交付前须人工看图、读报告。
"""
import sys, time, pathlib, subprocess, os
HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent / "99_代码"
PY = sys.executable
T1 = HERE / "t1"
STEPS = [("summary_main", ["pgss_r43_t1_summary.py", "main"]), ("summary_ext", ["pgss_r43_t1_summary.py", "ext"]),
         ("boot_verify", ["pgss_r43_boot_verify.py"]), ("figures", ["pgss_r43_figures.py"]),
         ("report", ["pgss_r43_report.py"]), ("package_tests", ["pgss_r43_package_tests.py"]),
         ("package", ["pgss_r43_package.py"])]
STOP = ("_FAILED_main", "_FAILED_driver", "_COST_STOP", "_FAILED_ext_wait", "_FAILED_ext_tests", "_FAILED_ext_sha",
        "_FAILED_ext_driver", "_FAILED_ext_cost", "_FAILED_ext_all", "_EXT_COST_STOP")


def main():
    while not (T1 / "_DONE_ext_all").exists():
        hit = [s for s in STOP if (T1 / s).exists()]
        if hit:
            (HERE / "_R43_POST_FAILED").write_text(f"T1 未正常完成（{hit}），收尾链不运行", encoding="utf-8"); return
        time.sleep(60)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for name, args in STEPS:
        r = subprocess.run([PY] + [str(CODE / args[0])] + args[1:], cwd=str(CODE), env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        (HERE / f"post_{name}.log").write_text(f"退出码 {r.returncode}\n{r.stdout}\n{r.stderr}", encoding="utf-8")
        if r.returncode != 0:
            (HERE / "_R43_POST_FAILED").write_text(name, encoding="utf-8"); return
        if name == "package_tests" and '"全部通过": true' not in r.stdout:
            (HERE / "_R43_POST_FAILED").write_text("package_tests 未全部通过", encoding="utf-8"); return
        (HERE / f"_post_{name}_ok").write_text(time.strftime("%m-%d %H:%M:%S"), encoding="utf-8")
    (HERE / "_R43_POST_DONE").write_text(time.strftime("%m-%d %H:%M:%S"), encoding="utf-8")


if __name__ == "__main__":
    main()
