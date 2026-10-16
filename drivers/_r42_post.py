# -*- coding: utf-8 -*-
r"""R42 收尾链：等 _R42_DONE → 推断（B=2000）→ 图 → 报告（md/xlsx 同源）→ 打包与自检。
逐步写日志 post_<步骤>.log；全部成功写 _R42_POST_DONE，任一步失败写 _R42_POST_FAILED（含步骤名）后停止。
驱动若写 _R42_FAILED，本链不执行任何步骤，直接写 _R42_POST_FAILED。
交付到 GPT交接 目录与看图核对仍由人工完成。
"""
import sys, time, pathlib, subprocess, json

S = pathlib.Path(__file__).resolve().parent
CODE = S.parent / "99_代码"
PY = sys.executable
STEPS = [("inference", ["pgss_r42_inference.py"]), ("figures", ["pgss_r42_figures.py"]),
         ("report", ["pgss_r42_report.py"]), ("package", ["pgss_r42_package.py"])]


def mark(name, **kw):
    (S / name).write_text(json.dumps(dict(时间=time.strftime("%m-%d %H:%M:%S"), **kw), ensure_ascii=False), encoding="utf-8")


def main():
    while not (S / "_R42_DONE").exists():
        if (S / "_R42_FAILED").exists():
            mark("_R42_POST_FAILED", 步骤="driver", 信息="驱动写了 _R42_FAILED，收尾链未执行"); return
        time.sleep(60)
    for name, args in STEPS:
        t0 = time.time()
        with open(S / f"post_{name}.log", "w", encoding="utf-8") as lg:
            r = subprocess.run([PY, *args], cwd=CODE, stdout=lg, stderr=subprocess.STDOUT,
                               env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode != 0:
            mark("_R42_POST_FAILED", 步骤=name, 返回码=r.returncode, 秒=round(time.time() - t0)); return
        mark(f"_post_{name}_ok", 秒=round(time.time() - t0))
    mark("_R42_POST_DONE")


if __name__ == "__main__":
    main()
