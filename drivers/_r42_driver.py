# -*- coding: utf-8 -*-
r"""R42 正式批次驱动：入口校验冻结输入 → 吞吐测量 → 主实验（main+cert）→ 完整性 → 学习曲线 → 哨兵。

状态：_R42_STATUS.json（随时更新）；哨兵：_R42_MAIN_DONE、_R42_LC_DONE、_R42_DONE / _R42_FAILED。
只对报错的单元暂停（写 _FAILED），其余继续；不搜索替代种子、不删候选、不改网格。
"""
import sys, json, pathlib, time, math
R = pathlib.Path(__file__).resolve().parent
CODE = R.parent / "99_代码"
sys.path.insert(0, str(CODE))
import pgss_threads  # noqa: F401
import pandas as pd

EXPECTED_FP = "1598182eaa10e060"      # data v5 加载对象指纹（R383 FINAL chaincheck 记录值）
STATUS = R / "_R42_STATUS.json"
LOG = R / "driver.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%m-%d %H:%M:%S')}  {m}\n")


def status(**kw):
    s = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    s.update(kw, 更新=time.strftime("%m-%d %H:%M:%S")); STATUS.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def run_pool(units, W, phase):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import pgss_xgb_run as RU
    done = failed = skipped = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=W, max_tasks_per_child=1) as ex:
        futs = [ex.submit(RU._unit_entry, u) for u in units]
        for i, fu in enumerate(as_completed(futs), 1):
            args, res = fu.result()
            if res == "OK": done += 1
            elif res == "SKIP_DONE": skipped += 1
            else: failed += 1; log(f"{phase} 单元失败 {args[:7]} → {res}")
            if i % 5 == 0 or i == len(units):
                status(**{f"{phase}_完成": done, f"{phase}_跳过": skipped, f"{phase}_失败": failed,
                          f"{phase}_总数": len(units), f"{phase}_已用分钟": round((time.time() - t0) / 60, 1)})
    return done, skipped, failed


def main():
    for f in ("_R42_DONE", "_R42_FAILED", "_R42_MAIN_DONE", "_R42_LC_DONE"):
        (R / f).unlink(missing_ok=True)
    import pgss_dataver as DV, pgss_r38_bench as BN, psutil
    fp = DV.loaded_fingerprint_all(BN.load)
    status(数据指纹=fp, 预期=EXPECTED_FP)
    if fp != EXPECTED_FP:
        (R / "_R42_FAILED").write_text(f"冻结输入指纹不符：{fp} ≠ {EXPECTED_FP}", encoding="utf-8"); return
    log(f"入口校验通过：data v5 加载对象指纹 {fp}")
    pa = pd.read_parquet(R / "pathaudit" / "R42_PATHAUDIT_units.parquet")
    n_task = pd.read_excel(R.parent / "22_R41_Stage0" / "stage0a" / "R41_STAGE0A.xlsx", sheet_name="1_任务与轨")
    size = {(r.库, r.结局, r.表型): int(r.n) for _, r in n_task.iterrows()}
    mc = pa[pa.分支.isin(["main", "cert"])]
    ok_mc = mc[mc.可执行]
    paused = mc[~mc.可执行]
    for _, r in paused.iterrows():
        log(f"按路径审计暂停（不可执行）：{r.库}/{r.结局}/{r.表型} seed={r.seed} 折={r.外层折} {r.分支}：{r.get('失败信息')}")
    units = [(r.库, r.结局, r.表型, int(r.seed), int(r.外层折), r.分支, None, fp) for _, r in ok_mc.iterrows()]
    units.sort(key=lambda u: (-size[(u[0], u[1], u[2])], u[5] != "main"))
    status(主实验单元=len(units), 主实验暂停=len(paused))
    # 吞吐测量：最大的 8 个 main 单元，8 并发
    probe = [u for u in units if u[5] == "main"][:8]
    tp = time.time()
    run_pool(probe, 8, "吞吐测量")
    peaks = []
    import pgss_xgb_run as RU
    for u in probe:
        m = R / "runs" / RU.tag_of(*u[:7]) / "meta.json"
        if m.exists():
            peaks.append(json.loads(m.read_text(encoding="utf-8"))["peak_rss_mb"])
    total_mb = psutil.virtual_memory().total / 2**20
    W = int(max(4, min(15, math.floor(0.7 * total_mb / max(max(peaks, default=1500), 1)))))
    log(f"吞吐测量：8 单元 {round((time.time()-tp)/60,1)} 分钟，峰值内存 MB {peaks} ⇒ 并发 W={W}")
    status(吞吐测量峰值MB=peaks, 并发=W)
    rest = [u for u in units if u not in probe]
    d, s, f = run_pool(rest, W, "主实验")
    # 完整性：每个可执行单元都有 _DONE 或 _FAILED；预测无缺失
    miss, bad = [], []
    for u in units:
        od = R / "runs" / RU.tag_of(*u[:7])
        if not (od / "_DONE").exists():
            miss.append(od.name); continue
        P = pd.read_parquet(od / "preds_test.parquet")
        need = [c for c in P.columns if c not in ("unit_id", "y", "z")]
        if P[need].isna().any().any():
            bad.append(od.name)
    status(主实验_未完成=len(miss), 主实验_预测缺失=len(bad))
    (R / "_R42_MAIN_DONE").write_text(json.dumps(dict(未完成=miss[:50], 预测缺失=bad[:50]), ensure_ascii=False), encoding="utf-8")
    log(f"主实验结束：未完成 {len(miss)}，预测缺失 {len(bad)}")
    # 学习曲线：路径审计判为可评价的点
    lc = pa[(pa.分支 == "LC") & (pa.LC可评价 == 1)]
    lunits = [(r.库, r.结局, r.表型, int(r.seed), int(r.外层折), "lc", float(r.比例), fp) for _, r in lc.iterrows()]
    lunits.sort(key=lambda u: (-size[(u[0], u[1], u[2])] * u[6]))
    status(学习曲线单元=len(lunits), 学习曲线不可评价=int(((pa.分支 == "LC") & (pa.LC可评价 != 1)).sum()))
    run_pool(lunits, W, "学习曲线")
    (R / "_R42_LC_DONE").write_text("ok", encoding="utf-8")
    (R / "_R42_DONE").write_text(json.dumps(json.loads(STATUS.read_text(encoding="utf-8")), ensure_ascii=False), encoding="utf-8")
    log("全部结束")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        (R / "_R42_FAILED").write_text(f"driver 异常 {type(e).__name__}: {e}\n{traceback.format_exc()}", encoding="utf-8")
        raise
