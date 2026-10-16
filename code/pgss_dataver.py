# -*- coding: utf-8 -*-
r"""活动数据版本的**单一真相源**。

为什么要这个模块
    数据目录（`F = .../data_vX`）与缓存键里的 `data_version="vX"` 是两个地方写的同一件事。
    逐处改字符串迟早会出现「目录指向 v3、键里写 v2」的不一致 ——
    那不会报错，只会让 provenance 说谎。放在一处，谁都不许再写字面值。

版本沿革
    v1  R35 冻结输入。eICU 化验只用 labresultoffset 作时间资格（**错**：
        采样在 24h 内但结果修订录入在 24h 后的化验，在 landmark 时其实不可获得）。
    v2  R38 §A1 纠错：eICU 化验加 `labresultrevisedoffset IS NULL OR < 1440`。
        15 个特征受影响（改变率 1.99%–3.18%）。MIMIC 未改。
    v3  R38.1 §2 纠错：九处 `DISTINCT ON ... ORDER BY <time> DESC` **无确定性并列打破规则**，
        同一变量由多个 itemid/多张表映射 ⇒ 最晚时点并列时取哪个值是任意的
        ⇒ **冻结输入无法按文档重新导出**（实测 MIMIC 体征 14% 不一致，最大差 114–146；
        收缩压 24.7% 的就诊在最晚时点取值不唯一，并列跨度最大 202 mmHg）。
        v3 规则：取最晚时点（同 v2），该时点多行则取**算术平均**（在 pandas 内稳定排序后计算，
        不用 SQL avg——并行聚合求和顺序可能变化）。
"""
import pathlib

ACTIVE = "v5"
ROOT = pathlib.Path(__file__).parent.parent
ACTIVE_DIRNAME = f"data_{ACTIVE}"
DIR = ROOT / "19_R38" / ACTIVE_DIRNAME


def manifest_path():
    return DIR / "DATA_VERSION.json"


def fingerprint():
    """**不要用它当验收记录的数据戳。**

    R382 实测：这个值取自清单的 `eicu_X`，而 v3 与 v4 的 `eicu_X` 完全相同
    （v4 只改 MIMIC 体征）—— 它对整个 v4 改动失明。闸门 4 曾用它做基准，
    变异对照证明：一份 v3 时代的验收记录能被原样放行。
    验收记录与闸门基准一律用 `loaded_fingerprint_all(loader)`（按实际加载的对象算）。
    此函数仅保留给"清单里写了什么"这类溯源用途。
    """
    import hashlib, json
    p = manifest_path()
    if not p.exists():
        return None
    try:
        return (json.loads(p.read_text(encoding="utf-8")).get(f"{ACTIVE}_SHA") or {}).get("eicu_X")
    except Exception:
        return None


def loaded_fingerprint(X, P, OK, meta):
    """从**实际加载的对象**算数据指纹，而不是从清单里读版本号。

    为什么必须这样
        R38.1 的 E2E 把 `DATA_VERSION.json` 里的版本 SHA 当作自己的数据指纹，
        但它实际 load 的是 `pgss_r36_bench`（目录 16_R35_FIX = v1）。
        于是戳标着 v3、跑的却是 v1，而闸门 4 恰恰靠这个戳判断"证据是否出自当前输入"。
        **戳必须由被加载的内容生成，否则它只是一句声明。**
    """
    import hashlib
    import pandas as _pd
    h = hashlib.sha256()
    for name, df in (("X", X), ("PHENO", P), ("PHENOOK", OK), ("META", meta)):
        h.update(name.encode("utf-8"))
        h.update("|".join(map(str, df.columns)).encode("utf-8"))
        h.update(_pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def loaded_fingerprint_all(loader, dbs=("mimic", "eicu")):
    """两个库都算，合成一个指纹。单个 eicu_X 的 SHA 不能代表全部输入。"""
    import hashlib
    h = hashlib.sha256()
    for db in dbs:
        X, P, OK, meta = loader(db)
        h.update(db.encode("utf-8"))
        h.update(loaded_fingerprint(X, P, OK, meta).encode("utf-8"))
    return h.hexdigest()[:16]
