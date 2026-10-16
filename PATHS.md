# Absolute paths in the code

The analysis scripts are published byte-for-byte as they were run, so they keep the original Windows paths of the authors' workstation.
Changing them would break the SHA-256 fingerprints recorded during the study (see `MANIFEST.csv`).
To re-run a script, edit the constant at the listed line, or recreate the same layout:

- **Project root** = the parent folder of `code/` (most scripts use `pathlib.Path(__file__).resolve().parent.parent`).
- **Output folders** such as `19_R38/` (frozen MIMIC-IV/eICU-CRD analysis data), `23_R42_正式主实验/` (main experiment) are created under the project root.
- **Restricted data**: `E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data` is the local NWICU 0.1.0 download; the MIMIC-IV/eICU-CRD builders read a local PostgreSQL
  server (`PGHOST`/`PGPORT`/`PGUSER`, password only via the `PGPASSWORD` environment variable).
- **NHANES helper**: `code/external/nhanes.py`; scripts that add `D:/NAH公共数据库/NHANES/2_代码` to `sys.path` only need `code/external` on the path instead.
- Paths under `D:/Research/3_对话与交接/...` are only used by packaging scripts to compare files with earlier deliveries; they do not affect any result.

| File | Line | Path |
|---|---:|---|
| `code/external/nhanes.py` | 4 | `E:\\公共数据库\\NHANES\\2_代码` |
| `code/external/nhanes.py` | 13 | `C:/path/to/output` |
| `code/pgss_core.py` | 7 | `C:\\pgss_tmp` |
| `code/pgss_core.py` | 8 | `C:\\pgss_tmp` |
| `code/pgss_core.py` | 9 | `C:\\Users\\<旧用户>\\Desktop\\PGSS-...` |
| `code/pgss_r43_report.py` | 23 | `D:\\Research\\1_数据\\MIMIC` |
| `code/pgss_r43_report.py` | 41 | `D:\\NAH公共数据库\\NHANES；PGSS` |
| `code/pgss_r44_nhanes_cohort.py` | 21 | `D:/NAH公共数据库/NHANES/2_代码` |
| `code/pgss_r45_nwicu_audit.py` | 22 | `E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data` |
| `code/pgss_r45_nwicu_build.py` | 40 | `E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data` |
| `code/pgss_r45_nwicu_stage.py` | 17 | `E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data` |
| `code/pgss_r45_package.py` | 20 | `D:/NAH公共数据库/NHANES/2_代码/nhanes.py` |
| `code/pgss_r45_r44supp.py` | 14 | `D:/NAH公共数据库/NHANES/2_代码` |
| `code/pgss_r45_r44supp.py` | 20 | `D:/NAH公共数据库/NHANES/2_代码/nhanes.py` |
| `code/pgss_r46_outcome_verify.py` | 13 | `D:/NAH公共数据库/NHANES/2_代码` |
| `code/pgss_r46_outcome_verify.py` | 20 | `E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data` |
| `code/pgss_r46_package.py` | 24 | `D:/NAH公共数据库/NHANES/2_代码/nhanes.py` |
| `code/pgss_r46_package_tests.py` | 43 | `e:\n` |
| `code/pgss_r46_readonly.py` | 23 | `E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data` |
| `code/pgss_r47_numbers.py` | 23 | `D:/Research/3_对话与交接/GPT交接_PGSS` |
| `code/pgss_r47_package_tests.py` | 42 | `e:\n` |
| `code/pgss_r48_sources.py` | 22 | `D:/Research/3_对话与交接/GPT交接_PGSS` |
| `code/pgss_r48_sources.py` | 470 | `E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data` |
| `code/pgss_r48_sources.py` | 471 | `D:/NAH公共数据库/NHANES/2_代码` |
| `code/pgss_r48_sources.py` | 472 | `D:/NAH公共数据库/NHANES/2_代码/nhanes.py` |
| `code/pgss_r48_sources.py` | 473 | `D:/Research/3_对话与交接/GPT交接_PGSS` |
| `code/pgss_r49_aux.py` | 19 | `D:/Research/3_对话与交接/GPT交接_PGSS` |
| `code/pgss_r49_cohort.py` | 20 | `D:/Research/3_对话与交接/GPT交接_PGSS` |
| `code/pgss_r51_baseline.py` | 24 | `E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data/nw_hosp` |
| `code/pgss_r51_baseline.py` | 43 | `D:/NAH公共数据库/NHANES/1_数据/索引/变量索引.csv` |
| `code/pgss_r51_baseline.py` | 44 | `D:/NAH公共数据库/NHANES/3_存档/官方codebook/2005-2006/BIOPRO_D.htm` |
| `code/pgss_r51_baseline.py` | 96 | `D:/NAH公共数据库/NHANES/2_代码` |
| `code/pgss_r51_package.py` | 9 | `D:\Research\3_对话与交接\GPT交接_PGSS\51\2_Claude_R51回复（待上传）` |
| `code/pgss_r52_package.py` | 9 | `D:\Research\3_对话与交接\GPT交接_PGSS\52\2_Claude_R52回复（待上传）` |
| `code/pgss_r52_sens.py` | 37 | `D:/NAH公共数据库/NHANES/3_存档/官方codebook/2005-2006/BIOPRO_D.htm` |
| `code/pgss_r52_sens.py` | 91 | `D:/NAH公共数据库/NHANES/2_代码` |
| `code/pgss_r52_tests.py` | 12 | `D:/NAH公共数据库/NHANES/1_数据/分文件/2005-2006/BIOPRO_D.parquet` |
| `code/pgss_r52_v04b_audit.py` | 12 | `D:\Research\3_对话与交接\GPT交接_PGSS\52\3_GPT_R52回应（已收到）` |
| `code/pgss_r53_audit.py` | 13 | `D:\Research\3_对话与交接\GPT交接_PGSS` |
| `code/pgss_r54_supp_build.py` | 14 | `D:\Research\3_对话与交接\GPT交接_PGSS\53\3_GPT_R54交付包（已收到）` |
| `code/pgss_r54_supp_build.py` | 15 | `D:\Research\3_对话与交接\GPT交接_PGSS` |
| `code/pgss_r54_supp_package.py` | 8 | `D:\Research\3_对话与交接\GPT交接_PGSS\53\4_Claude_R54补充材料v0.1（待上传）` |
| `code/pgss_r56_missingness.py` | 24 | `D:\Research\3_对话与交接\GPT交接_PGSS\53\5_GPT_R55交付包（已收到）` |
| `code/pgss_r56_package.py` | 8 | `D:\Research\3_对话与交接\GPT交接_PGSS\53\6_Claude_R56缺失率（待上传）` |
| `code/pgss_r56_tests.py` | 22 | `D:\Research\3_对话与交接\GPT交接_PGSS\53\5_GPT_R55交付包（已收到）\02_SUPPLEMENTARY\PGSS_Supplementary_v0.2_numbered_review.docx` |
| `drivers/_r383chain.py` | 16 | `D:\Research\2_课题\PGSS_表型引导选择性专业化` |
| `drivers/_r383chain.py` | 18 | `D:\Python\python.exe` |
| `drivers/_r383chain.py` | 19 | `D:\Research\3_对话与交接\GPT交接_PGSS\38.3` |
| `drivers/_r41_driver.py` | 10 | `D:\Python\python.exe` |
| `drivers/_r52_driver.py` | 5 | `D:\Python\python.exe` |
| `drivers/_r52_driver.py` | 6 | `D:\Research\2_课题\PGSS_表型引导选择性专业化` |
| `drivers/R47_收尾链.sh` | 3 | `D:/Research/2_课题/PGSS_表型引导选择性专业化/99_代码` |
| `drivers/R48_收尾链.sh` | 3 | `D:/Research/2_课题/PGSS_表型引导选择性专业化/99_代码` |
| `drivers/R49_收尾链.sh` | 3 | `D:/Research/2_课题/PGSS_表型引导选择性专业化/99_代码` |
