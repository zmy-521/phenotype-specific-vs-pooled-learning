# Phenotype-specific versus phenotype-aware pooled learning — analysis code

Code for the study *Phenotype-specific versus phenotype-aware pooled learning: a multicohort benchmark and simulation study*
(citation to be added on publication).

The scripts compare frozen pooled and phenotype-specific XGBoost development programs in three ICU databases (MIMIC-IV 3.1, eICU-CRD 2.0, NWICU 0.1.0),
in NHANES 1999–2008 with public-use linked mortality, and in known-truth simulations.

## What is and is not in this repository

- `code/` — 105 analysis scripts plus `external/nhanes.py`, published **byte-for-byte as run** (SHA-256 in `MANIFEST.csv`).
  The only edited file is `external/nhanes.py`, where one example path in a docstring was replaced; its code is unchanged (checked by comparing syntax trees).
- `drivers/` — orchestration scripts that launched the main runs; they still point to the original project folders.
- `t2_records/` — per-replicate records of the T2 known-truth simulation (synthetic data only).
- **Not included**: any MIMIC-IV, eICU-CRD or NWICU data or derived participant-level files, out-of-fold predictions, NHANES extracts, or manuscript files.
  MIMIC-IV, eICU-CRD and NWICU are credentialed-access PhysioNet databases; obtain access from PhysioNet and follow its data use agreement.
  NHANES and the NCHS public-use linked mortality files are publicly available from NCHS.

## Pipeline

| Stage | Main scripts | What it produces |
|---|---|---|
| Cohorts and analysis data (MIMIC-IV, eICU-CRD) | pgss_r35_cohort.py, pgss_r35_extract.py, pgss_r35_qc.py, pgss_r383_build_v5.py, pgss_dataver.py | Eligible ICU stays at the 24-h landmark, predictors, outcomes; frozen analysis data (read from a local PostgreSQL copy of the databases) |
| Model-development engine | pgss_xgb_engine.py, pgss_xgb_run.py, pgss_threads.py, pgss_core.py, pgss_sharing.py | Frozen XGBoost pooled (P*) and phenotype-specific (HTFS, HGFS) learning programs, nested model selection, calibration |
| ICU benchmark and inference | drivers/_r41_driver.py, drivers/_r42_driver.py, pgss_r42_inference.py, pgss_r43_*.py | 36 outcome–phenotype tasks, clustered resampling of fixed out-of-fold predictions, max-T simultaneous judgments |
| Known-truth simulations T1/T2 | pgss_xgb_sim0.py, pgss_r43_t1*.py, pgss_r46_t2.py, pgss_r46_t2_summary.py, pgss_r47_example.py | Eight T2 conditions × 200 redevelopment replicates; selector comparison (FCT, CV rules, MIX) |
| NHANES cross-domain replication | pgss_r44_nhanes_*.py, pgss_r45_target*.py, pgss_r46_outcome_verify.py, code/external/nhanes.py | 10-year mortality tasks, objective-alignment follow-up |
| NWICU independent-program replication | pgss_r45_nwicu_*.py, pgss_r46_rerun.py, pgss_r46_readonly.py | Program redevelopment in NWICU 0.1.0, outcome-status correction |
| Plasmode, learning curves, certification | pgss_r45_plasmode*.py, pgss_r43_readonly.py, pgss_r46_inference.py | Exploratory empirical-covariate simulation; training-fraction curves; fit–certify–test |
| Post hoc NHANES creatinine standardization | pgss_r52_*.py, drivers/_r52_driver.py | Sensitivity redevelopment with NCHS-recommended 2005–2006 standardization |
| Numbers, figures and supplementary tables | pgss_r47_numbers.py, pgss_r47_figures.py, pgss_r48_*.py, pgss_r49_*.py, pgss_r51_*.py, pgss_r54_supp_*.py, pgss_r56_*.py | Frozen number pack, Figures 1–6, Table 1 sources, Supplementary tables, predictor completeness |

Script names keep the internal round numbers used during the study (for example `r42` = main ICU experiment). Comments and console messages are in Chinese.

## Environment

Python 3.13.5; exact package versions in `requirements.txt`. Runs used Windows 11 on an Intel i7-12700F; bit-identical replay of the
frozen simulation records depends on the same xgboost/numpy versions and platform.

## Quick start without restricted data

```
pip install -r requirements.txt
cd code
python pgss_r47_example.py quick --out ../example_out        # about 2 minutes; demonstration only, numbers are not comparable with the paper
python pgss_r47_example.py reproduce --out ../example_out    # re-runs one frozen T2 replicate and compares it with t2_records/ (5–8 minutes)
```

## Re-running the clinical analyses

Requires your own authorized copies of the source databases. Builders read a local PostgreSQL server; set `PGHOST`, `PGPORT`, `PGUSER` and pass the
password only through the `PGPASSWORD` environment variable. The frozen scripts keep the original absolute paths; see `PATHS.md` for every path
and how to adapt it.

## Development note

Parts of this code were written with the assistance of AI coding tools under the authors' direction; see the article's Methods for the AI-use statement.

## License

To be added by the authors before public release.
