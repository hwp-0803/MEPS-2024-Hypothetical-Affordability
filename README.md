# Hypothetical Medical Bill Affordability and Experienced Financial Hardship by Race and Ethnicity

Analysis code for an exploratory study of the 2024 Medical Expenditure Panel Survey (MEPS HC-256):
whether stated inability to pay an unexpected $500 medical bill (EQPAYB53 = 7) is associated with
family medical-bill payment problems (PROBPY42), family medical debt (MEDDEBT42), and health-related
financial sacrifice (EQSNON53) to the same extent among non-Hispanic White, Hispanic, and
non-Hispanic Black adults (analytic n = 10,360).

Author: Hyowon Park, Gachon University

The raw data are not included. Download the public-use HC-256 Stata file from AHRQ (link below),
save it as `input.dta`, and run the scripts as shown. Each script's log from the run used for the
manuscript is included for comparison.

Exhibit numbering: the manuscript has Tables 1-3 and Figure 1 (forest plot) in the main text and a
supplementary appendix. Script outputs use earlier labels: "Fig. 1" = Appendix Figure A1 (sample flow),
"Table 4" = Appendix Table A1, "Table S1"/"Table S2" = Appendix Tables A2/A3, and the discrimination
analysis = Appendix Text A1.

---

These scripts recompute every estimate in the manuscript from the public MEPS HC-256 Stata file
(`h256.dta`, here called `input.dta`), available from AHRQ:
https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-256

| Script | Reproduces | Log of the run used for the manuscript |
|---|---|---|
| `01_survey_models.R` | Sample flow (Appendix Figure A1), Table 1 key rows, Table 2, Table 3, Appendix Table A1, ratio-of-odds-ratio contrasts | `01_survey_models.log` |
| `03_survey_models_python_check.py` | Independent Python implementation of the R models (Table 2, Table 3 intervals, Appendix Table A1), for environments without R; compares all 150 printed values with `01_survey_models.log` | `03_survey_models_python_check.log` |
| `02_auc_bootstrap.py` | Supplementary discrimination analysis (Appendix Text A1) (item-alone AUC, cross-validated AUC, Rao–Wu bootstrap intervals, within-group comparison pipeline) | `02_auc_bootstrap.log` |
| `04_sensitivity_supplementary.R` | Appendix Table A2 (additional adjustment for total health care charges and highest degree) and Appendix Table A3 (health care use and cost-related delayed or forgone care, adjusted predicted percentages) | `04_sensitivity_supplementary.log` |

Run:

    Rscript 01_survey_models.R input.dta
    python 02_auc_bootstrap.py input.dta
    python 03_survey_models_python_check.py input.dta 01_survey_models.log   # no R needed
    Rscript 04_sensitivity_supplementary.R input.dta

`03_survey_models_python_check.py` needs pandas, NumPy, and SciPy. It uses the same estimators as R survey
(Taylor-linearized PSU variance; design df counted over PSUs containing domain observations; svyciprop logit intervals),
so agreement with the R log is an implementation cross-check, not a substitute for running R.

Software used for the logged runs: R 4.5.3 with survey 4.5 and haven 2.5.5; Python 3.13 with pandas and NumPy.
The bootstrap uses seed 20261111, so `02_auc_bootstrap.py` reproduces the logged intervals exactly.

Notes
- Survey design: weights ESAQWT24F, strata VARSTR, PSUs VARPSU nested within strata; the analytic sample is
  taken with `subset()` from the full ESAQ design so that variance estimation uses all strata and PSUs.
- Interaction tests are 2-df design-based Wald F tests of the Hispanic-by-exposure and Black-by-exposure terms,
  F = W/2 with denominator df equal to the design df (159).
- `04_sensitivity_supplementary.R` was added after the main analyses to examine alternative explanations (exploratory).
  Its main-model rows reproduce Table 2 exactly; family total charges sum TOTTCH24 over all persons sharing DUID and FAMID42.
- The earlier Python AUC pipeline fitted models within each group; the Hispanic covariate-only AUC is unstable
  under that approach (0.46 here). The manuscript's main AUC results use pooled models; both are printed.
