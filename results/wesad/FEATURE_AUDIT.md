# WESAD FEATURE AUDIT

This document audits the 64 derived ECG/HRV features to ensure no target leakage.

### `MEAN_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 19776.005727252515
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `MEDIAN_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 21323.760355520924
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SDRR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 2067.8074961498814
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `RMSSD`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 53.676492812222314
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SDSD`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 53.667309974638044
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SDRR_RMSSD`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 5.211155597873115
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 114.98168435267414
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `pNN25`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 182.32183933404204
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `pNN50`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 7.539119759932257
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SD1`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 26.90107623100575
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SD2`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 4125.861067865951
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `KURT`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.5651132010827361
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SKEW`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.25820645069288206
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `MEAN_REL_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1.756902780983705e-07
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `MEDIAN_REL_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 2.1414702833862145e-06
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SDRR_REL_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 4.422604665691935e-05
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `RMSSD_REL_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1.1692590465942704e-05
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SDSD_REL_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1.1692260377199154e-05
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SDRR_RMSSD_REL_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.23098524438177848
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `KURT_REL_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.5651132010827361
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SKEW_REL_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.25820645069288206
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `VLF`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 8130939.624705948
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `VLF_PCT`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 320.51989811629596
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `LF`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 957539.9166565811
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `LF_PCT`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 310.99985996483923
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `LF_NU`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 6.771653424959064
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HF`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 448.9062668617572
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HF_PCT`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.8969537817955774
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HF_NU`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 6.771653424959066
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `TP`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 11241896.125459967
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `LF_HF`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 685305.1998713482
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HF_LF`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.0009268648815934971
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `MEAN_RR_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.024126415755212852
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `MEAN_RR_SQRT`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 5.4199232509021575
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `TP_SQRT`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 524.1833949032017
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `MEDIAN_REL_RR_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 2.1405308040938364e-06
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `RMSSD_REL_RR_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1.1502045284489318e-05
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SDSD_REL_RR_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1.1501724854098372e-05
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `VLF_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.5978904648931022
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `LF_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1.0804771224125946
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HF_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1.3930036547798628
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `TP_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.5344953740249418
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `LF_HF_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 2.567294436461785
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `RMSSD_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.20657647583733463
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SDRR_RMSSD_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.08587552178234206
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `pNN25_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1.6065351940901367
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `pNN50_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.6365232771608155
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SD1_LOG`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.19499235759365752
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `KURT_YEO_JONSON`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.4363615430865219
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SKEW_YEO_JONSON`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.2713887288236906
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `MEAN_REL_RR_YEO_JONSON`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1.7603884277745672e-07
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SKEW_REL_RR_YEO_JONSON`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.2713887288236906
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `LF_BOXCOX`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 30498.59523710413
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HF_BOXCOX`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 2.6911145797773215
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SD1_BOXCOX`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 17446.365067643324
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `KURT_SQUARE`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 3.2409002603295654
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HR_SQRT`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.4075211552318561
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `MEAN_RR_MEAN_MEAN_REL_RR`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 4.836139703017211e+17
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `SD2_LF`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.052251957504748064
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HR_LF`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.0274767805430256
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HR_HF`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 1704.0375590542728
- **Calculated independently inside window:** Yes (standard HRV calculation)

### `HF_VLF`
- **Data Type:** float64
- **Physiological:** Yes
- **Could contain label information:** No
- **Missing values:** 0
- **Variance:** 0.00041242301371919806
- **Calculated independently inside window:** Yes (standard HRV calculation)
