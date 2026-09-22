# WESAD Pilot Final Validation

## Model Comparison
```json
{
  "LR": {
    "Default 0.5": {
      "Accuracy": 0.9506744677393142,
      "Balanced Accuracy": 0.9043111515527718,
      "Precision": 0.9394725464764375,
      "Recall": 0.8231060606060606,
      "F1": 0.877448011306279,
      "ROC-AUC": 0.9912948933782267,
      "Threshold": 0.5
    },
    "Validation Threshold": {
      "Accuracy": 0.9491305054444986,
      "Balanced Accuracy": 0.8903876991516656,
      "Precision": 0.9696828358208955,
      "Recall": 0.7875,
      "F1": 0.8691471571906354,
      "ROC-AUC": 0.9912948933782267,
      "Threshold": 0.6869585865186543
    }
  },
  "RF": {
    "Default 0.5": {
      "Accuracy": 0.9052494718023728,
      "Balanced Accuracy": 0.9182095630419653,
      "Precision": 0.7109330280480824,
      "Recall": 0.9409090909090909,
      "F1": 0.8099119660906423,
      "ROC-AUC": 0.9490306149640414,
      "Threshold": 0.5
    },
    "Validation Threshold": {
      "Accuracy": 0.843653502356574,
      "Balanced Accuracy": 0.8884989325282622,
      "Precision": 0.5815489749430524,
      "Recall": 0.9670454545454545,
      "F1": 0.7263157894736842,
      "ROC-AUC": 0.9490306149640414,
      "Threshold": 0.16
    }
  },
  "XGB": {
    "Default 0.5": {
      "Accuracy": 0.8773768893222819,
      "Balanced Accuracy": 0.8895913276150706,
      "Precision": 0.6537102473498233,
      "Recall": 0.9109848484848485,
      "F1": 0.761196391834151,
      "ROC-AUC": 0.8946113250757105,
      "Threshold": 0.5
    },
    "Validation Threshold": {
      "Accuracy": 0.8588493417844953,
      "Balanced Accuracy": 0.896520018621974,
      "Precision": 0.6080402010050251,
      "Recall": 0.9625,
      "F1": 0.7452705675318961,
      "ROC-AUC": 0.8946113250757105,
      "Threshold": 0.0005284844082780182
    }
  },
  "DL": {
    "Default 0.5": {
      "Accuracy": 0.9691207541036893,
      "Balanced Accuracy": 0.9572155446457122,
      "Precision": 0.9210134128166915,
      "Recall": 0.9363636363636364,
      "F1": 0.9286250939143501,
      "ROC-AUC": 0.9956400990036931,
      "Threshold": 0.5
    },
    "Validation Threshold": {
      "Accuracy": 0.9687144482366326,
      "Balanced Accuracy": 0.9610868931399658,
      "Precision": 0.9101491451436886,
      "Recall": 0.9477272727272728,
      "F1": 0.9285581740582668,
      "ROC-AUC": 0.9956400990036931,
      "Threshold": 0.22035616636276245
    }
  },
  "Permutation Test (DL)": {
    "Default 0.5": {
      "Accuracy": 0.7854705021940517,
      "Balanced Accuracy": 0.5,
      "Precision": 0.0,
      "Recall": 0.0,
      "F1": 0.0,
      "ROC-AUC": 0.12207707114597247,
      "Threshold": 0.5
    },
    "Validation Threshold": {
      "Accuracy": 0.21461075897935966,
      "Balanced Accuracy": 0.500051727705359,
      "Precision": 0.21454693214140594,
      "Recall": 1.0,
      "F1": 0.35329541652726665,
      "ROC-AUC": 0.12207707114597247,
      "Threshold": 0.060858987271785736
    }
  },
  "Ablation (Top 10) (DL)": {
    "Top 10 Features": [
      "HF_VLF",
      "HF_PCT",
      "SDSD",
      "SD1_BOXCOX",
      "MEAN_RR_SQRT",
      "MEAN_RR_LOG",
      "MEAN_RR",
      "HR",
      "MEDIAN_RR",
      "HR_SQRT"
    ],
    "Default 0.5": {
      "Accuracy": 0.9538436535023566,
      "Balanced Accuracy": 0.9520337217613755,
      "Precision": 0.8526208304969367,
      "Recall": 0.9488636363636364,
      "F1": 0.8981713875941197,
      "ROC-AUC": 0.9908104947676643,
      "Threshold": 0.5
    },
    "Validation Threshold": {
      "Accuracy": 0.9525434747277751,
      "Balanced Accuracy": 0.9582270564114139,
      "Precision": 0.8363874345549738,
      "Recall": 0.9681818181818181,
      "F1": 0.8974719101123596,
      "ROC-AUC": 0.9908104947676643,
      "Threshold": 0.1593589186668396
    }
  }
}
```

## Conclusion
**CLEARED FOR LOSO**