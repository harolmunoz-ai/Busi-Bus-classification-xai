# BUSI-BUS: Breast Ultrasound Lesion Classification with XAI (Pointing Game)

Reproducible pipeline for breast ultrasound image classification (benign / malignant / normal)
on the **BUSI-BUS** dataset, including:

- Architecture comparison (`mixer_b16_224`, `resnet50`, `efficientnet_b0`) via `timm`.
- Hyperparameter optimization with Optuna (including an optional explainability term
  based on the *pointing game* with Integrated Gradients).
- Multi-seed evaluation (>=5 runs), bootstrap 95% confidence intervals, and paired
  statistical tests (exact McNemar, DeLong, Holm-Bonferroni correction).
- Explainability (XAI) analysis: Integrated Gradients, pointing game, Dice score
  between attribution and lesion mask.

## Structure

```
.
├── busi_bus_pipeline.ipynb   # Main pipeline (Jupyter notebook, as run on Kaggle)
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Requirements

See `requirements.txt`. Designed to run on Kaggle (with GPU) or locally with a CUDA GPU.

```bash
pip install -r requirements.txt
```

## Dataset

Expects the **BUSI (Breast Ultrasound Images)** dataset organized into subfolders `benign/`,
`malignant/`, `normal/`, each containing images and their masks (`_mask`, `_mask_1`, ...).
Adjust the path in `DATASET_DIR` inside the notebook/script (defaults to a Kaggle path).

## Usage

The pipeline is controlled by the `RUN_MODE` variable at the top of the notebook/script:

- `"manual_seed"`: trains/evaluates a single seed (`RUN_SEED`).
- `"all_seeds"`: trains/evaluates all seeds in `RUN_SEEDS` and generates final statistics
  (means, standard deviations, bootstrap 95% CIs, paired tests).
- `"final_statistics_only"`: does not train; only reads existing predictions and aggregates
  statistics.

Results are saved under `RESULTS_BASE/EXPERIMENT_ID` (defaults to
`/kaggle/working/results/busi_bus_results`), including per-seed metrics, ROC/PR curves,
confusion matrices, XAI heatmaps, and the statistical summary files used for the paper.

## Citation

If you use this code, please cite it using the metadata in `CITATION.cff` or the Zenodo DOI
generated for the corresponding release (see badge below once published).

<!-- Once linked to Zenodo, replace this line with the badge they provide, e.g.:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

## License

This project is licensed under the MIT License (see `LICENSE`).
