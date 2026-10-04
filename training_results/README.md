# Training results

One folder per day of experiments (`YYYY_M_D`, no zero padding), written by the
training scripts in `src/`. Each run holds `hyperparams.json` (config, per-epoch
metrics and, from mid-April on, test metrics), plus plots such as
`training_metrics.jpg`, `regression_metrics.jpg`, `residuals_test.jpg` and
`embedding_pca.jpg`. Model weights (`saved_models/`, `*.pth`) are gitignored.

The folder naming changed as the experiments evolved and was never
back-filled:

| Dates | Layout | What it was |
|-------|--------|-------------|
| 4/1 - 4/5 | `magnitude_pred_{300,500,800}_samples/run_N/` | First runs at 3 s / 5 s / 8 s windows |
| 4/6 - 4/9 | `mag_pred_{N}_{attention,baseline,bilstm,nopool,resnet}/` | Architecture ablation |
| 4/8 onward | `INSTANCE/<preprocessing variant>/magnitude_pred_N_samples/` or `mixed/...` | Per-dataset and per-preprocessing runs |
| 4/13 onward | `.../hparam_search/trial_N/` + `search_summary.json` | Random hyperparameter search |

The window length in a folder name is in samples at 100 Hz (300 = 3 s).
Folders from 4/14 onward were not committed to git in the original history.
