# Earthquake Magnitude Estimation from Early P-Wave Observations

Senior thesis (AUCA, defended May 2026) by Aidarbek Papakov. A CNN–LSTM–MLP model
estimates earthquake magnitude from the first 3–10 seconds of a three-component
waveform after the P-wave arrival, trained on the INSTANCE dataset.

## Results

Test-set performance by P-wave window length (Chapter 6, Table 6.4):

| Window | Bias | MAE | Spearman r | R² |
|--------|------|-----|------------|-----|
| 3 s | −0.040 | 0.3699 | 0.7755 | 0.6483 |
| 5 s | −0.071 | 0.3453 | 0.8028 | 0.6851 |
| 8 s | −0.027 | 0.3217 | 0.8290 | 0.7225 |
| 10 s | −0.032 | 0.3152 | 0.8392 | 0.7478 |

Best hyperparameters (100-trial random search): lr 0.001368, weight decay 0.005068,
batch size 128, Huber loss (δ ≈ 0.3), cosine schedule, early stopping after 150 epochs.

## Repository layout

| Path | Contents |
|------|----------|
| `chapters/markdowns/` | Thesis text, Chapters 1–8 (Markdown, in sync with the final PDF) |
| `docs/final/` | Defense package: thesis PDF, supervisor review, presentation, pre-defense speech |
| `docs/notes/eew_background.md` | Personal glossary and background notes on earthquake early warning |
| `src/model_definition.py` | The model (STFT → CNN → LSTM → MLP head) |
| `src/dataset_definition.py` | `UnifiedSeismicDataset`, a memmap-backed dataset |
| `src/preprocess_instance.py` | Builds the stratified INSTANCE dataset used in the thesis |
| `src/preprocess_unified.py` | Variant that also mixes in STEAD |
| `src/training_on_stratified_sample.py` | Main training script: experiments, random search, test evaluation, PCA plot |
| `src/training_on_sample.py`, `src/training_on_stratified_with_noise.py` | Earlier experiment variants, kept for history (own dataset/model code) |
| `notebooks/` | Exploratory notebooks (EDA, embedding projection, distributions) |
| `example_scripts/INSTANCE/` | Reference plotting code from the INSTANCE authors |
| `data/` | Preprocessed memmaps and metadata (binaries are gitignored) |
| `training_results/` | Per-day experiment outputs, see its own README |

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements are unpinned because the original environment was not recorded. A CUDA GPU is recommended.

## Running

1. Download INSTANCE (and STEAD, if you use the unified variant) and set the paths at the top of
   `src/preprocess_instance.py` (they currently point to the author's `/mnt/d/...` locations).
   Then run it to write `data/INSTANCE/peak_normalized/`.
2. Run training from the repository root:

   ```bash
   python src/training_on_stratified_sample.py
   ```

   The `__main__` block at the bottom selects what runs (hyperparameter search by default; the
   per-window experiments are commented out). Results go to `training_results/<date>/...`.

## Notes

- The final architecture has no attention pooling. An attention-pooling variant was tried and
  gave no measurable gain; see the docstring of `src/model_definition.py` and Chapter 5.
- `src/model_definition.py` was aligned to the thesis text after the defense. The original
  experiment scripts in git history used attention pooling and `Dropout2d`, so checkpoints from
  those runs will not load into the current model.
