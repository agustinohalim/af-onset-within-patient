# af-onset-within-patient

Analysis code for the manuscript *Atrial fibrillation onset predictors trained against healthy
controls recognise AF patients, not approaching episodes* (Halim, Lukman, Andika, Macnamara, Erwin;
submitted).

Two parts:

- **`analysis/af_mamba/`** runs the published, trained weights of AF-Mamba (Lee & Chon, 2026,
  arXiv:2609.06984) on sinus-rhythm hours far from any AF episode in the same patients. The Mamba
  layer of the original code needs GPU kernels; `mamba_murni.py` re-implements it in plain PyTorch
  (following the reference selective scan of `mamba_ssm`) so the released weights load unchanged and
  run on a CPU. `01_reproduksi.py` checks the port against the paper (AUC 0.974).
- **`analysis/retrained/`** compares a between-database design (pre-onset sinus of AF patients
  against healthy subjects from other databases) with a within-patient design (against the same
  patients' sinus rhythm far from any AF episode) using HRV features, logistic regression, random
  forest and a small 1D CNN on one-hour RR sequences.

**No data are included**, and no code from the AF-Mamba repository (it carries no licence). The
scripts read everything from `data/` (or the folder in the environment variable `AFWP_DATA`):

| Folder | Source | Licence |
|---|---|---|
| `data/ltafdb`, `data/shdb-af`, `data/nsr2db` | PhysioNet; fetched by `retrained/01_jendela.py` (annotations only) | ODC-By 1.0 |
| `data/nsrdb` | PhysioNet; fetched by `00_get_data.py` (.hea, .atr) | ODC-By 1.0 |
| `data/AF_Mamba` | https://github.com/yongbin98/AF_Mamba (trained weights, subject folds), cloned by `00_get_data.py` | none stated; used only to load the weights |
| `data/af_mamba_zenodo` | `structured_dataset_1hz.pt`, https://doi.org/10.5281/zenodo.22572206 | ODC-By |
| `data/iridia-af` | IRIDIA-AF v1.0.1, https://doi.org/10.5281/zenodo.8405941; download by hand, then `00_get_data.py --iridia <zip>` extracts only the RR files | CC BY 4.0 |

Third-party `.pt` files are loaded with PyTorch's restricted unpickler (`weights_only=True`, with
only the NumPy array classes allowed; `muat_aman.py`), so no arbitrary code is executed.

## Requirements

Python 3.12 with NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.0, PyTorch 2.14.0 (CPU),
wfdb 4.3.1, h5py 3.16.0 and matplotlib 3.11.1. No GPU is needed.

## Scripts and where their results appear

Comments and log text are in Indonesian (*jendela* = window, *pra_af* = pre-onset, *jauh* = distant,
*sehat* = healthy, *lipatan* = fold, *ambang* = threshold, *dalam pasien* = within patient).

| Script | Log | Manuscript |
|---|---|---|
| `00_get_data.py` | — | Data |
| `af_mamba/01_reproduksi.py` | `logs/af_mamba/reproduksi.log` | Port check, Section 4.1 |
| `af_mamba/02_dalam_pasien.py` | `logs/af_mamba/dalam_pasien.log` | SHDB-AF and LTAFDB windows, thresholds |
| `af_mamba/03_ringkas.py` | `logs/af_mamba/ringkas.log` | Table 1, SHDB-AF column; LTAFDB |
| `af_mamba/04_iridia.py` | `logs/af_mamba/iridia.log` | Table 1, IRIDIA-AF column |
| `retrained/01_jendela.py` | `logs/retrained/jendela.log` | Five-minute sinus windows and HRV features |
| `retrained/02_uji_af.py` | `logs/retrained/uji_af.log` | Tables 2–3 |
| `retrained/03_rr_sejam.py` | `logs/retrained/rr_sejam.log` | One-hour RR sequences for the CNN |
| `retrained/04_cnn.py` | `logs/retrained/cnn.log` | Table 4 |
| `retrained/05_jam.py` | `logs/retrained/jam.log` | Table 5, time of day |
| `retrained/06_gambar.py` | — | Figure 1 (`figures/gambar_ekg2_horizon.png`) |

Run each from its own folder, in order, for example:

```bash
cd analysis && python 00_get_data.py
cd af_mamba && python -u 01_reproduksi.py > ../../logs/af_mamba/reproduksi.log
```

Intermediate files go to `keluaran/` next to each script. The AF-Mamba scripts reproduce the logs
exactly. CNN training on CPU is not fully deterministic, so `retrained/04_cnn.py` reproduces its values
to about the second decimal; tree models are seeded and reproduce exactly.

## Licence

Code in this repository: MIT (see `LICENSE`). Data and the AF-Mamba weights are not included and
remain under their owners' terms.
