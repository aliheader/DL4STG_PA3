# AI651 Assignment 3 – Deep Learning for Space, Time and Graphs

Ali Haider (Roll No: 24280058)

Code and report for Programming Assignment 3: message passing on road-sensor graphs (Questions 1–2) and spatio-temporal traffic forecasting on the LargeST San Diego dataset (Questions 3–7).

## Files

| File | Contents |
|---|---|
| `Q1-2.ipynb` | Questions 1 and 2 (NumPy, runs locally) |
| `Q3.ipynb` | Questions 3 to 6: data preparation, graph, forecasting models and analysis (runs on Google Colab) |
| `report/report.pdf` | Final report (Questions 1 to 7) |
| `report/REPORT.md` | Same report as Markdown |
| `report/report.tex`, `report/sections/` | LaTeX source of the report |
| `report/figures/` | Figures used in the report |
| `requirements.txt` | Packages for `Q1-2.ipynb` |
| `requirements-colab.txt` | Packages used by `Q3.ipynb` (all preinstalled on Colab) |

## Questions 1–2 (local)

Tested on an Apple M1 Pro (16 GB RAM, CPU only) with Python 3.14.6.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Q1-2.ipynb
```

The notebook uses a fixed random seed (0) and saves `report/figures/q2_heatmaps.png`.

## Questions 3–6 (Google Colab TPU)

Tested on a Colab TPU runtime with 1 TPU chip and a host with 24 vCPUs and 50.5 GB RAM (Python 3.13.18, PyTorch 2.14.1 with `torch-tpu` 0.2.0).

1. Open `Q3.ipynb` in Google Colab, or in VS Code with the Colab extension, and connect to a **TPU** runtime.
2. Run all cells from top to bottom. Nothing needs to be installed: every package in `requirements-colab.txt` is preinstalled. Do **not** install `torch_xla`; it breaks Colab's `torch-tpu`.
3. The Kaggle login cell asks for your Kaggle username and API key (from kaggle.com → Settings → API). They are typed in at run time and not saved in the notebook.
4. The download cell fetches the LargeST files `ca_meta.csv`, `ca_rn_adj.npy`, `ca_his_raw_2020.h5` and `ca_his_raw_2021.h5` (about 15 GB) into `/content/largest`. The runtime needs about 25 GB of RAM.
5. The Drive cell mounts Google Drive and saves one checkpoint per model in `MyDrive/PA-3/checkpoints/` after every epoch:
   - `temporal_only.pt`, `stgnn_fixed.pt`, `stgnn_adaptive.pt`, `stgnn_hybrid.pt`
   - A finished checkpoint is loaded instead of training again, and an interrupted run resumes from its last epoch.
   - Delete a `.pt` file to retrain that model from scratch.
   - If mounting fails in VS Code, run the command **Colab: Mount Google Drive to Server** from the Command Palette.

Training settings (same for all models): 8 input and 4 output steps of 15-minute flow, hidden size 32, Adam with learning rate 0.001, batch size 64, MAE loss on real readings only, at most 100 epochs with early stopping after 10 epochs without a better validation MAE, random seed 0. Each model trains in about 4–10 minutes on the TPU (about 5 s per epoch).

Figures are saved as cell outputs; the ones in `report/figures/` were copied from them.

## Building the report

The PDF is built from the LaTeX source with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd report
tectonic report.tex
```
