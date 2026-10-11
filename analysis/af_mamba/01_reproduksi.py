"""Uji AFM 01 — reproduksi AUROC AF-Mamba dengan port PyTorch murni (mamba_murni.py).

Data, lipatan dan bobot dari Lee & Chon 2026 (Zenodo 10.5281/zenodo.22572206, ODC-By; GitHub
yongbin98/AF_Mamba). Jendela dibangun seperti build_windows_for_subject mereka: subjek AF = jam ke-2
sampai ke-1 sebelum awitan (rri[-7200:-3600]); subjek NSR = 3.600 sampel pertama tiap episode.
Lipatan k: uji = af_sets[k] + nsr_sets[k], model = fold_k.pt. Probabilitas = softmax kelas 1.

Lolos bila AUROC per lipatan dan rerata mendekati yang dilaporkan (0,974 rerata 5 lipatan). Skor
OOF disimpan untuk dipakai ulang.

Keluaran: keluaran/oof_reproduksi.csv. Pakai: python 01_reproduksi.py
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from mamba_murni import AFMamba
from muat_aman import muat
DATA_ROOT = os.environ.get('AFWP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(DATA_ROOT, 'AF_Mamba')
OUT = os.path.join(HERE, 'keluaran')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(os.cpu_count())


def jendela_subjek(s, masuk=3600, horizon=3600):
    out = []
    for eid in sorted(s['episodes']):
        r = np.asarray(s['episodes'][eid]['rri'], dtype=np.float32).reshape(-1)
        if int(s['label']) == 1:
            if len(r) < masuk + horizon:
                continue
            w = r[-(masuk + horizon):-horizon]
        else:
            if len(r) < masuk:
                continue
            w = r[:masuk]
        if len(w) == masuk:
            out.append(w)
    return out


def model_lipatan(k, sub=''):
    m = AFMamba()
    p = os.path.join(REPO, 'Trained_Models', 'af_mamba', sub, f'fold_{k}.pt')
    m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
    return m.to(DEV).eval()


@torch.no_grad()
def skor(m, W, bs=64):
    X = torch.from_numpy(np.stack(W)).unsqueeze(1)
    return np.concatenate([torch.softmax(m(X[i:i + bs].to(DEV)), 1)[:, 1].cpu().numpy() for i in range(0, len(X), bs)])


def main():
    os.makedirs(OUT, exist_ok=True)
    data = muat(os.path.join(DATA_ROOT, 'af_mamba_zenodo', 'structured_dataset_1hz.pt'))
    f = torch.load(os.path.join(REPO, 'Data', 'subject_folds.pt'), weights_only=True)
    baris = []
    for k in range(5):
        m = model_lipatan(k)
        W, meta = [], []
        for sid in list(f['af_sets'][k]) + list(f['nsr_sets'][k]):
            for w in jendela_subjek(data[sid]):
                W.append(w); meta.append((sid, data[sid]['dataset'], int(data[sid]['label'])))
        s = skor(m, W)
        for (sid, ds, y), p in zip(meta, s):
            baris.append({'lipatan': k, 'sid': sid, 'dataset': ds, 'y': y, 'prob': p})
        d = pd.DataFrame(baris); d = d[d.lipatan == k]
        print(f'lipatan {k}: {len(d)} jendela, AF {d.y.sum()}, AUROC {roc_auc_score(d.y, d.prob):.4f}', flush=True)
    d = pd.DataFrame(baris)
    d.to_csv(os.path.join(OUT, 'oof_reproduksi.csv'), index=False)
    per = [roc_auc_score(g.y, g.prob) for _, g in d.groupby('lipatan')]
    print(f'AUROC rerata 5 lipatan {np.mean(per):.4f} ± {np.std(per):.4f}; gabungan {roc_auc_score(d.y, d.prob):.4f}')


if __name__ == '__main__':
    main()
