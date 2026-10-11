"""Artikel EKG 2 — D1 lawan D2 dengan jaringan konvolusi 1D atas deret RR satu jam.

Menanggapi keberatan yang paling mungkin: ciri HRV 5 menit terlalu dangkal. Masukannya sama jenis
dengan Lee dkk. 2026 — deret RR satu jam — dan modelnya CNN 1D kecil yang dilatih dari nol.

  D1 antarbasis   pra(h) lawan sehat (nsrdb + nsr2db); AUC atas subjek
  D2 dalam-pasien pra(h) lawan jauh (pasien AF yang sama, >= 120 menit dari AF); AUC atas sampel,
                  selang bootstrap pasien

Arsitektur: tiga blok Conv1d(kernel 7) + BatchNorm + ReLU + MaxPool(4), rerata global, satu
lapis linear; sekitar 20 ribu parameter. Adam 1e-3, 25 epoch, batch 32, bobot kelas seimbang.
Standardisasi skalar dari data latih lipatan. StratifiedGroupKFold 5 lipatan (grup = rekaman),
3 ulangan. Berjalan di CPU.

Keluaran: keluaran/hasil_cnn.csv dan ringkasan di layar.
Pakai:    python 04_cnn.py
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
DATA_ROOT = os.environ.get('AFWP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
ULANG, EPOCH, BATCH, BOOT = 3, 25, 32, 1000
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


class Jaringan(nn.Module):
    def __init__(self):
        super().__init__()
        blok = []
        for cin, cout in ((1, 16), (16, 32), (32, 32)):
            blok += [nn.Conv1d(cin, cout, 7, padding=3), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(4)]
        self.fitur = nn.Sequential(*blok)
        self.kepala = nn.Linear(32, 1)

    def forward(self, x):
        return self.kepala(self.fitur(x).mean(dim=2)).squeeze(1)


def latih_prediksi(Xtr, ytr, Xte, benih):
    torch.manual_seed(benih)
    mu, sd = Xtr.mean(), Xtr.std() + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(ytr, dtype=torch.float32)
    m = Jaringan()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    pos = max(ytr.mean(), 1e-3)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    for _ in range(EPOCH):
        m.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss(m(xt[b]), yt[b]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor((Xte - mu) / sd, dtype=torch.float32).unsqueeze(1))).numpy()


def oof(X, y, g):
    ps = []
    for b in range(ULANG):
        p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, y, g):
            p[te] = latih_prediksi(X[tr], y[tr], X[te], b)
        ps.append(p)
    return np.mean(ps, axis=0)


def auc_subjek(y, p, g, rng):
    s = pd.DataFrame({'g': g, 'y': y, 'p': p}).groupby('g').agg(y=('y', 'first'), p=('p', 'mean'))
    yy, pp = s.y.to_numpy(), s.p.to_numpy()
    bs = [roc_auc_score(yy[i], pp[i]) for i in (rng.integers(0, len(yy), len(yy)) for _ in range(BOOT)) if 0 < yy[i].sum() < len(i)]
    return roc_auc_score(yy, pp), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def auc_sampel(y, p, g, rng):
    subj = np.unique(g)
    idx = {s: np.where(g == s)[0] for s in subj}
    bs = []
    for _ in range(BOOT):
        i = np.concatenate([idx[s] for s in rng.choice(subj, len(subj))])
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    X = np.load(os.path.join(OUT, 'rr_sejam.npz'))['X'].astype(np.float32)
    m = pd.read_csv(os.path.join(OUT, 'rr_sejam_meta.csv'), dtype={'rec': str})
    m['g'] = m.db + ':' + m.rec
    # grup = pasien: lima pasien shdb-af punya dua rekaman (AdditionalData.csv)
    a = pd.read_csv(os.path.join(DATA_ROOT, 'shdb-af', 'AdditionalData.csv'))
    peta = dict(zip(a.Data_ID.astype(str).str.zfill(3), a.Subject_ID.astype(str)))
    s = m.db == 'shdb-af'
    m.loc[s, 'g'] = 'shdb-af:p' + m.loc[s, 'rec'].str.zfill(3).map(peta)
    rng = np.random.default_rng(2026)
    baris = []
    for h in (0, 60, 240):
        pra = m[(m.jenis == 'pra') & (m.h == h)]
        # D1
        i1 = np.r_[pra.index, m[m.jenis == 'sehat'].index]
        y1 = np.r_[np.ones(len(pra)), np.zeros(len(i1) - len(pra))].astype(np.float32)
        p1 = oof(X[i1], y1, m.g.to_numpy()[i1])
        r1 = auc_subjek(y1, p1, m.g.to_numpy()[i1], rng)
        # D2: pasien yang punya pra dan jauh
        # kontrol berjarak >= max(120, h + 90) menit dari AF, supaya tidak bertumpang dengan jam pra
        jauh = m[(m.jenis == 'jauh') & (m.jarak_menit >= max(120, h + 90)) & m.g.isin(pra.g)]
        pr = pra[pra.g.isin(jauh.g)]
        i2 = np.r_[pr.index, jauh.index]
        y2 = np.r_[np.ones(len(pr)), np.zeros(len(jauh))].astype(np.float32)
        p2 = oof(X[i2], y2, m.g.to_numpy()[i2])
        r2 = auc_sampel(y2, p2, m.g.to_numpy()[i2], rng)
        baris += [dict(uji='D1_antarbasis_cnn', h=h, auc=r1[0], lo=r1[1], hi=r1[2], n_pra=len(pra), n_pasien=pra.g.nunique()),
                  dict(uji='D2_dalam_pasien_cnn', h=h, auc=r2[0], lo=r2[1], hi=r2[2], n_pra=len(pr), n_pasien=pr.g.nunique())]
        print(f'CNN h={h:4d}  D1 antarbasis AUC {r1[0]:.3f} [{r1[1]:.2f},{r1[2]:.2f}] pasien AF={pra.g.nunique()}'
              f'   D2 dalam-pasien AUC {r2[0]:.3f} [{r2[1]:.2f},{r2[2]:.2f}] pasien={pr.g.nunique()} sampel pra={len(pr)}', flush=True)
    pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_cnn.csv'), index=False)


if __name__ == '__main__':
    main()
