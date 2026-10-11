"""Artikel EKG 2 — apakah prediksi awitan AF bertahan bila kontrolnya dari pasien yang sama?

Pasien AF = rekaman shdb-af dan ltafdb yang punya episode AF. Positif pada horizon h = jendela
sinus yang berakhir h sampai h + 30 menit sebelum awitan AF berikutnya.

  D1 antarbasis-data (gaya Lee dkk. 2026): positif lawan subjek sehat nsrdb + nsr2db.
     Satu subjek satu kelas; AUC atas subjek, seperti artikel 1.
  D2 dalam-pasien: positif lawan jendela sinus PASIEN YANG SAMA yang berjarak >= max(120, h + 90)
     menit dari episode AF mana pun, supaya kontrol tidak pernah bertumpang dengan jendela positif
     (revisi 27 Sep 2026; aturan lama >= 120 menit membuat jendela 120-150 dan 240-270 menit
     sebelum awitan masuk ke kedua kelas). Kepekaan: kontrol tetap >= 360 menit di semua horizon.
     AUC atas jendela, pembagian tetap tingkat pasien.
  D3 awitan palsu dalam-pasien: "positif" = jendela 0-30 menit sebelum titik acak di sinus jauh
     (>= 120 menit dari AF); lawan jendela jauh lain dari pasien yang sama. Garis dasar D2.
  D4 pertukaran basis data untuk D2: latih ltafdb, uji shdb-af, dan sebaliknya (h = 0).

Pembagian StratifiedGroupKFold 5 lipatan, grup = PASIEN, 10 ulangan. Untuk shdb-af, rekaman
dipetakan ke Subject_ID lewat data/shdb-af/AdditionalData.csv (lima pasien punya dua rekaman). Selang 95 % bootstrap
pasien. Model: logistik dan random forest, seperti artikel 1.

Keluaran: keluaran/hasil_af.csv dan ringkasan di layar.
Pakai:    python 02_uji_af.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
DATA_ROOT = os.environ.get('AFWP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
CIRI = ['mean_nn', 'sdnn', 'rmssd', 'pnn50', 'cv_nn', 'lf', 'hf', 'lf_hf', 'sd1', 'sd2']
LOG = ['sdnn', 'rmssd', 'lf', 'hf', 'lf_hf', 'sd1', 'sd2']
HORIZON = [0, 30, 60, 120, 240]
JAUH = 120
JAUH_TETAP = 360
ULANG = 10
BOOT = 1000
PER = 6


def siapkan(f):
    f = f.copy()
    for c in LOG:
        f[c] = np.log(f[c].clip(lower=1e-8))
    f['lf_hf'] = f['lf_hf'].fillna(f['lf_hf'].median())
    a = pd.read_csv(os.path.join(DATA_ROOT, 'shdb-af', 'AdditionalData.csv'))
    peta = dict(zip(a.Data_ID.astype(str).str.zfill(3), a.Subject_ID.astype(str)))
    f['subj'] = f.db + ':' + f.rec.astype(str)
    s = f.db == 'shdb-af'
    f.loc[s, 'subj'] = 'shdb-af:p' + f.loc[s, 'rec'].astype(str).str.zfill(3).map(peta)
    return f


def model(j):
    if j == 'logistik':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0)


def ambil(g, n, rng):
    """Maksimal n jendela acak per subjek (acak dulu, lalu ambil n teratas per subjek)."""
    return g.sample(frac=1, random_state=int(rng.integers(1 << 30))).groupby('subj').head(n)


def oof(d, j, ciri=CIRI):
    X, y, g = d[ciri].to_numpy(), d.y.to_numpy(), d.subj.to_numpy()
    ps = []
    for b in range(ULANG):
        p = np.zeros(len(d))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, y, g):
            p[te] = model(j).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ps.append(p)
    return np.mean(ps, axis=0)


def boot_auc(d, p, tingkat, rng):
    """tingkat 'subjek': rerata skor per subjek; 'jendela': AUC atas jendela, bootstrap subjek."""
    if tingkat == 'subjek':
        s = pd.DataFrame({'s': d.subj.to_numpy(), 'y': d.y.to_numpy(), 'p': p}).groupby('s').agg(y=('y', 'first'), p=('p', 'mean'))
        y, q = s.y.to_numpy(), s.p.to_numpy()
        a = roc_auc_score(y, q)
        bs = [roc_auc_score(y[i], q[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(BOOT)) if 0 < y[i].sum() < len(i)]
        return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5), int(y.sum()), int((1 - y).sum())
    y = d.y.to_numpy()
    a = roc_auc_score(y, p)
    subj = d.subj.unique()
    idx = {s: np.where(d.subj.to_numpy() == s)[0] for s in subj}
    bs = []
    for _ in range(BOOT):
        i = np.concatenate([idx[s] for s in rng.choice(subj, len(subj))])
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], p[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(subj), len(d)


def positif(f, h):
    g = f[(f.pasien_af == 1) & (f.menit_ke_af >= h) & (f.menit_ke_af < h + 30)]
    n = g.groupby('subj').size()
    return g[g.subj.isin(n[n >= 3].index)]


def jauh(f, subj, batas=JAUH):
    return f[f.subj.isin(subj) & (f.menit_dari_af >= batas)]


def dalam_pasien(f, pos, batas, j, rng):
    neg = jauh(f, pos.subj.unique(), batas)
    neg = ambil(neg, PER, rng) if len(neg) else neg
    d2 = pd.concat([pos[pos.subj.isin(neg.subj.unique())].assign(y=1), neg.assign(y=0)])
    assert not d2[d2.y == 1].index.isin(d2[d2.y == 0].index).any(), 'jendela di dua kelas'
    return boot_auc(d2, oof(d2, j), 'jendela', rng)


def main():
    f = siapkan(pd.read_csv(os.path.join(OUT, 'fitur_af.csv'), dtype={'rec': str}))
    rng = np.random.default_rng(2026)
    sehat = ambil(f[f.db.isin(['nsrdb', 'nsr2db'])], PER, rng)
    baris = []
    for j in ('logistik', 'rf'):
        for h in HORIZON:
            pos = positif(f, h)
            d1 = pd.concat([pos.assign(y=1), sehat.assign(y=0)])
            r1 = boot_auc(d1, oof(d1, j), 'subjek', rng)
            r2 = dalam_pasien(f, pos, max(JAUH, h + 90), j, rng)
            r2t = dalam_pasien(f, pos, JAUH_TETAP, j, rng)
            baris += [dict(uji='D1_antarbasis', model=j, h=h, auc=r1[0], lo=r1[1], hi=r1[2], n_pos=r1[3], n_neg=r1[4]),
                      dict(uji='D2_dalam_pasien', model=j, h=h, auc=r2[0], lo=r2[1], hi=r2[2], n_pasien=r2[3], n_jendela=r2[4]),
                      dict(uji='D2_kontrol_360', model=j, h=h, auc=r2t[0], lo=r2t[1], hi=r2t[2], n_pasien=r2t[3], n_jendela=r2t[4])]
            print(f'{j:8s} h={h:4d}  D1 antarbasis AUC {r1[0]:.3f} [{r1[1]:.2f},{r1[2]:.2f}] n={r1[3]}/{r1[4]}'
                  f'   D2 AUC {r2[0]:.3f} [{r2[1]:.2f},{r2[2]:.2f}] pasien={r2[3]}'
                  f'   D2-360 AUC {r2t[0]:.3f} [{r2t[1]:.2f},{r2t[2]:.2f}] pasien={r2t[3]}', flush=True)
        # D3 awitan palsu dalam-pasien
        pos0 = positif(f, 0)
        far = jauh(f, pos0.subj.unique())
        hasil3 = []
        for k in range(10):
            ps, ng = [], []
            for s, g in far.groupby('subj'):
                g = g.sort_values('mulai_s')
                if len(g) < 12:
                    continue
                i = int(rng.integers(6, len(g)))
                tf = g.mulai_s.iloc[i]
                blok = g[(g.mulai_s < tf) & (g.mulai_s >= tf - 1800)]
                if len(blok) < 3:
                    continue
                ps.append(blok)
                sisa = g.drop(blok.index)
                ng.append(sisa.sample(min(PER, len(sisa)), random_state=int(rng.integers(1 << 30))))
            d3 = pd.concat([pd.concat(ps).assign(y=1), pd.concat(ng).assign(y=0)])
            hasil3.append(boot_auc(d3, oof(d3, j), 'jendela', rng)[0])
        baris.append(dict(uji='D3_awitan_palsu', model=j, h=0, auc=float(np.mean(hasil3)),
                          lo=float(np.min(hasil3)), hi=float(np.max(hasil3))))
        print(f'{j:8s} D3 awitan palsu dalam-pasien AUC rerata {np.mean(hasil3):.3f} [min {np.min(hasil3):.2f}, maks {np.max(hasil3):.2f}]', flush=True)
        # D4 pertukaran basis data, h = 0
        neg0 = ambil(jauh(f, pos0.subj.unique()), PER, rng)
        d = pd.concat([pos0[pos0.subj.isin(neg0.subj.unique())].assign(y=1), neg0.assign(y=0)])
        for latih, uji in (('ltafdb', 'shdb-af'), ('shdb-af', 'ltafdb')):
            a, b = d[d.db == latih], d[d.db == uji]
            m = model(j).fit(a[CIRI].to_numpy(), a.y.to_numpy())
            p = m.predict_proba(b[CIRI].to_numpy())[:, 1]
            r4 = boot_auc(b.reset_index(drop=True), p, 'jendela', rng)
            baris.append(dict(uji=f'D4_{latih}_ke_{uji}', model=j, h=0, auc=r4[0], lo=r4[1], hi=r4[2], n_pasien=r4[3]))
            print(f'{j:8s} D4 latih {latih} uji {uji}: AUC {r4[0]:.3f} [{r4[1]:.2f},{r4[2]:.2f}] pasien={r4[3]}', flush=True)
    pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_af.csv'), index=False)


if __name__ == '__main__':
    main()
