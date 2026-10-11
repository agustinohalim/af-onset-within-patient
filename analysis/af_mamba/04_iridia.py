"""Uji AFM 04 — jam jauh dari AF di IRIDIA-AF, sumber utama data latih AF-Mamba.

IRIDIA-AF v1.0.1 (Zenodo 10.5281/zenodo.8405941, CC BY 4.0), hanya berkas RR (*_rr_NN.h5, ms) dan
label AF (*_rr_labels.csv: indeks awal/akhir episode AF) yang diekstrak ke data/iridia-af (lihat README).
IRIDIA hanya menandai AF; jam "jauh" = tanpa AF berlabel, bukan dijamin sinus murni (aritmia lain
tidak dianotasi) — dicatat sebagai batas.

Pemetaan pasien: sid AF-Mamba = nomor patient_NNN di metadata IRIDIA (berkas mereka
RRI_IRIDIA_subj<sid>_...). Pasien yang mereka pakai dinilai HANYA oleh model lipatan yang mengujinya
(out-of-fold); pasien yang tidak lolos kriteria mereka dinilai rerata 5 model.

Jendela jauh: 1 jam, prapemrosesan makalah (potong RR > 10 s, buang RR di luar 0,2-5 s, spline 1 Hz),
berjarak >= 4 jam dari episode AF berlabel mana pun. Jendela pra-AF: skor OOF mereka sendiri dari
01_reproduksi.py (dataset IRIDIA_new, y = 1).

Keluaran: keluaran/iridia.csv. Pakai: python 04_iridia.py [folder IRIDIA]
"""
import glob
import importlib.util
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
DATA_ROOT = os.environ.get('AFWP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
REPO = os.path.join(DATA_ROOT, 'AF_Mamba')
IRIDIA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_ROOT, 'iridia-af')
JAUH = 4 * 3600


def muat(nama, berkas):
    sp = importlib.util.spec_from_file_location(nama, os.path.join(HERE, berkas))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


D = muat('dp', '02_dalam_pasien.py')
R = muat('rep', '01_reproduksi.py')


def rekaman(rec_dir):
    files = sorted(glob.glob(os.path.join(rec_dir, '*_rr_[0-9][0-9].h5')))
    rr_list = [h5py.File(f, 'r')['rr'][:].astype(np.float64) / 1000 for f in files]
    offset = np.cumsum([0] + [len(r) for r in rr_list])
    rr = np.concatenate(rr_list)
    lab = pd.read_csv(glob.glob(os.path.join(rec_dir, '*_rr_labels.csv'))[0])
    t = np.concatenate([[0], np.cumsum(rr)])               # waktu denyut (s); rr[i] berakhir di t[i+1]
    af = [(t[offset[r.start_file_index] + r.start_rr_index], t[offset[r.end_file_index] + r.end_rr_index])
          for r in lab.itertuples()]
    return t, af


def main():
    meta = pd.read_csv(os.path.join(IRIDIA, 'iridia-af-metadata-v1.0.1.csv'))
    base = os.path.join(IRIDIA, 'iridia-af-records-v1.0.1')
    folds = torch.load(os.path.join(REPO, 'Data', 'subject_folds.pt'), weights_only=True)
    lip = {int(sid): k for k in range(5) for sid in folds['af_sets'][k]}
    model, ambang = [], []
    for k in range(5):
        m = R.model_lipatan(k)
        model.append(m)
    # ambang validasi seperti 02 (model k pada lipatan (k+1) mod 5)
    rep = D.muat(os.path.join(DATA_ROOT, 'af_mamba_zenodo', 'structured_dataset_1hz.pt'))
    from sklearn.metrics import roc_curve
    for k in range(5):
        vk = (k + 1) % 5
        Wv, yv = [], []
        for sid in list(folds['af_sets'][vk]) + list(folds['nsr_sets'][vk]):
            for w in R.jendela_subjek(rep[sid]):
                Wv.append(w); yv.append(int(rep[sid]['label']))
        fpr, tpr, thr = roc_curve(yv, R.skor(model[k], Wv))
        ambang.append(thr[np.argmin(np.abs(tpr - 0.9))])
    print('ambang:', np.round(ambang, 4), flush=True)

    W, info = [], []
    for r in meta.itertuples():
        sid = int(r.patient_id.split('_')[1])
        d = os.path.join(base, r.record_id)
        if not os.path.isdir(d):
            continue
        t, af = rekaman(d)
        if not af:
            continue
        for g, x in D.seri_1hz(t):
            for a in np.arange(g[0], g[-1] - 3600 + 1, 3600):
                b = a + 3600
                jarak = min(0 if (s < b and e > a) else min(abs(s - b), abs(a - e)) for s, e in af)
                if jarak < JAUH:
                    continue
                w = D.ambil(g, x, a, b)
                if w is not None:
                    W.append(w); info.append((sid, r.record_id, jarak / 3600))
    print(f'jendela jauh: {len(W)} dari {len(set(i[0] for i in info))} pasien', flush=True)
    S = np.stack([R.skor(m, W) for m in model], 1)
    baris = []
    for (sid, rec, jj), s in zip(info, S):
        if sid in lip:
            k = lip[sid]; p, alarm, oof = s[k], float(s[k] >= ambang[k]), True
        else:
            p, alarm, oof = s.mean(), float(np.mean(s >= np.array(ambang))), False
        baris.append({'sid': sid, 'rec': rec, 'jenis': 'jauh', 'jarak_jam': jj, 'prob': p, 'alarm': alarm,
                      'pasien_latih_mereka': oof})
    o = pd.read_csv(os.path.join(OUT, 'oof_reproduksi.csv'))
    for r in o[o.dataset == 'IRIDIA_new'].itertuples():
        baris.append({'sid': r.sid, 'rec': '', 'jenis': 'pra_af', 'jarak_jam': 1.0, 'prob': r.prob,
                      'alarm': float(r.prob >= ambang[r.lipatan]), 'pasien_latih_mereka': True})
    for r in o[o.y == 0].itertuples():
        baris.append({'sid': r.sid, 'rec': r.dataset, 'jenis': 'sehat', 'jarak_jam': np.nan, 'prob': r.prob,
                      'alarm': float(r.prob >= ambang[r.lipatan]), 'pasien_latih_mereka': True})
    h = pd.DataFrame(baris)
    h.to_csv(os.path.join(OUT, 'iridia.csv'), index=False)

    rng = np.random.default_rng(0)

    def ci(pos, neg, kol='sid'):
        x = pd.concat([pos.assign(y=1), neg.assign(y=0)]); x['g'] = x[kol].astype(str) + x.jenis.where(x.jenis == 'sehat', '')
        u = x.g.unique(); grp = {k: x[x.g == k] for k in u}; a = []
        for _ in range(1000):
            bb = pd.concat([grp[k] for k in rng.choice(u, len(u))])
            if bb.y.nunique() == 2:
                a.append(roc_auc_score(bb.y, bb.prob))
        return roc_auc_score(x.y, x.prob), *np.percentile(a, [2.5, 97.5])

    def alarm_ci(df):
        u = df.sid.unique(); g = {k: df[df.sid == k].alarm.to_numpy() for k in u}
        a = [np.concatenate([g[k] for k in rng.choice(u, len(u))]).mean() for _ in range(1000)]
        return df.alarm.mean(), *np.percentile(a, [2.5, 97.5])

    pra, jauh, sehat = h[h.jenis == 'pra_af'], h[h.jenis == 'jauh'], h[h.jenis == 'sehat']
    jauh_oof = jauh[jauh.pasien_latih_mereka]
    print(f'pra_af {len(pra)}/{pra.sid.nunique()} pasien; jauh {len(jauh)}/{jauh.sid.nunique()} '
          f'(OOF {len(jauh_oof)}/{jauh_oof.sid.nunique()}); sehat {len(sehat)}')
    print('AUROC pra_af lawan sehat            : %.3f [%.3f; %.3f]' % ci(pra, sehat))
    print('AUROC jauh (OOF) lawan sehat        : %.3f [%.3f; %.3f]' % ci(jauh_oof, sehat))
    print('AUROC pra_af lawan jauh (OOF)       : %.3f [%.3f; %.3f]' % ci(pra, jauh_oof))
    both = set(pra.sid) & set(jauh_oof.sid)
    print(f'  pasien yang punya keduanya ({len(both)}): ' + '%.3f [%.3f; %.3f]' % ci(pra[pra.sid.isin(both)], jauh_oof[jauh_oof.sid.isin(both)]))
    print('alarm pra_af : %.3f [%.3f; %.3f]' % alarm_ci(pra))
    for lo, hi in [(4, 6), (6, 12), (12, 24), (24, 1e9)]:
        j = jauh_oof[(jauh_oof.jarak_jam >= lo) & (jauh_oof.jarak_jam < hi)]
        if len(j):
            print(f'alarm jauh OOF {lo}-{hi if hi < 1e9 else "inf"} jam ({len(j)}/{j.sid.nunique()} pasien): '
                  + '%.3f [%.3f; %.3f]' % alarm_ci(j))
    j2 = jauh[~jauh.pasien_latih_mereka]
    if len(j2):
        print(f'alarm jauh, pasien di luar data mereka ({len(j2)}/{j2.sid.nunique()}): ' + '%.3f [%.3f; %.3f]' % alarm_ci(j2))


if __name__ == '__main__':
    main()
