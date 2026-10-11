"""Uji AFM 02 — model AF-Mamba terlatih (bobot asli Lee & Chon 2026) pada jam sinus pasien AF sendiri.

Pertanyaan: apakah AF-Mamba memberi skor "pra-AF" pada jam sinus pasien AF yang JAUH dari episode AF
sama tingginya dengan jam yang benar-benar sebelum awitan? Bila ya, AUROC 0,974 mereka mengenali
pasien AF lawan orang sehat, bukan awitan yang mendekat.

Data (anotasi saja, ODC-By): ltafdb (.atr denyut + irama) dan shdb-af (.qrs denyut, .atr irama) dari
data/. shdb-af tidak pernah dipakai Lee & Chon → uji eksternal murni. Dari ltafdb, sembilan
rekaman yang mereka pakai (00, 08, 23, 35, 102, 104, 114, 116, 120) dinilai hanya dengan model lipatan
yang mengujinya; rekaman lain dan shdb-af dengan kelima model (skor dirata-rata).

Prapemrosesan meniru makalah (bagian IV.A): RR dari anotasi denyut, potong di RR > 10 s, buang RR di
luar 0,2-5 s, spline kubik 1 Hz, jendela 3.600 sampel.

Jendela:
  pra_af   jam ke-2 sampai ke-1 sebelum awitan AF (episode '(AFIB' >= 60 s), dengan 2 jam sebelum
           awitan seluruhnya '(N' dan tanpa celah — seperti jendela positif mereka
  jauh     jam yang seluruhnya '(N', tanpa celah, berjarak >= JAUH_JAM dari episode AF/AFL mana pun
           (sebelum maupun sesudah), dari pasien yang sama; tidak bertumpang
  sehat    jendela NSR mereka sendiri (nsrdb + nsr2db) dengan skor OOF dari 01_reproduksi.py

Ambang tiap lipatan = ambang validasi mereka (sensitivitas 0,90 pada lipatan (k+1) mod 5), seperti
test.ipynb. "Alarm" = skor >= ambang.

Keluaran: keluaran/dalam_pasien.csv (skor per jendela). Pakai: python 02_dalam_pasien.py
"""
import os

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.interpolate import CubicSpline
from sklearn.metrics import roc_curve

from mamba_murni import AFMamba
from muat_aman import muat
import importlib.util as _iu
DATA_ROOT = os.environ.get('AFWP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))
_sp = _iu.spec_from_file_location('rep01', os.path.join(os.path.dirname(os.path.abspath(__file__)), '01_reproduksi.py'))
R = _iu.module_from_spec(_sp); _sp.loader.exec_module(R)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
REPO = os.path.join(DATA_ROOT, 'AF_Mamba')
SUMBER = {'ltafdb': (os.path.join(DATA_ROOT, 'ltafdb'), 'atr'),
          'shdb-af': (os.path.join(DATA_ROOT, 'shdb-af'), 'qrs')}
DIPAKAI_MEREKA = {'00': 10000, '08': 10008, '23': 10023, '35': 10035, '102': 10102, '104': 10104,
                  '114': 10114, '116': 10116, '120': 10120}
BEAT = list('NLRBAaJSVrFejnE/fQ')
JAUH_JAM = [4, 6]
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(os.cpu_count())


def selang(ann, akhir):
    t = ann.sample / ann.fs
    u = [(ti, a.strip('\x00').strip()) for ti, a in zip(t, ann.aux_note) if a and a.strip('\x00').startswith('(')]
    return [(ti, u[k + 1][0] if k + 1 < len(u) else akhir, a) for k, (ti, a) in enumerate(u)]


def seri_1hz(tb):
    """Daftar (t_mulai, larik 1 Hz) per segmen kontinu, aturan makalah."""
    rr = np.diff(tb); t = tb[1:]
    seg, mulai = [], 0
    potong = np.flatnonzero(rr > 10)
    for p in list(potong) + [len(rr)]:
        r, tt = rr[mulai:p], t[mulai:p]
        ok = (r >= 0.2) & (r <= 5)
        r, tt = r[ok], tt[ok]
        if len(r) > 10 and tt[-1] - tt[0] >= 3600:
            g = np.arange(np.ceil(tt[0]), np.floor(tt[-1]) + 1)
            seg.append((g, CubicSpline(tt, r)(g).astype(np.float32)))
        mulai = p + 1
    return seg


def ambil(g, x, a, b):
    m = (g >= a) & (g < b)
    return x[m] if m.sum() == 3600 else None


def jendela_rekaman(db, rec, folder, ekst, rng):
    p = os.path.join(folder, rec)
    h = wfdb.rdheader(p)
    beat = wfdb.rdann(p, ekst)
    tb = beat.sample[np.isin(np.array(beat.symbol), BEAT)] / h.fs
    akhir = h.sig_len / h.fs if h.sig_len else tb.max()
    ir = selang(wfdb.rdann(p, 'atr'), akhir)
    if not ir:
        return []
    af = [(s, e) for s, e, k in ir if k.startswith('(AFIB') or k.startswith('(AFL')]
    onset = [s for s, e, k in ir if k.startswith('(AFIB') and e - s >= 60]
    nsr = [(s, e) for s, e, k in ir if k == '(N']
    segs = seri_1hz(tb)
    seluruh_n = lambda a, b: any(s <= a and e >= b for s, e in nsr)
    out = []
    for o in onset:
        if not seluruh_n(o - 7200, o):
            continue
        for g, x in segs:
            w = ambil(g, x, o - 7200, o - 3600)
            if w is not None:
                out.append({'db': db, 'rec': rec, 'jenis': 'pra_af', 'jarak_jam': 1.0, 'w': w})
                break
    for g, x in segs:
        for a in np.arange(g[0], g[-1] - 3600 + 1, 3600):
            b = a + 3600
            if not seluruh_n(a, b):
                continue
            jarak = min([min(abs(s - b), abs(a - e)) if not (s < b and e > a) else 0 for s, e in af], default=np.inf) / 3600
            if jarak < min(JAUH_JAM) or not af:
                continue
            w = ambil(g, x, a, b)
            if w is not None:
                out.append({'db': db, 'rec': rec, 'jenis': 'jauh', 'jarak_jam': jarak, 'w': w})
    return out


@torch.no_grad()
def skor(m, W, bs=32):
    X = torch.from_numpy(np.stack(W)).unsqueeze(1)
    return np.concatenate([torch.softmax(m(X[i:i + bs].to(DEV)), 1)[:, 1].cpu().numpy() for i in range(0, len(X), bs)])


def main():
    rng = np.random.default_rng(0)
    rep = muat(os.path.join(DATA_ROOT, 'af_mamba_zenodo', 'structured_dataset_1hz.pt'))
    folds = torch.load(os.path.join(REPO, 'Data', 'subject_folds.pt'), weights_only=True)
    oof = pd.read_csv(os.path.join(OUT, 'oof_reproduksi.csv'))
    model, ambang = [], []
    for k in range(5):
        m = AFMamba()
        m.load_state_dict(torch.load(os.path.join(REPO, 'Trained_Models', 'af_mamba', f'fold_{k}.pt'),
                                     map_location='cpu', weights_only=True))
        model.append(m.to(DEV).eval())
        # set validasi model k = subjek lipatan (k+1) mod 5, dinilai oleh model k sendiri (test.ipynb)
        vk = (k + 1) % 5
        Wv, yv = [], []
        for sid in list(folds['af_sets'][vk]) + list(folds['nsr_sets'][vk]):
            for w in R.jendela_subjek(rep[sid]):
                Wv.append(w); yv.append(int(rep[sid]['label']))
        fpr, tpr, thr = roc_curve(yv, skor(model[k], Wv))
        ambang.append(thr[np.argmin(np.abs(tpr - 0.9))])
    print('ambang validasi per lipatan:', np.round(ambang, 4), flush=True)
    lipatan_sid = {sid: k for k in range(5) for sid in folds['af_sets'][k]}

    J = []
    for db, (folder, ekst) in SUMBER.items():
        for rec in sorted(f[:-4] for f in os.listdir(folder) if f.endswith('.hea')):
            if os.path.exists(os.path.join(folder, rec + '.' + ekst)) and os.path.exists(os.path.join(folder, rec + '.atr')):
                J += jendela_rekaman(db, rec, folder, ekst, rng)
        print(db, 'jendela:', pd.Series([j['jenis'] for j in J if j['db'] == db]).value_counts().to_dict(), flush=True)
    W = [j['w'] for j in J]
    S = np.stack([skor(m, W) for m in model], 1)             # [n, 5]
    baris = []
    for j, s in zip(J, S):
        sid = DIPAKAI_MEREKA.get(j['rec']) if j['db'] == 'ltafdb' else None
        if sid is not None:
            k = lipatan_sid[sid]; p, alarm = s[k], s[k] >= ambang[k]
        else:
            p, alarm = s.mean(), np.mean(s >= np.array(ambang))
        baris.append({'db': j['db'], 'rec': j['rec'], 'jenis': j['jenis'], 'jarak_jam': j['jarak_jam'],
                      'dipakai_mereka': sid is not None, 'prob': p, 'alarm': alarm})
    # jendela sehat: skor OOF mereka sendiri, alarm dengan ambang lipatannya
    for _, r in oof[oof.y == 0].iterrows():
        baris.append({'db': r.dataset, 'rec': str(r.sid), 'jenis': 'sehat', 'jarak_jam': np.nan,
                      'dipakai_mereka': True, 'prob': r.prob, 'alarm': float(r.prob >= ambang[int(r.lipatan)])})
    d = pd.DataFrame(baris)
    d.to_csv(os.path.join(OUT, 'dalam_pasien.csv'), index=False)
    print(d.groupby(['db', 'jenis']).agg(n=('prob', 'size'), pasien=('rec', 'nunique'),
                                         prob_median=('prob', 'median'), alarm=('alarm', 'mean')).round(3).to_string())


if __name__ == '__main__':
    main()
