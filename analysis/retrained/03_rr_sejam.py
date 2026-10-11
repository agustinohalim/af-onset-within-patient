"""Artikel EKG 2 — deret RR satu jam untuk model jaringan saraf (menanggapi Lee dkk. 2026).

Masukan Lee dkk. adalah deret RR satu jam. Di sini deret RR setiap jendela satu jam yang
seluruhnya sinus diinterpolasi ke kisi 2 Hz (7.200 titik). Tidak dinormalisasi per deret, supaya
laju jantung tetap menjadi informasi yang bisa dipakai model; standardisasi global dilakukan saat
pelatihan.

Jenis sampel:
  pra     satu jam yang berakhir h menit sebelum awitan AF (h = 0, 60, 240), maksimal 3 awitan
          per pasien, jendela harus seluruhnya sinus
  jauh    satu jam sinus pasien AF yang berjarak >= 120 menit dari episode AF mana pun,
          maksimal 6 per pasien
  sehat   satu jam acak dari nsrdb dan nsr2db, maksimal 6 per subjek

Anotasi dibaca dari folder data/ yang sudah diunduh 01_jendela.py.

Keluaran: keluaran/rr_sejam.npz (X float16, n x 7200) dan keluaran/rr_sejam_meta.csv
Pakai:    python 03_rr_sejam.py
"""
import os
from importlib import import_module

import numpy as np
import pandas as pd
import wfdb
DATA_ROOT = os.environ.get('AFWP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))

j1 = import_module('01_jendela')
HERE = j1.HERE
OUT = j1.OUT
SEJAM = 3600.0
HZ = 2.0
N = int(SEJAM * HZ)
HORIZON = [0, 60, 240]
MAKS_PRA, MAKS_JAUH, MAKS_SEHAT = 3, 6, 6
JAUH = 120 * 60


def deret(t, a):
    m = (t >= a) & (t < a + SEJAM)
    tb = t[m]
    if len(tb) < 1800:          # kurang dari 30 denyut per menit rata-rata = sinyal hilang
        return None
    rr = np.diff(tb)
    ok = (rr > 0.3) & (rr < 2.0)
    if ok.mean() < 0.9:
        return None
    tt, rr = tb[1:][ok], rr[ok]
    kisi = a + np.arange(N) / HZ
    return np.interp(kisi, tt, rr).astype(np.float16)


def sinus(a, non):
    return not any(s < a + SEJAM and e > a for s, e in non)


def main():
    rng = np.random.default_rng(2026)
    X, meta = [], []
    for db, cfg in j1.SUMBER.items():
        folder = os.path.join(j1.DATA, db)
        for r in sorted({p.split('.')[0] for p in os.listdir(folder) if p.endswith('.hea')}):
            path = os.path.join(folder, r)
            if not os.path.exists(f'{path}.{cfg["beat"]}') or (cfg['ritme'] and not os.path.exists(f'{path}.{cfg["ritme"]}')):
                continue
            h = wfdb.rdheader(path)
            beat = wfdb.rdann(path, cfg['beat'])
            sym = np.array(beat.symbol)
            t = beat.sample[np.isin(sym, list('NLRBAaJSVrFejnE/fQ'))] / h.fs
            dur = h.sig_len / h.fs if h.sig_len else t.max()
            non, af = j1.episode(wfdb.rdann(path, cfg['ritme'])) if cfg['ritme'] else ([], [])
            if db != 'nsr2db' and not af:
                continue
            if db == 'nsr2db':
                calon = rng.permutation(np.arange(0, dur - SEJAM, SEJAM))[:MAKS_SEHAT]
                for a in calon:
                    x = deret(t, a)
                    if x is not None:
                        X.append(x); meta.append(dict(db=db, rec=r, jenis='sehat', h=np.nan, mulai_s=a))
                continue
            for hz in HORIZON:
                for s, _ in af[:MAKS_PRA]:
                    a = s - hz * 60 - SEJAM
                    if a >= 0 and sinus(a, non):
                        x = deret(t, a)
                        if x is not None:
                            X.append(x); meta.append(dict(db=db, rec=r, jenis='pra', h=hz, mulai_s=a))
            jauh = [a for a in np.arange(0, dur - SEJAM, SEJAM / 2)
                    if sinus(a, non) and all(a - e >= JAUH or s - (a + SEJAM) >= JAUH for s, e in af)]
            for a in rng.permutation(jauh)[:MAKS_JAUH]:
                x = deret(t, a)
                if x is not None:
                    jarak = min(min(abs(a - e), abs(s - (a + SEJAM))) for s, e in af) / 60
                    X.append(x); meta.append(dict(db=db, rec=r, jenis='jauh', h=np.nan, mulai_s=a, jarak_menit=jarak))
            print(f'{db:8s} {r:6s} {sum(1 for m in meta if m["rec"] == r and m["db"] == db):3d} sampel', flush=True)
    nsr = os.path.join(DATA_ROOT, 'nsrdb')
    for p in sorted(os.listdir(nsr)):
        if p.endswith('.hea'):
            r = p[:-4]
            h = wfdb.rdheader(os.path.join(nsr, r))
            beat = wfdb.rdann(os.path.join(nsr, r), 'atr')
            sym = np.array(beat.symbol)
            t = beat.sample[np.isin(sym, list('NLRBAaJSVrFejnE/fQ'))] / h.fs
            for a in rng.permutation(np.arange(0, h.sig_len / h.fs - SEJAM, SEJAM))[:MAKS_SEHAT]:
                x = deret(t, a)
                if x is not None:
                    X.append(x); meta.append(dict(db='nsrdb', rec=r, jenis='sehat', h=np.nan, mulai_s=a))
    np.savez_compressed(os.path.join(OUT, 'rr_sejam.npz'), X=np.stack(X))
    m = pd.DataFrame(meta)
    m.to_csv(os.path.join(OUT, 'rr_sejam_meta.csv'), index=False)
    print(m.groupby(['jenis', 'h'], dropna=False).size(), '\n', m.groupby('jenis').rec.nunique())


if __name__ == '__main__':
    main()
