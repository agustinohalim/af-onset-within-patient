"""Artikel EKG 2 — apakah sisa kecil AUC dalam-pasien berasal dari jam dalam sehari?

Hanya LTAF: header SHDB-AF tidak memuat jam mulai, sedangkan 84 rekaman LTAF punya jam mulai
yang berbeda-beda (bukan tempat-isi seperti sddb). Jam jendela = jam mulai rekaman + mulai_s.

  J1 jam saja      model logistik dengan satu-satunya masukan sin/cos jam, rancangan D2 biasa.
                   Bila AUC-nya setara dengan D2 berciri HRV, sisa itu penjelasnya jam.
  J2 HRV dipadankan D2 dengan kontrol jauh yang dipilih pada jam yang sama (+-60 menit, melingkar)
                   dengan salah satu jendela pra-AF pasien itu.
  J3 HRV biasa     D2 tanpa pemadanan, hanya LTAF, sebagai pembanding J2.

Horizon 0 dan 240 menit. Pembagian dan selang sama dengan 02_uji_af.py.

Keluaran: keluaran/hasil_jam.csv dan ringkasan di layar.
Pakai:    python 05_jam.py
"""
import os
from importlib import import_module

import numpy as np
import pandas as pd
import wfdb
DATA_ROOT = os.environ.get('AFWP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))

u = import_module('02_uji_af')
HERE, OUT = u.HERE, u.OUT
PADAN = 60.0   # menit


def jam_mulai():
    folder = os.path.join(DATA_ROOT, 'ltafdb')
    out = {}
    for f in os.listdir(folder):
        if f.endswith('.hea'):
            t = wfdb.rdheader(os.path.join(folder, f[:-4])).base_time
            if t is not None:
                out[f[:-4]] = t.hour * 60 + t.minute + t.second / 60
    return out


def jarak_jam(a, b):
    d = np.abs(a - b) % 1440
    return np.minimum(d, 1440 - d)


def main():
    f = u.siapkan(pd.read_csv(os.path.join(OUT, 'fitur_af.csv'), dtype={'rec': str}))
    mulai = jam_mulai()
    f = f[(f.db == 'ltafdb') & f.rec.isin(mulai)].copy()
    f['menit_jam'] = (f.rec.map(mulai) + f.mulai_s / 60 + 2.5) % 1440
    f['jam_sin'] = np.sin(2 * np.pi * f.menit_jam / 1440)
    f['jam_cos'] = np.cos(2 * np.pi * f.menit_jam / 1440)
    rng = np.random.default_rng(2026)
    baris = []
    for h in (0, 240):
        pos = u.positif(f, h)
        far = u.jauh(f, pos.subj.unique(), max(u.JAUH, h + 90))   # tidak bertumpang dengan positif (revisi 27 Sep 2026)
        biasa = u.ambil(far, u.PER, rng)
        padan = []
        for s, g in far.groupby('subj'):
            ref = pos[pos.subj == s].menit_jam.to_numpy()
            if not len(ref):
                continue
            ok = g[g.menit_jam.apply(lambda m: jarak_jam(m, ref).min() <= PADAN)]
            if len(ok):
                padan.append(ok.sample(min(u.PER, len(ok)), random_state=int(rng.integers(1 << 30))))
        padan = pd.concat(padan)
        for nama, neg, ciri, jenis in (('J1_jam_saja', biasa, ['jam_sin', 'jam_cos'], 'logistik'),
                                       ('J3_HRV_biasa', biasa, u.CIRI, 'logistik'),
                                       ('J3_HRV_biasa', biasa, u.CIRI, 'rf'),
                                       ('J2_HRV_padan_jam', padan, u.CIRI, 'logistik'),
                                       ('J2_HRV_padan_jam', padan, u.CIRI, 'rf')):
            d = pd.concat([pos[pos.subj.isin(neg.subj.unique())].assign(y=1), neg.assign(y=0)])
            if d.subj.nunique() < 5:
                print(f'h={h:4d} {nama:17s} {jenis:8s} dilewati: hanya {d.subj.nunique()} pasien', flush=True)
                baris.append(dict(uji=nama, model=jenis, h=h, n_pasien=d.subj.nunique()))
                continue
            r = u.boot_auc(d, u.oof(d, jenis, ciri), 'jendela', rng)
            baris.append(dict(uji=nama, model=jenis, h=h, auc=r[0], lo=r[1], hi=r[2], n_pasien=r[3], n_jendela=r[4]))
            print(f'h={h:4d} {nama:17s} {jenis:8s} AUC {r[0]:.3f} [{r[1]:.2f},{r[2]:.2f}] pasien={r[3]}', flush=True)
    pd.DataFrame(baris).to_csv(os.path.join(OUT, 'hasil_jam.csv'), index=False)


if __name__ == '__main__':
    main()
