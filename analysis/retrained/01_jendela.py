"""Artikel EKG 2 — jendela 5 menit sinus dari pasien AF paroksismal dan subjek sehat.

Hanya anotasi yang diunduh dari PhysioNet (semua ODC-By 1.0), tanpa sinyal mentah:

  shdb-af  98 rekaman beranotasi irama (.atr, aux '(AFIB' dan seterusnya) + R-peak (.qrs)
  ltafdb   84 rekaman, denyut dan irama di .atr
  nsrdb    18 subjek sehat, .atr (diunduh analysis/00_get_data.py ke data/nsrdb)
  nsr2db   54 subjek sehat, .ecg

Awitan AF = awal episode AF atau flutter yang berlangsung >= 60 detik. Semburan lebih pendek tidak
dihitung sebagai awitan, tetapi jendela yang bertumpang dengannya tetap dibuang.

Setiap jendela 5 menit yang seluruhnya sinus (tidak bertumpang dengan AF, AFL, AT, SVTA, atau
irama non-sinus lain) disimpan dengan:
  menit_ke_af     jarak ujung jendela ke awitan AF berikutnya (NaN bila tidak ada)
  menit_dari_af   jarak minimum ke episode AF mana pun, sebelum atau sesudah

Ciri HRV sama dengan artikel 1 kecuali proporsi denyut ektopik: .qrs SHDB-AF tidak memuat jenis
denyut, jadi ciri itu sendiri akan menjadi sidik basis data. Semua jarak RR antara dua denyut
berurutan dipakai (0,3 s < RR < 2 s), bukan hanya N-N, supaya keempat basis data diperlakukan sama.

Keluaran: keluaran/fitur_af.csv
Pakai:    python 01_jendela.py
"""
import os

import numpy as np
import pandas as pd
import wfdb
DATA_ROOT = os.environ.get('AFWP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = DATA_ROOT
OUT = os.path.join(HERE, 'keluaran')
NSRDB = os.path.join(DATA_ROOT, 'nsrdb')
JENDELA = 300.0
NON_SINUS = ('(AFIB', '(AFL', '(AT', '(SVTA', '(PAT', '(NOD', '(J', '(VT', '(B', '(T', '(IVR', '(P')
AF = ('(AFIB', '(AFL')
MIN_RR = 150
MIN_AF = 60.0   # detik; episode AF lebih pendek tidak dihitung awitan (seperti LTAF), tetapi tetap dibuang dari jendela sinus
FREK = np.linspace(0.01, 0.5, 200)

SUMBER = {
    'shdb-af': {'versi': '1.0.1', 'beat': 'qrs', 'ritme': 'atr', 'wajib_ritme': True},
    'ltafdb': {'versi': '1.0.0', 'beat': 'atr', 'ritme': 'atr', 'wajib_ritme': True},
    'nsr2db': {'versi': '1.0.0', 'beat': 'ecg', 'ritme': None, 'wajib_ritme': False},
}


def ciri(t):
    from scipy.signal import lombscargle
    rr = np.diff(t)
    ok = (rr > 0.3) & (rr < 2.0)
    nn, tn = rr[ok], t[1:][ok]
    if len(nn) < MIN_RR:
        return None
    d = np.diff(nn)
    p = lombscargle(tn - tn[0], nn - nn.mean(), 2 * np.pi * FREK, normalize=False)
    df = FREK[1] - FREK[0]
    lf = p[(FREK >= 0.04) & (FREK < 0.15)].sum() * df
    hf = p[(FREK >= 0.15) & (FREK < 0.4)].sum() * df
    return {'mean_nn': nn.mean(), 'sdnn': nn.std(), 'rmssd': np.sqrt(np.mean(d ** 2)),
            'pnn50': np.mean(np.abs(d) > 0.05), 'cv_nn': nn.std() / nn.mean(),
            'lf': lf, 'hf': hf, 'lf_hf': lf / hf if hf > 0 else np.nan,
            'sd1': np.sqrt(np.var(d) / 2), 'sd2': np.sqrt(max(2 * np.var(nn) - np.var(d) / 2, 0))}


def unduh(db, cfg):
    tujuan = os.path.join(DATA, db)
    os.makedirs(tujuan, exist_ok=True)
    if db == 'shdb-af':
        # SHDB-AF tidak punya berkas RECORDS; daftar rekaman dibaca dari halaman direktorinya.
        import re
        import urllib.request
        url = f'https://physionet.org/files/{db}/{cfg["versi"]}/'
        html = urllib.request.urlopen(url, timeout=60).read().decode()
        rek = sorted(set(re.findall(r'href="(\d+)\.hea"', html)))
    else:
        rek = wfdb.get_record_list(db)
    ext = {'hea', cfg['beat']} | ({cfg['ritme']} if cfg['ritme'] else set())
    perlu = [f'{r}.{e}' for r in rek for e in ext if not os.path.exists(os.path.join(tujuan, f'{r}.{e}'))]
    if perlu:
        try:
            wfdb.dl_files(db, tujuan, perlu)
        except Exception:
            # Sebagian rekaman SHDB-AF tidak punya .atr (tidak beranotasi); unduh satu per satu.
            for f in perlu:
                try:
                    wfdb.dl_files(db, tujuan, [f])
                except Exception:
                    pass
    return tujuan, rek


def episode(ann):
    """Selang [mulai, akhir) detik untuk irama non-sinus dan untuk AF, dari aux_note."""
    t = ann.sample / ann.fs
    ubah = [(ti, a.strip('\x00')) for ti, a in zip(t, ann.aux_note) if a and a.startswith('(')]
    non, af = [], []
    for k, (ti, a) in enumerate(ubah):
        akhir = ubah[k + 1][0] if k + 1 < len(ubah) else np.inf
        if a.startswith(NON_SINUS):
            non.append((ti, akhir))
        if a.startswith(AF) and akhir - ti >= MIN_AF:
            af.append((ti, akhir))
    return non, af


def jendela(db, rec, t_beat, dur, non, af):
    baris = []
    awal_af = np.array([a for a, _ in af]) if af else np.array([])
    for a in np.arange(0, dur - JENDELA + 1, JENDELA):
        b = a + JENDELA
        if any(s < b and e > a for s, e in non):
            continue
        c = ciri(t_beat[(t_beat >= a) & (t_beat < b)])
        if c is None:
            continue
        nxt = awal_af[awal_af >= b]
        c.update({'db': db, 'rec': rec, 'mulai_s': a,
                  'menit_ke_af': (nxt.min() - b) / 60 if len(nxt) else np.nan,
                  'menit_dari_af': min((min(abs(s - b), abs(a - e)) for s, e in af), default=np.nan) / 60,
                  'pasien_af': int(bool(af))})
        baris.append(c)
    return baris


def main():
    os.makedirs(OUT, exist_ok=True)
    semua = []
    for db, cfg in SUMBER.items():
        folder, rek = unduh(db, cfg)
        for r in rek:
            path = os.path.join(folder, r)
            if not os.path.exists(path + '.' + cfg['beat']):
                continue
            if cfg['wajib_ritme'] and not os.path.exists(path + '.' + cfg['ritme']):
                continue
            h = wfdb.rdheader(path)
            beat = wfdb.rdann(path, cfg['beat'])
            sym = np.array(beat.symbol)
            t_beat = beat.sample[np.isin(sym, list('NLRBAaJSVrFejnE/fQ'))] / h.fs
            non, af = episode(wfdb.rdann(path, cfg['ritme'])) if cfg['ritme'] else ([], [])
            b = jendela(db, r, t_beat, h.sig_len / h.fs if h.sig_len else t_beat.max(), non, af)
            semua += b
            print(f'{db:8s} {r:6s} {len(af):3d} episode AF  {len(b):4d} jendela sinus', flush=True)
    for p in sorted(os.listdir(NSRDB)):
        if p.endswith('.hea'):
            r = p[:-4]
            h = wfdb.rdheader(os.path.join(NSRDB, r))
            beat = wfdb.rdann(os.path.join(NSRDB, r), 'atr')
            sym = np.array(beat.symbol)
            t_beat = beat.sample[np.isin(sym, list('NLRBAaJSVrFejnE/fQ'))] / h.fs
            semua += jendela('nsrdb', r, t_beat, h.sig_len / h.fs, [], [])
    f = pd.DataFrame(semua)
    f.to_csv(os.path.join(OUT, 'fitur_af.csv'), index=False)
    print(f'\n{len(f)} jendela sinus;', f.groupby('db').rec.nunique().to_dict(), 'rekaman')


if __name__ == '__main__':
    main()
