"""Uji AFM 03 — ringkasan keluaran/dalam_pasien.csv: AUROC pra-AF lawan jauh (dalam pasien) dan lawan
sehat, serta proporsi alarm menurut jarak dari AF; IK 95 % bootstrap pasien (B = 1000).
Pakai: python 03_ringkas.py"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, 'keluaran', 'dalam_pasien.csv'), dtype={'rec': str})
d['alarm'] = d.alarm.replace({'True': 1.0, 'False': 0.0, True: 1.0, False: 0.0}).astype(float)
rng = np.random.default_rng(0)


def auc_ci(pos, neg):
    """pos, neg: DataFrame dengan kolom rec, prob; bootstrap pasien gabungan."""
    x = pd.concat([pos.assign(y=1), neg.assign(y=0)])
    u = x.rec.unique(); g = {r: x[x.rec == r] for r in u}
    a = []
    for _ in range(1000):
        b = pd.concat([g[r] for r in rng.choice(u, len(u))])
        if b.y.nunique() == 2:
            a.append(roc_auc_score(b.y, b.prob))
    return roc_auc_score(x.y, x.prob), *np.percentile(a, [2.5, 97.5])


def alarm_ci(df):
    u = df.rec.unique(); g = {r: df[df.rec == r].alarm.to_numpy() for r in u}
    a = [np.concatenate([g[r] for r in rng.choice(u, len(u))]).mean() for _ in range(1000)]
    return df.alarm.mean(), *np.percentile(a, [2.5, 97.5])


sehat = d[d.jenis == 'sehat']
for db in ['shdb-af', 'ltafdb']:
    pra = d[(d.db == db) & (d.jenis == 'pra_af')]
    jauh = d[(d.db == db) & (d.jenis == 'jauh')]
    print(f'\n== {db}: pra_af {len(pra)} jendela/{pra.rec.nunique()} pasien; jauh {len(jauh)}/{jauh.rec.nunique()}')
    if len(pra) >= 10:
        print('AUROC pra_af lawan sehat     : %.3f [%.3f; %.3f]' % auc_ci(pra, sehat))
        print('AUROC pra_af lawan jauh (sama db): %.3f [%.3f; %.3f]' % auc_ci(pra, jauh))
        sama = jauh[jauh.rec.isin(pra.rec)]
        print(f'  hanya pasien yang punya keduanya ({sama.rec.nunique()} pasien): ' + '%.3f [%.3f; %.3f]' % auc_ci(pra[pra.rec.isin(sama.rec)], sama))
        print('AUROC jauh lawan sehat       : %.3f [%.3f; %.3f]' % auc_ci(jauh.assign(prob=jauh.prob), sehat))
    print('alarm pra_af : %.3f [%.3f; %.3f]' % alarm_ci(pra) if len(pra) >= 10 else f'alarm pra_af : {pra.alarm.mean():.3f} (n kecil)')
    for lo, hi in [(4, 6), (6, 12), (12, 24), (24, 1e9)]:
        j = jauh[(jauh.jarak_jam >= lo) & (jauh.jarak_jam < hi)]
        if len(j):
            print(f'alarm jauh {lo}-{hi if hi < 1e9 else "inf"} jam ({len(j)} jendela/{j.rec.nunique()} pasien): ' + '%.3f [%.3f; %.3f]' % alarm_ci(j))
print('\nalarm sehat nsrdb  : %.3f' % sehat[sehat.db == 'NSR_new'].alarm.mean(), ' nsr2db: %.3f' % sehat[sehat.db == 'NSRRR_new'].alarm.mean())
