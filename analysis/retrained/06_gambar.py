"""Artikel EKG 2 — gambar 1: AUC terhadap horizon sebelum awitan AF, dua rancangan, dua model.

Warna = rancangan (biru antarbasis-data, jingga dalam-pasien; dua slot pertama palet kategoris
yang lolos validator dataviz). Garis penuh dan bulatan = logistik atas ciri HRV 5 menit (pita
selang 95 %); garis putus dan belah ketupat = CNN 1D atas deret RR satu jam. Sumbu mendatar
berjarak sama per horizon.

Keluaran: Gambar/gambar_ekg2_horizon.png (300 dpi) dan .pdf
Pakai:    python 06_gambar.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'keluaran')
GAMBAR = os.path.join(HERE, 'Gambar')
PERMUKAAN, TEKS, TEKS2, KISI = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BIRU, JINGGA = '#2a78d6', '#eb6834'
HOR = [0, 30, 60, 120, 240]


def main():
    os.makedirs(GAMBAR, exist_ok=True)
    h = pd.read_csv(os.path.join(OUT, 'hasil_af.csv'))
    c = pd.read_csv(os.path.join(OUT, 'hasil_cnn.csv'))
    h = h[h.model == 'logistik']
    pos = {v: i for i, v in enumerate(HOR)}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=PERMUKAAN)
    ax.set_facecolor(PERMUKAAN)
    for uji, cnn_uji, warna, label in (('D1_antarbasis', 'D1_antarbasis_cnn', BIRU, 'AF patients vs healthy\nfrom other databases'),
                                       ('D2_dalam_pasien', 'D2_dalam_pasien_cnn', JINGGA, 'same patient,\nsinus far from AF')):
        d = h[h.uji == uji].sort_values('h')
        x = d.h.map(pos)
        ax.fill_between(x, d.lo, d.hi, color=warna, alpha=0.10, linewidth=0)
        ax.plot(x, d.auc, color=warna, linewidth=2, marker='o', markersize=8,
                markeredgecolor=PERMUKAAN, markeredgewidth=2, zorder=3)
        e = c[c.uji == cnn_uji].sort_values('h')
        ax.plot(e.h.map(pos), e.auc, color=warna, linewidth=1.5, linestyle=(0, (4, 3)), marker='D',
                markersize=7, markeredgecolor=PERMUKAAN, markeredgewidth=1.5, zorder=3)
        ax.annotate(label, (x.iloc[-1], d.auc.iloc[-1]), xytext=(10, 0), textcoords='offset points',
                    va='center', fontsize=9, color=TEKS)
    ax.axhline(0.5, color=TEKS2, linewidth=1, linestyle=(0, (1, 2)))
    ax.text(0.05, 0.505, 'chance', fontsize=8.5, color=TEKS2, va='bottom')
    ax.plot([], [], color=TEKS2, linewidth=2, marker='o', label='HRV features, logistic regression')
    ax.plot([], [], color=TEKS2, linewidth=1.5, linestyle=(0, (4, 3)), marker='D', label='1-h RR sequence, 1D CNN')
    ax.legend(loc='lower left', fontsize=8.5, frameon=False, labelcolor=TEKS)
    n = h[h.uji == 'D1_antarbasis'].sort_values('h')
    ax.set_xticks(range(len(HOR)))
    ax.set_xticklabels([f'{v / 60:g}\nn={k}' for v, k in zip(n.h, n.n_pos.astype(int))], fontsize=8.5, color=TEKS2)
    ax.set_xlabel('Hours before AF onset (n = AF patients, HRV design)', fontsize=9.5, color=TEKS)
    ax.set_ylabel('AUC, patient-level split', fontsize=9.5, color=TEKS)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlim(-0.3, len(HOR) - 0.7)
    ax.grid(axis='y', color=KISI, linewidth=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(KISI)
    ax.tick_params(colors=TEKS2, labelsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.70, bottom=0.2, top=0.95)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(GAMBAR, f'gambar_ekg2_horizon.{ext}'), dpi=300, facecolor=PERMUKAAN)
    print('tersimpan')


if __name__ == '__main__':
    main()
