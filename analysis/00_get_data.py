"""Fetch everything the analyses need into data/ (or $AFWP_DATA). Nothing here is redistributed.

  data/ltafdb, data/shdb-af, data/nsr2db   annotations, fetched by retrained/01_jendela.py itself
  data/nsrdb                               headers and beat annotations (.hea, .atr), fetched here
  data/AF_Mamba                            the authors' repository (trained weights, subject folds),
                                           cloned here; it has no licence and is used only to load weights
  data/af_mamba_zenodo                     their processed RR windows, doi:10.5281/zenodo.22572206 (ODC-By)
  data/iridia-af                           IRIDIA-AF v1.0.1, doi:10.5281/zenodo.8405941 (CC BY 4.0):
                                           download iridia-af-records-v1.0.1.zip (10.5 GB) and the metadata
                                           CSV by hand, then run this script with --iridia ZIP to extract
                                           only the RR files (*_rr_*.h5, *_rr_labels.csv, about 44 MB)

Usage: python 00_get_data.py [--iridia path/to/iridia-af-records-v1.0.1.zip]
"""
import os
import subprocess
import sys
import urllib.request
import zipfile

import wfdb

DATA_ROOT = os.environ.get('AFWP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    nsr = os.path.join(DATA_ROOT, 'nsrdb')
    os.makedirs(nsr, exist_ok=True)
    rec = wfdb.get_record_list('nsrdb')
    need = [f'{r}.{e}' for r in rec for e in ('hea', 'atr') if not os.path.exists(os.path.join(nsr, f'{r}.{e}'))]
    if need:
        wfdb.dl_files('nsrdb', nsr, need)
    repo = os.path.join(DATA_ROOT, 'AF_Mamba')
    if not os.path.isdir(repo):
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/yongbin98/AF_Mamba.git', repo], check=True)
    z = os.path.join(DATA_ROOT, 'af_mamba_zenodo')
    os.makedirs(z, exist_ok=True)
    f = os.path.join(z, 'structured_dataset_1hz.pt')
    if not os.path.exists(f):
        urllib.request.urlretrieve('https://zenodo.org/records/22572206/files/structured_dataset_1hz.pt?download=1', f)
    if '--iridia' in sys.argv:
        zp = sys.argv[sys.argv.index('--iridia') + 1]
        out = os.path.join(DATA_ROOT, 'iridia-af')
        with zipfile.ZipFile(zp) as zz:
            zz.extractall(out, members=[n for n in zz.namelist() if '_rr_' in n])
        print('IRIDIA-AF RR files extracted to', out, '- put iridia-af-metadata-v1.0.1.csv there too')
    print('data ready in', os.path.abspath(DATA_ROOT))


if __name__ == '__main__':
    main()
