"""Muat berkas .pt pihak ketiga dengan unpickler terbatas torch (weights_only=True) — hanya kelas
numpy yang dibutuhkan larik yang diizinkan; tidak ada eksekusi kode sembarang."""
import numpy as np
import torch


def muat(path):
    izin = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np._core.multiarray.scalar]
    izin += [type(np.dtype(t)) for t in ('float64', 'float32', 'int64', 'int32', 'int16', 'uint8', 'bool', 'U1', 'O')]
    with torch.serialization.safe_globals(izin):
        return torch.load(path, weights_only=True)
