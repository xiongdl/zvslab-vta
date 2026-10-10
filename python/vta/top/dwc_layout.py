"""Native DwC DMA layout and subvector source addresses.

Input DMA vectors retain BATCH x BLOCK_IN elements. Spatial points contain
consecutive channel vectors; compute sources count BATCH x min(BI, BO) slices.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DWCInput:
    """Packed input [N/B, H, W, C/BI, B, BI] and its address contract."""
    data: np.ndarray
    logical_shape: tuple
    env: object

    def unpack(self):
        n, c, h, w = self.logical_shape
        return self.data.transpose(0, 4, 3, 5, 1, 2).reshape(n, c, h, w).copy()

    def source_index(self, h, w, channel_block, batch_group=0, base_vector=0):
        """DwC source for one BO-channel block; base_vector is a DMA index."""
        n, c, height, width = self.logical_shape
        e = self.env
        if not (0 <= h < height and 0 <= w < width and
                0 <= channel_block < c // e.BLOCK_OUT and 0 <= batch_group < n // e.BATCH):
            raise IndexError("DwC input coordinate outside logical shape")
        if base_vector < 0 or base_vector * e.INP_SLICES % e.INP_BANKS_PER_BATCH:
            raise ValueError("DwC DMA base must align to a complete parallel channel group")
        return (base_vector * e.INP_SLICES +
                ((batch_group * height + h) * width + w) * (c // e.INP_BANK_LANES) +
                channel_block * (e.BLOCK_OUT // e.INP_BANK_LANES))

    def bank_address(self, vector, batch=0, slice_index=0, vme_bits=64):
        """Physical (bank,row), including extra row stripes for wider VME buses."""
        e = self.env
        if not (0 <= vector < e.INP_BUFF_DEPTH and 0 <= batch < e.BATCH and
                0 <= slice_index < e.INP_SLICES):
            raise IndexError("Input scratchpad coordinate outside capacity")
        stripes = max(1, vme_bits // e.INP_PARALLEL_BITS)
        s = vector * e.INP_SLICES + slice_index
        banks = e.INP_BANKS_PER_BATCH
        return (batch * banks * stripes + s % (banks * stripes), s // (banks * stripes))


def pack_dwc_input(data, env):
    """Pack NCHW int input without channel padding; spatial padding is caller-owned."""
    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError("DwC input must be NCHW")
    n, c, h, w = data.shape
    if not all(data.shape) or n % env.BATCH or c % max(env.BLOCK_IN, env.BLOCK_OUT):
        raise ValueError("DwC requires complete batch and parallel channel groups")
    if data.dtype != np.dtype(env.inp_dtype):
        raise ValueError("DwC input dtype must match environment")
    packed = data.reshape(n // env.BATCH, env.BATCH, c // env.BLOCK_IN,
                          env.BLOCK_IN, h, w).transpose(0, 4, 5, 2, 1, 3)
    return DWCInput(np.ascontiguousarray(packed), data.shape, env)


def pack_dwc_weight(kernel, env):
    """Pack C,KH,KW into [C/BO,ceil(KH*KW/BI),BO,BI], low tap first."""
    kernel = np.asarray(kernel)
    if kernel.ndim != 3 or not all(kernel.shape) or kernel.shape[0] % env.BLOCK_OUT:
        raise ValueError("DwC weights require C,KH,KW and complete output channel blocks")
    if kernel.dtype != np.dtype(env.wgt_dtype):
        raise ValueError("DwC weight dtype must match environment")
    c, kh, kw = kernel.shape
    entries = (kh * kw + env.BLOCK_IN - 1) // env.BLOCK_IN
    padded = np.zeros((c, entries * env.BLOCK_IN), dtype=kernel.dtype)
    padded[:, :kh * kw] = kernel.reshape(c, kh * kw)
    return np.ascontiguousarray(padded.reshape(c // env.BLOCK_OUT, env.BLOCK_OUT,
                                              entries, env.BLOCK_IN).transpose(0, 2, 1, 3))
