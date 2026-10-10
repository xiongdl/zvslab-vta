"""Real model/WAV provenance and independent first-depthwise arithmetic."""
import importlib
import sys
from pathlib import Path
import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP.parent))


def test_first_depthwise_sample_is_reproducible_and_exact():
    sample_module = importlib.import_module('keyword_spotting_v1.python.dwc_sample')
    first = sample_module.extract_dwc_sample()
    second = sample_module.extract_dwc_sample()
    assert first.activation.shape == (1, 25, 5, 64)
    assert first.weight.shape == (3, 3, 64, 1)
    assert first.reference.shape == (1, 25, 5, 64)
    assert first.strides == (1, 1)
    assert first.padding == (1, 1, 1, 1)
    assert first.hashes == second.hashes
    assert set(first.hashes) == {'model', 'wav', 'activation', 'weight', 'reference'}
    for name in ('activation', 'weight', 'reference'):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert np.all(np.any(first.activation != 0, axis=(0, 1, 2)))
    assert np.all(np.any(first.weight != 0, axis=(0, 1, 3)))
    np.testing.assert_array_equal(first.reference, sample_module.scalar_reference(first.activation, first.weight, first.strides, first.padding))
