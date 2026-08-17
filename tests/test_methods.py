import numpy as np
import pytest
import torch

from whales.methods import (
    apply_chunked_standardization,
    apply_rolling_standardization,
)


def test_chunked_standardization_respects_nodata():
    nodata = -9999
    data = np.array(
        [[[1, 2, nodata], [3, 4, nodata], [5, 6, nodata]]],
        dtype=np.float32,
    )

    deviations = apply_chunked_standardization(
        data,
        step_size=3,
        nodata=nodata,
    )

    assert deviations.shape == data.shape
    assert np.all(deviations[data == nodata] == 0)
    assert np.all(np.isfinite(deviations))
    assert deviations[data != nodata].mean() == pytest.approx(0, abs=1e-6)


def test_rolling_standardization_runs_on_cpu_with_nodata():
    nodata = -9999
    data = np.arange(1, 50, dtype=np.float32).reshape(1, 7, 7)
    data[0, 0, 0] = nodata

    deviations = apply_rolling_standardization(
        data,
        device=torch.device("cpu"),
        patch_size=20,
        kernel_size=3,
        nodata=nodata,
    )

    assert deviations.shape == data.shape
    assert deviations[0, 0, 0] == 0
    assert np.all(np.isfinite(deviations))
