import numpy as np
import pytest
import uncertainties as unc
import uncertainties.unumpy as unp

import pyexp.avg as avg


def test_avg():
    meas = np.arange(1, 7)
    print("Hello!")
    average = avg.weighted_avg(meas)
    assert unp.nominal_values(average) == 3.5
    assert unp.std_devs(average) == np.std(meas, ddof=1) / np.sqrt(len(meas))
