"""Unit and benchmark tests for functions in `speed_parameters.py`.

The expected matrices mirror the historical atmosphere speed references used in
`test_atmosphere.py` so behavior checks remain consistent while testing functions directly.
"""
#  This file is part of StdAtm
#  Copyright (C) 2023 ONERA & ISAE-SUPAERO
#  StdAtm is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..atmosphere import Atmosphere
from ..speed_parameters import (
    SEA_LEVEL_SPEED_OF_SOUND,
    _compute_cas_high_speed,
    _compute_cas_low_speed,
    _compute_subsonic_impact_pressure,
    _compute_supersonic_impact_pressure,
    _equation_cas_high_speed,
    compute_calibrated_airspeed,
    compute_dynamic_pressure,
    compute_equivalent_airspeed,
    compute_impact_pressure,
    compute_mach,
    compute_tas_from_eas,
    compute_tas_from_mach,
    compute_tas_from_pdyn,
    compute_tas_from_unit_re,
    compute_unitary_reynolds,
)

# Reference outputs for TAS/EAS/CAS/Mach and derived quantities across three altitudes.
EXPECTED_TAS = np.array(
    [
        [100.0, 100.0, 100.0],
        [200.0, 200.0, 200.0],
        [270.0, 270.0, 270.0],
        [300.0, 300.0, 300.0],
        [400.0, 400.0, 400.0],
        [800.0, 800.0, 800.0],
    ]
)
EXPECTED_EAS = np.array(
    [
        [100.0, 98.543, 55.666],
        [200.0, 197.085, 111.333],
        [270.0, 266.065, 150.299],
        [300.0, 295.628, 166.999],
        [400.0, 394.170, 222.666],
        [800.0, 788.341, 445.332],
    ]
)
EXPECTED_CAS = np.array(
    [
        [100.0, 98.580, 56.269],
        [200.0, 197.362, 116.073],
        [270.0, 266.698, 161.732],
        [300.0, 296.465, 182.507],
        [400.0, 395.578, 252.396],
        [800.0, 789.350, 479.567],
    ]
)
EXPECTED_MACH = np.array(
    [
        [0.29386, 0.29488, 0.33723],
        [0.58773, 0.58976, 0.67446],
        [0.79343, 0.79617, 0.91051],
        [0.88159, 0.88464, 1.01168],
        [1.17545, 1.17952, 1.34891],
        [2.35091, 2.35903, 2.69782],
    ]
)
EXPECTED_RE1 = np.array(
    [
        [6845941, 6683613, 2648139],
        [13691882, 13367227, 5296278],
        [18484040, 18045756, 7149975],
        [20537823, 20050840, 7944417],
        [27383763, 26734454, 10592556],
        [54767527, 53468908, 21185111],
    ]
)
EXPECTED_DYNAMIC_PRESSURE = np.array(
    [
        [6125.0, 5947.8, 1898.0],
        [24500.0, 23791.1, 7591.9],
        [44651.2, 43359.2, 13836.3],
        [55125.0, 53529.9, 17081.9],
        [97999.9, 95164.2, 30367.8],
        [391999.7, 380656.9, 121471.0],
    ]
)
EXPECTED_IMPACT_PRESSURE = np.array(
    [
        [6258.4, 6078.2, 1952.6],
        [26689.4, 25932.3, 8495.0],
        [52127.8, 50672.8, 16946.6],
        [66684.1, 64838.1, 21911.2],
        [135479.4, 131780.5, 44684.1],
        [668493.9, 649486.1, 210939.7],
    ]
)


@pytest.fixture(scope="session")
def speed_inputs():
    """State inputs at representative altitudes: sea level, 1 km, and 35 kft."""
    atm = Atmosphere([0.0, 1000.0, 35000.0])
    return {
        "speed_of_sound": np.asarray(atm.speed_of_sound),
        "density": np.asarray(atm.density),
        "kinematic_viscosity": np.asarray(atm.kinematic_viscosity),
        "pressure": np.asarray(atm.pressure),
    }


def _iter_indices(mask=None):
    """Yield `(row, col)` pairs over the expected grids, optionally filtered by mask."""
    if mask is None:
        mask = np.ones_like(EXPECTED_TAS, dtype=bool)
    for i, j in np.argwhere(mask):
        yield int(i), int(j)


def test_compute_tas_from_mach_scalar(speed_inputs):
    for i, j in _iter_indices():
        computed = compute_tas_from_mach(
            float(EXPECTED_MACH[i, j]), float(speed_inputs["speed_of_sound"][j])
        )
        assert EXPECTED_TAS[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_tas_from_mach_array(speed_inputs):
    computed = compute_tas_from_mach(EXPECTED_MACH, speed_inputs["speed_of_sound"])
    assert_allclose(computed, EXPECTED_TAS, rtol=1e-4)


def test_compute_tas_from_eas_scalar(speed_inputs):
    for i, j in _iter_indices():
        computed = compute_tas_from_eas(
            float(EXPECTED_EAS[i, j]), float(speed_inputs["density"][j])
        )
        assert EXPECTED_TAS[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_tas_from_eas_array(speed_inputs):
    computed = compute_tas_from_eas(EXPECTED_EAS, speed_inputs["density"])
    assert_allclose(computed, EXPECTED_TAS, rtol=1e-4)


def test_compute_tas_from_unit_re_scalar(speed_inputs):
    for i, j in _iter_indices():
        computed = compute_tas_from_unit_re(
            float(EXPECTED_RE1[i, j]), float(speed_inputs["kinematic_viscosity"][j])
        )
        assert EXPECTED_TAS[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_tas_from_unit_re_array(speed_inputs):
    computed = compute_tas_from_unit_re(EXPECTED_RE1, speed_inputs["kinematic_viscosity"])
    assert_allclose(computed, EXPECTED_TAS, rtol=1e-4)


def test_compute_tas_from_pdyn_scalar(speed_inputs):
    for i, j in _iter_indices():
        computed = compute_tas_from_pdyn(
            float(EXPECTED_DYNAMIC_PRESSURE[i, j]), float(speed_inputs["density"][j])
        )
        assert EXPECTED_TAS[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_tas_from_pdyn_array(speed_inputs):
    computed = compute_tas_from_pdyn(EXPECTED_DYNAMIC_PRESSURE, speed_inputs["density"])
    assert_allclose(computed, EXPECTED_TAS, rtol=1e-4)


def test_compute_mach_scalar(speed_inputs):
    for i, j in _iter_indices():
        computed = compute_mach(float(EXPECTED_TAS[i, j]), float(speed_inputs["speed_of_sound"][j]))
        assert EXPECTED_MACH[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_mach_array(speed_inputs):
    computed = compute_mach(EXPECTED_TAS, speed_inputs["speed_of_sound"])
    assert_allclose(computed, EXPECTED_MACH, rtol=1e-4)


def test_compute_equivalent_airspeed_scalar(speed_inputs):
    for i, j in _iter_indices():
        computed = compute_equivalent_airspeed(
            float(EXPECTED_TAS[i, j]), float(speed_inputs["density"][j])
        )
        assert EXPECTED_EAS[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_equivalent_airspeed_array(speed_inputs):
    computed = compute_equivalent_airspeed(EXPECTED_TAS, speed_inputs["density"])
    assert_allclose(computed, EXPECTED_EAS, rtol=1e-4)


def test_compute_unitary_reynolds_scalar(speed_inputs):
    for i, j in _iter_indices():
        computed = compute_unitary_reynolds(
            float(EXPECTED_TAS[i, j]), float(speed_inputs["kinematic_viscosity"][j])
        )
        assert EXPECTED_RE1[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_unitary_reynolds_array(speed_inputs):
    computed = compute_unitary_reynolds(EXPECTED_TAS, speed_inputs["kinematic_viscosity"])
    assert_allclose(computed, EXPECTED_RE1, rtol=1e-4)


def test_compute_dynamic_pressure_scalar(speed_inputs):
    for i, j in _iter_indices():
        computed = compute_dynamic_pressure(
            float(EXPECTED_TAS[i, j]), float(speed_inputs["density"][j])
        )
        assert EXPECTED_DYNAMIC_PRESSURE[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_dynamic_pressure_array(speed_inputs):
    computed = compute_dynamic_pressure(EXPECTED_TAS, speed_inputs["density"])
    assert_allclose(computed, EXPECTED_DYNAMIC_PRESSURE, rtol=1e-4)


def test_compute_subsonic_impact_pressure_scalar(speed_inputs):
    for i, j in _iter_indices(EXPECTED_MACH <= 1.0):
        computed = _compute_subsonic_impact_pressure(
            float(EXPECTED_MACH[i, j]), float(speed_inputs["pressure"][j])
        )
        assert EXPECTED_IMPACT_PRESSURE[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_subsonic_impact_pressure_array(speed_inputs):
    computed = _compute_subsonic_impact_pressure(
        EXPECTED_MACH[:3, :2], speed_inputs["pressure"][:2]
    )
    assert_allclose(computed, EXPECTED_IMPACT_PRESSURE[:3, :2], rtol=1e-4)


def test_compute_supersonic_impact_pressure_scalar(speed_inputs):
    for i, j in _iter_indices(EXPECTED_MACH > 1.0):
        computed = _compute_supersonic_impact_pressure(
            float(EXPECTED_MACH[i, j]), float(speed_inputs["pressure"][j])
        )
        assert EXPECTED_IMPACT_PRESSURE[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_supersonic_impact_pressure_array(speed_inputs):
    supersonic_mach = np.array([EXPECTED_MACH[3, 2], EXPECTED_MACH[4, 2], EXPECTED_MACH[5, 2]])
    computed = _compute_supersonic_impact_pressure(supersonic_mach, speed_inputs["pressure"][2])
    assert_allclose(computed, EXPECTED_IMPACT_PRESSURE[[3, 4, 5], 2], rtol=1e-4)


def test_compute_impact_pressure_scalar(speed_inputs):
    for i, j in _iter_indices():
        computed = compute_impact_pressure(
            float(EXPECTED_MACH[i, j]), float(speed_inputs["pressure"][j])
        )
        assert EXPECTED_IMPACT_PRESSURE[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_impact_pressure_array(speed_inputs):
    computed = compute_impact_pressure(EXPECTED_MACH, speed_inputs["pressure"])
    assert_allclose(computed, EXPECTED_IMPACT_PRESSURE, rtol=1e-4)


def test_compute_cas_low_speed_scalar():
    for i, j in _iter_indices(EXPECTED_CAS <= SEA_LEVEL_SPEED_OF_SOUND):
        computed = _compute_cas_low_speed(float(EXPECTED_IMPACT_PRESSURE[i, j]))
        assert EXPECTED_CAS[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_cas_low_speed_array():
    computed = _compute_cas_low_speed(EXPECTED_IMPACT_PRESSURE[:3])
    assert_allclose(computed, EXPECTED_CAS[:3], rtol=1e-4)


def test_equation_cas_high_speed_scalar():
    for i, j in _iter_indices(EXPECTED_CAS > SEA_LEVEL_SPEED_OF_SOUND):
        residual = _equation_cas_high_speed(
            float(EXPECTED_CAS[i, j]), float(EXPECTED_IMPACT_PRESSURE[i, j])
        )
        assert residual == pytest.approx(0.0, abs=1e-3)


def test_equation_cas_high_speed_array():
    residual = _equation_cas_high_speed(EXPECTED_CAS[5], EXPECTED_IMPACT_PRESSURE[5])
    assert_allclose(residual, np.zeros_like(residual), atol=1e-3)


def test_compute_cas_high_speed_scalar():
    for i, j in _iter_indices(EXPECTED_CAS > SEA_LEVEL_SPEED_OF_SOUND):
        computed = _compute_cas_high_speed(float(EXPECTED_IMPACT_PRESSURE[i, j]))
        assert EXPECTED_CAS[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_cas_high_speed_array():
    impact = np.array([EXPECTED_IMPACT_PRESSURE[5, 0]] * 3)
    computed = _compute_cas_high_speed(impact)
    assert_allclose(computed, np.array([EXPECTED_CAS[5, 0]] * 3), rtol=1e-4)


def test_compute_calibrated_airspeed_scalar():
    for i, j in _iter_indices():
        computed = compute_calibrated_airspeed(float(EXPECTED_IMPACT_PRESSURE[i, j]))
        assert EXPECTED_CAS[i, j] == pytest.approx(computed, rel=1e-4)


def test_compute_calibrated_airspeed_array():
    computed = compute_calibrated_airspeed(EXPECTED_IMPACT_PRESSURE)
    assert_allclose(computed, EXPECTED_CAS, rtol=1e-4)


@pytest.fixture(scope="session")
def benchmark_speed_inputs(speed_inputs):
    """Pre-tiled arrays used to exercise vectorized code paths in benchmark tests."""
    repeat = 20
    return {
        "mach": np.tile(EXPECTED_MACH, (repeat, 1)),
        "tas": np.tile(EXPECTED_TAS, (repeat, 1)),
        "eas": np.tile(EXPECTED_EAS, (repeat, 1)),
        "cas": np.tile(EXPECTED_CAS, (repeat, 1)),
        "re1": np.tile(EXPECTED_RE1, (repeat, 1)),
        "dynamic_pressure": np.tile(EXPECTED_DYNAMIC_PRESSURE, (repeat, 1)),
        "impact_pressure": np.tile(EXPECTED_IMPACT_PRESSURE, (repeat, 1)),
        "speed_of_sound": np.tile(speed_inputs["speed_of_sound"], (6 * repeat, 1)),
        "density": np.tile(speed_inputs["density"], (6 * repeat, 1)),
        "kinematic_viscosity": np.tile(speed_inputs["kinematic_viscosity"], (6 * repeat, 1)),
        "pressure": np.tile(speed_inputs["pressure"], (6 * repeat, 1)),
    }


def test_performances_array_compute_tas_from_mach(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_tas_from_mach(
            benchmark_speed_inputs["mach"], benchmark_speed_inputs["speed_of_sound"]
        )

    benchmark(func)


def test_performances_scalar_compute_tas_from_mach(benchmark, speed_inputs):
    def func():
        for i in range(EXPECTED_MACH.shape[0]):
            for j in range(EXPECTED_MACH.shape[1]):
                _ = compute_tas_from_mach(
                    float(EXPECTED_MACH[i, j]), float(speed_inputs["speed_of_sound"][j])
                )

    benchmark(func)


def test_performances_array_compute_tas_from_eas(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_tas_from_eas(benchmark_speed_inputs["eas"], benchmark_speed_inputs["density"])

    benchmark(func)


def test_performances_scalar_compute_tas_from_eas(benchmark, speed_inputs):
    def func():
        for i in range(EXPECTED_EAS.shape[0]):
            for j in range(EXPECTED_EAS.shape[1]):
                _ = compute_tas_from_eas(
                    float(EXPECTED_EAS[i, j]), float(speed_inputs["density"][j])
                )

    benchmark(func)


def test_performances_array_compute_tas_from_unit_re(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_tas_from_unit_re(
            benchmark_speed_inputs["re1"], benchmark_speed_inputs["kinematic_viscosity"]
        )

    benchmark(func)


def test_performances_scalar_compute_tas_from_unit_re(benchmark, speed_inputs):
    def func():
        for i in range(EXPECTED_RE1.shape[0]):
            for j in range(EXPECTED_RE1.shape[1]):
                _ = compute_tas_from_unit_re(
                    float(EXPECTED_RE1[i, j]), float(speed_inputs["kinematic_viscosity"][j])
                )

    benchmark(func)


def test_performances_array_compute_tas_from_pdyn(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_tas_from_pdyn(
            benchmark_speed_inputs["dynamic_pressure"], benchmark_speed_inputs["density"]
        )

    benchmark(func)


def test_performances_scalar_compute_tas_from_pdyn(benchmark, speed_inputs):
    def func():
        for i in range(EXPECTED_DYNAMIC_PRESSURE.shape[0]):
            for j in range(EXPECTED_DYNAMIC_PRESSURE.shape[1]):
                _ = compute_tas_from_pdyn(
                    float(EXPECTED_DYNAMIC_PRESSURE[i, j]), float(speed_inputs["density"][j])
                )

    benchmark(func)


def test_performances_array_compute_mach(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_mach(benchmark_speed_inputs["tas"], benchmark_speed_inputs["speed_of_sound"])

    benchmark(func)


def test_performances_scalar_compute_mach(benchmark, speed_inputs):
    def func():
        for i in range(EXPECTED_TAS.shape[0]):
            for j in range(EXPECTED_TAS.shape[1]):
                _ = compute_mach(
                    float(EXPECTED_TAS[i, j]), float(speed_inputs["speed_of_sound"][j])
                )

    benchmark(func)


def test_performances_array_compute_equivalent_airspeed(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_equivalent_airspeed(
            benchmark_speed_inputs["tas"], benchmark_speed_inputs["density"]
        )

    benchmark(func)


def test_performances_scalar_compute_equivalent_airspeed(benchmark, speed_inputs):
    def func():
        for i in range(EXPECTED_TAS.shape[0]):
            for j in range(EXPECTED_TAS.shape[1]):
                _ = compute_equivalent_airspeed(
                    float(EXPECTED_TAS[i, j]), float(speed_inputs["density"][j])
                )

    benchmark(func)


def test_performances_array_compute_unitary_reynolds(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_unitary_reynolds(
            benchmark_speed_inputs["tas"], benchmark_speed_inputs["kinematic_viscosity"]
        )

    benchmark(func)


def test_performances_scalar_compute_unitary_reynolds(benchmark, speed_inputs):
    def func():
        for i in range(EXPECTED_TAS.shape[0]):
            for j in range(EXPECTED_TAS.shape[1]):
                _ = compute_unitary_reynolds(
                    float(EXPECTED_TAS[i, j]), float(speed_inputs["kinematic_viscosity"][j])
                )

    benchmark(func)


def test_performances_array_compute_dynamic_pressure(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_dynamic_pressure(
            benchmark_speed_inputs["tas"], benchmark_speed_inputs["density"]
        )

    benchmark(func)


def test_performances_scalar_compute_dynamic_pressure(benchmark, speed_inputs):
    def func():
        for i in range(EXPECTED_TAS.shape[0]):
            for j in range(EXPECTED_TAS.shape[1]):
                _ = compute_dynamic_pressure(
                    float(EXPECTED_TAS[i, j]), float(speed_inputs["density"][j])
                )

    benchmark(func)


def test_performances_array_compute_subsonic_impact_pressure(benchmark_speed_inputs, benchmark):
    def func():
        _ = _compute_subsonic_impact_pressure(
            benchmark_speed_inputs["mach"][:300, :2], benchmark_speed_inputs["pressure"][:300, :2]
        )

    benchmark(func)


def test_performances_scalar_compute_subsonic_impact_pressure(benchmark, speed_inputs):
    def func():
        for i in range(3):
            for j in range(2):
                _ = _compute_subsonic_impact_pressure(
                    float(EXPECTED_MACH[i, j]), float(speed_inputs["pressure"][j])
                )

    benchmark(func)


def test_performances_array_compute_supersonic_impact_pressure(benchmark_speed_inputs, benchmark):
    def func():
        _ = _compute_supersonic_impact_pressure(
            benchmark_speed_inputs["mach"][100:, 2], benchmark_speed_inputs["pressure"][100:, 2]
        )

    benchmark(func)


def test_performances_scalar_compute_supersonic_impact_pressure(benchmark, speed_inputs):
    def func():
        for i in [3, 4, 5]:
            _ = _compute_supersonic_impact_pressure(
                float(EXPECTED_MACH[i, 2]), float(speed_inputs["pressure"][2])
            )

    benchmark(func)


def test_performances_array_compute_impact_pressure(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_impact_pressure(
            benchmark_speed_inputs["mach"], benchmark_speed_inputs["pressure"]
        )

    benchmark(func)


def test_performances_scalar_compute_impact_pressure(benchmark, speed_inputs):
    def func():
        for i in range(EXPECTED_MACH.shape[0]):
            for j in range(EXPECTED_MACH.shape[1]):
                _ = compute_impact_pressure(
                    float(EXPECTED_MACH[i, j]), float(speed_inputs["pressure"][j])
                )

    benchmark(func)


def test_performances_array_compute_cas_low_speed(benchmark_speed_inputs, benchmark):
    def func():
        _ = _compute_cas_low_speed(benchmark_speed_inputs["impact_pressure"][:600])

    benchmark(func)


def test_performances_scalar_compute_cas_low_speed(benchmark):
    def func():
        for i in range(4):
            for j in range(EXPECTED_IMPACT_PRESSURE.shape[1]):
                _ = _compute_cas_low_speed(float(EXPECTED_IMPACT_PRESSURE[i, j]))

    benchmark(func)


def test_performances_array_equation_cas_high_speed(benchmark_speed_inputs, benchmark):
    def func():
        _ = _equation_cas_high_speed(
            benchmark_speed_inputs["cas"][500:], benchmark_speed_inputs["impact_pressure"][500:]
        )

    benchmark(func)


def test_performances_scalar_equation_cas_high_speed(benchmark):
    def func():
        for j in range(EXPECTED_CAS.shape[1]):
            _ = _equation_cas_high_speed(
                float(EXPECTED_CAS[5, j]), float(EXPECTED_IMPACT_PRESSURE[5, j])
            )

    benchmark(func)


def test_performances_array_compute_cas_high_speed(benchmark_speed_inputs, benchmark):
    def func():
        _ = _compute_cas_high_speed(benchmark_speed_inputs["impact_pressure"][50::10])

    benchmark(func)


def test_performances_scalar_compute_cas_high_speed(benchmark):
    def func():
        for j in range(EXPECTED_IMPACT_PRESSURE.shape[1]):
            _ = _compute_cas_high_speed(float(EXPECTED_IMPACT_PRESSURE[5, j]))

    benchmark(func)


def test_performances_array_compute_calibrated_airspeed(benchmark_speed_inputs, benchmark):
    def func():
        _ = compute_calibrated_airspeed(benchmark_speed_inputs["impact_pressure"][::10])

    benchmark(func)


def test_performances_scalar_compute_calibrated_airspeed(benchmark):
    def func():
        for i in range(EXPECTED_IMPACT_PRESSURE.shape[0]):
            for j in range(EXPECTED_IMPACT_PRESSURE.shape[1]):
                _ = compute_calibrated_airspeed(float(EXPECTED_IMPACT_PRESSURE[i, j]))

    benchmark(func)
