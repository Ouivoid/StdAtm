"""Tests for `Atmosphere` and `AtmosphereSI` class behavior.

Ownership note:
- low-level state/speed formulas and their scalar/array benchmarks are covered in
  `test_state_parameters.py` and `test_speed_parameters.py`
- this file focuses on what happens once those formulas are wired into the
  `Atmosphere` classes: unit handling, shape/broadcast semantics, speed setter
  workflows, cache/re-ask behavior, and end-to-end class benchmarks
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

from typing import ClassVar

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import foot

from ..atmosphere import Atmosphere, AtmosphereSI


class Checker:
    """Reference class-level conversion results reused across integration scenarios."""

    expected_tas: ClassVar[np.ndarray] = np.array(
        [
            [100.0, 100.0, 100.0],
            [200.0, 200.0, 200.0],
            [270.0, 270.0, 270.0],
            [300.0, 300.0, 300.0],
            [400.0, 400.0, 400.0],
            [800.0, 800.0, 800.0],
        ]
    )
    expected_eas: ClassVar[np.ndarray] = np.array(
        [
            [100.0, 98.543, 55.666],
            [200.0, 197.085, 111.333],
            [270.0, 266.065, 150.299],
            [300.0, 295.628, 166.999],
            [400.0, 394.170, 222.666],
            [800.0, 788.341, 445.332],
        ]
    )
    expected_cas: ClassVar[np.ndarray] = np.array(
        [
            [100.0, 98.580, 56.269],
            [200.0, 197.362, 116.073],
            [270.0, 266.698, 161.732],
            [300.0, 296.465, 182.507],
            [400.0, 395.578, 252.396],
            [800.0, 789.350, 479.567],
        ]
    )
    expected_mach: ClassVar[np.ndarray] = np.array(
        [
            [0.29386, 0.29488, 0.33723],
            [0.58773, 0.58976, 0.67446],
            [0.79343, 0.79617, 0.91051],
            [0.88159, 0.88464, 1.01168],
            [1.17545, 1.17952, 1.34891],
            [2.35091, 2.35903, 2.69782],
        ]
    )
    expected_re1: ClassVar[np.ndarray] = np.array(
        [
            [6845941, 6683613, 2648139],
            [13691882, 13367227, 5296278],
            [18484040, 18045756, 7149975],
            [20537823, 20050840, 7944417],
            [27383763, 26734454, 10592556],
            [54767527, 53468908, 21185111],
        ]
    )
    expected_dynamic_pressure: ClassVar[np.ndarray] = np.array(
        [
            [6125.0, 5947.8, 1898.0],
            [24500.0, 23791.1, 7591.9],
            [44651.2, 43359.2, 13836.3],
            [55125.0, 53529.9, 17081.9],
            [97999.9, 95164.2, 30367.8],
            [391999.7, 380656.9, 121471.0],
        ]
    )
    expected_impact_pressure: ClassVar[np.ndarray] = np.array(
        [
            [6258.4, 6078.2, 1952.6],
            [26689.4, 25932.3, 8495.0],
            [52127.8, 50672.8, 16946.6],
            [66684.1, 64838.1, 21911.2],
            [135479.4, 131780.5, 44684.1],
            [668493.9, 649486.1, 210939.7],
        ]
    )

    @classmethod
    def check_speeds(cls, atm, tol=1e-4):
        assert_allclose(atm.true_airspeed, cls.expected_tas, rtol=tol)
        assert_allclose(atm.equivalent_airspeed, cls.expected_eas, rtol=tol)
        assert_allclose(atm.calibrated_airspeed, cls.expected_cas, rtol=tol)
        assert_allclose(atm.mach, cls.expected_mach, rtol=tol)
        assert_allclose(atm.unitary_reynolds, cls.expected_re1, rtol=tol)
        assert_allclose(atm.dynamic_pressure, cls.expected_dynamic_pressure, rtol=tol)
        assert_allclose(atm.impact_pressure, cls.expected_impact_pressure, rtol=tol)


@pytest.mark.parametrize("altitude_m,delta_t", [(0.0, 0.0), (10000.0, 10.0), (14000.0, 0.0)])
def test_atmosphere_units_and_scalar_like_inputs(altitude_m, delta_t):
    """Atmosphere and AtmosphereSI must agree for equivalent inputs and scalar-like types."""

    scalar_like_altitudes = [
        altitude_m / foot,
        np.array(altitude_m / foot),
        [altitude_m / foot],
    ]

    for altitude in scalar_like_altitudes:
        atm = Atmosphere(altitude, np.array(delta_t))
        atm_si = AtmosphereSI(altitude_m, delta_t)

        assert_allclose(np.asarray(atm.temperature), np.asarray(atm_si.temperature), rtol=1e-7)
        assert_allclose(np.asarray(atm.pressure), np.asarray(atm_si.pressure), rtol=1e-7)
        assert_allclose(np.asarray(atm.density), np.asarray(atm_si.density), rtol=1e-7)
        assert_allclose(
            np.asarray(atm.speed_of_sound), np.asarray(atm_si.speed_of_sound), rtol=1e-7
        )
        assert_allclose(
            np.asarray(atm.dynamic_viscosity), np.asarray(atm_si.dynamic_viscosity), rtol=1e-7
        )
        assert_allclose(
            np.asarray(atm.kinematic_viscosity), np.asarray(atm_si.kinematic_viscosity), rtol=1e-7
        )


def test_get_altitude_roundtrip_and_shape():
    altitudes_m = np.array([0.0, 500.0, 1000.0, 3000.0])

    atm_si = AtmosphereSI(altitudes_m)
    atm_ft = Atmosphere(altitudes_m / foot)

    assert_allclose(atm_si.altitude, altitudes_m, rtol=0.0, atol=0.0)
    assert_allclose(atm_ft.get_altitude(altitude_in_feet=False), altitudes_m, rtol=0.0, atol=1e-12)
    assert_allclose(atm_ft.get_altitude(), altitudes_m / foot, rtol=0.0, atol=1e-12)


def test_speed_setter_shape_validation_and_reset_behavior():
    atm = Atmosphere([0.0, 5000.0, 10000.0], altitude_in_feet=False)

    with pytest.raises(RuntimeError):
        atm.true_airspeed = [[100.0, 200.0]]

    atm.true_airspeed = [100.0, 120.0, 140.0]
    mach_from_tas = np.asarray(atm.mach)

    atm.mach = [0.5, 0.6, 0.7]
    assert_allclose(atm.true_airspeed, np.asarray(atm.speed_of_sound) * np.array([0.5, 0.6, 0.7]))
    assert not np.allclose(mach_from_tas, np.asarray(atm.mach))

    atm.true_airspeed = None
    assert atm.true_airspeed is None
    assert atm.mach is None
    assert atm.equivalent_airspeed is None
    assert atm.unitary_reynolds is None


def test_speed_conversions_reference_results_with_and_without_broadcast():
    _run_speed_conversion_tests(with_broadcast=False)
    _run_speed_conversion_tests(with_broadcast=True)


def _run_speed_conversion_tests(with_broadcast: bool):
    # These checks intentionally stay at the class-integration level: they validate how
    # `Atmosphere` wires speed setters/getters together, not the individual formulas.
    if with_broadcast:
        altitudes = [0.0, 1000.0, 35000.0]
        tas = np.array(Checker.expected_tas)[:, [0]]
    else:
        altitudes = [[0.0, 1000.0, 35000.0]] * 6
        tas = Checker.expected_tas

    atm = Atmosphere(altitudes)
    atm.true_airspeed = tas
    Checker.check_speeds(atm)

    # Re-initialize from each speed definition and verify convergence to reference TAS.
    for attr_name, expected in [
        ("equivalent_airspeed", Checker.expected_eas),
        ("mach", Checker.expected_mach),
        ("unitary_reynolds", Checker.expected_re1),
        ("dynamic_pressure", Checker.expected_dynamic_pressure),
        ("impact_pressure", Checker.expected_impact_pressure),
        ("calibrated_airspeed", Checker.expected_cas),
    ]:
        test_atm = Atmosphere(altitudes)
        setattr(test_atm, attr_name, expected)
        Checker.check_speeds(test_atm)

    # Single-altitude case with vector speeds keeps expected shape semantics.
    one_alt_atm = Atmosphere(35000)
    one_alt_atm.true_airspeed = np.array(Checker.expected_tas)[:, 2]
    assert_allclose(
        one_alt_atm.equivalent_airspeed, np.array(Checker.expected_eas)[:, 2], rtol=1e-4
    )
    assert_allclose(one_alt_atm.mach, np.array(Checker.expected_mach)[:, 2], rtol=1e-4)


@pytest.fixture(scope="session")
def altitude():
    return np.linspace(0.0, 20000.0, 3000)


def _all_state_properties(atm):
    _ = atm.temperature
    _ = atm.pressure
    _ = atm.density
    _ = atm.dynamic_viscosity
    _ = atm.kinematic_viscosity
    _ = atm.speed_of_sound


def _all_speed_properties(atm):
    _ = atm.true_airspeed
    _ = atm.equivalent_airspeed
    _ = atm.calibrated_airspeed
    _ = atm.mach
    _ = atm.unitary_reynolds
    _ = atm.dynamic_pressure
    _ = atm.impact_pressure


def test_performances_array_state_bundle(altitude, benchmark):
    # Benchmark the full state-property access pattern exposed by the class.
    def func():
        atm = AtmosphereSI(altitude)
        _all_state_properties(atm)

    benchmark(func)


def test_performances_array_speed_bundle_init_tas(altitude, benchmark):
    # Benchmark a common class workflow: initialize from TAS, then query all derived speeds.
    def func():
        atm = AtmosphereSI(altitude)
        atm.true_airspeed = 200.0
        _all_speed_properties(atm)

    benchmark(func)


def test_performances_array_speed_bundle_init_cas(altitude, benchmark):
    # Sparse slice keeps this benchmark practical while still exercising the root-solver path.
    altitude_sparse = altitude[::1000]

    def func():
        atm = AtmosphereSI(altitude_sparse)
        atm.calibrated_airspeed = 120.0
        _ = atm.true_airspeed

    benchmark(func)


def test_performances_array_cached_reask_bundle(altitude, benchmark):
    # Re-asking cached properties is a distinct class behavior worth tracking separately.
    atm = AtmosphereSI(altitude)
    atm.true_airspeed = 200.0
    _all_state_properties(atm)
    _all_speed_properties(atm)

    def func():
        _all_state_properties(atm)
        _all_speed_properties(atm)

    benchmark(func)


def test_performances_scalar_state_bundle(altitude, benchmark):
    def func():
        for alt in altitude[::100]:
            atm = AtmosphereSI(float(alt))
            _all_state_properties(atm)

    benchmark(func)


def test_performances_scalar_speed_bundle_init_tas(altitude, benchmark):
    def func():
        for alt in altitude[::100]:
            atm = AtmosphereSI(float(alt))
            atm.true_airspeed = 120.0
            _all_speed_properties(atm)

    benchmark(func)


def test_performances_scalar_tas_from_cas(altitude, benchmark):
    def func():
        for alt in altitude[::100]:
            atm = AtmosphereSI(float(alt))
            atm.calibrated_airspeed = 100.0
            _ = atm.true_airspeed

    benchmark(func)
