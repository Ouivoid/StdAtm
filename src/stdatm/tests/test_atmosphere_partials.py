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

"""Tests for `AtmosphereWithPartials` class behavior and performance.

Ownership note:
- low-level partial-derivative formulas and their scalar/array benchmarks are covered in
  `test_partials_state_parameters.py`
- this file focuses on what happens once those formulas are exposed through the
  `AtmosphereWithPartials` class: unit scaling, output shape, caching, representative
  end-to-end checks, and class-level benchmarks
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import foot

from ..atmosphere import Atmosphere
from ..atmosphere_partials import AtmosphereWithPartials

PARTIAL_TO_PARAMETER = {
    "partial_temperature_altitude": "temperature",
    "partial_pressure_altitude": "pressure",
    "partial_density_altitude": "density",
    "partial_speed_of_sound_altitude": "speed_of_sound",
    "partial_dynamic_viscosity_altitude": "dynamic_viscosity",
    "partial_kinematic_viscosity_altitude": "kinematic_viscosity",
}

# Parameter-specific tolerances kept here because the class-level checks below compare
# end-to-end finite differences through `Atmosphere`, not the isolated derivative helpers.
PARTIAL_RTOL = {
    "partial_temperature_altitude": 5e-5,
    "partial_pressure_altitude": 5e-5,
    "partial_density_altitude": 5e-5,
    "partial_speed_of_sound_altitude": 5e-5,
    "partial_dynamic_viscosity_altitude": 1e-4,
    "partial_kinematic_viscosity_altitude": 5e-5,
}


@pytest.fixture(scope="session")
def altitude():
    return np.linspace(0.0, 20000.0, int(1e6))


def _get_fd_partial(altitude_value, parameter_name, altitude_in_feet=False, step=1e-6):
    atm_minus_step = Atmosphere(altitude_value - step, altitude_in_feet=altitude_in_feet)
    atm_plus_step = Atmosphere(altitude_value + step, altitude_in_feet=altitude_in_feet)
    return (getattr(atm_plus_step, parameter_name) - getattr(atm_minus_step, parameter_name)) / (
        2.0 * step
    )


def _all_partials(atm):
    """Force evaluation of every exposed partial property on the class."""
    return {name: getattr(atm, name) for name in PARTIAL_TO_PARAMETER}


def test_partials_units_and_shape_consistency_between_feet_and_meters():
    altitudes_m = np.array([0.0, 3000.0, 7000.0, 10000.0, 15000.0])

    atm_m = AtmosphereWithPartials(altitudes_m, altitude_in_feet=False)
    atm_ft = AtmosphereWithPartials(altitudes_m / foot, altitude_in_feet=True)

    partials_m = _all_partials(atm_m)
    partials_ft = _all_partials(atm_ft)

    for name in PARTIAL_TO_PARAMETER:
        assert np.shape(partials_m[name]) == np.shape(altitudes_m)
        assert np.shape(partials_ft[name]) == np.shape(altitudes_m)
        # d()/d(ft) = d()/d(m) * foot
        assert_allclose(partials_ft[name], np.asarray(partials_m[name]) * foot, rtol=1e-10)


def test_partials_against_fd_representative_points_for_both_units():
    # Class-level spot checks are enough here because exhaustive formula validation lives in
    # `test_partials_state_parameters.py`.
    # Avoid exactly 11,000 m to keep finite differences away from the ISA slope break.
    altitudes_m = np.array([2000.0, 10000.0, 15000.0])

    for altitude_in_feet in [False, True]:
        altitude_input = altitudes_m / foot if altitude_in_feet else altitudes_m
        atm = AtmosphereWithPartials(altitude_input, altitude_in_feet=altitude_in_feet)

        for partial_name, parameter_name in PARTIAL_TO_PARAMETER.items():
            computed = getattr(atm, partial_name)
            expected = _get_fd_partial(altitude_input, parameter_name, altitude_in_feet)
            assert_allclose(computed, expected, rtol=PARTIAL_RTOL[partial_name])


def test_scalar_partials_are_scalar_like():
    atm = AtmosphereWithPartials(3500.0, altitude_in_feet=False)

    for partial_name in PARTIAL_TO_PARAMETER:
        value = getattr(atm, partial_name)
        assert np.asarray(value).shape == ()


def test_partial_properties_are_cached_for_reask():
    atm = AtmosphereWithPartials(np.array([1000.0, 5000.0, 9000.0]), altitude_in_feet=False)

    first = atm.partial_density_altitude
    second = atm.partial_density_altitude
    assert first is second


# Benchmarks focus on class workflows rather than isolated derivative formulas.
def test_performances_array_partials_bundle(altitude, benchmark):
    def func():
        atm = AtmosphereWithPartials(altitude[::10], altitude_in_feet=False)
        _ = atm.partial_temperature_altitude
        _ = atm.partial_pressure_altitude
        _ = atm.partial_density_altitude
        _ = atm.partial_speed_of_sound_altitude
        _ = atm.partial_dynamic_viscosity_altitude
        _ = atm.partial_kinematic_viscosity_altitude

    benchmark(func)


def test_performances_array_partials_reask_bundle(altitude, benchmark):
    # Cached re-ask benchmark: measure repeated property access on an already-populated object.
    atm = AtmosphereWithPartials(altitude, altitude_in_feet=False)
    _ = atm.partial_temperature_altitude
    _ = atm.partial_pressure_altitude
    _ = atm.partial_density_altitude
    _ = atm.partial_speed_of_sound_altitude
    _ = atm.partial_dynamic_viscosity_altitude
    _ = atm.partial_kinematic_viscosity_altitude

    def func():
        _ = atm.partial_temperature_altitude
        _ = atm.partial_pressure_altitude
        _ = atm.partial_density_altitude
        _ = atm.partial_speed_of_sound_altitude
        _ = atm.partial_dynamic_viscosity_altitude
        _ = atm.partial_kinematic_viscosity_altitude

    benchmark(func)


def test_performances_scalar_partials_bundle(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            atm = AtmosphereWithPartials(float(alt), altitude_in_feet=False)
            _ = atm.partial_temperature_altitude
            _ = atm.partial_pressure_altitude
            _ = atm.partial_density_altitude
            _ = atm.partial_speed_of_sound_altitude
            _ = atm.partial_dynamic_viscosity_altitude
            _ = atm.partial_kinematic_viscosity_altitude

    benchmark(func)


def test_performances_scalar_partials_bundle_in_feet(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            atm = AtmosphereWithPartials(float(alt / foot), altitude_in_feet=True)
            _ = atm.partial_temperature_altitude
            _ = atm.partial_pressure_altitude
            _ = atm.partial_density_altitude
            _ = atm.partial_speed_of_sound_altitude
            _ = atm.partial_dynamic_viscosity_altitude
            _ = atm.partial_kinematic_viscosity_altitude

    benchmark(func)
