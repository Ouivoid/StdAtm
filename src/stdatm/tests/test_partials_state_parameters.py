"""Unit and benchmark tests for functions in `partials_state_parameters.py`.

Ownership note:
- this file owns low-level partial-derivative formula validation and scalar/array benchmarks
- class-level exposure of those derivatives through `AtmosphereWithPartials`
  belongs in `test_atmosphere_partials.py`

These tests compare analytical partial derivatives against finite-difference
references and also benchmark scalar and vectorized call patterns.
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

from ..atmosphere import AtmosphereSI
from ..partials_state_parameters import (
    compute_partial_density,
    compute_partial_dynamic_viscosity,
    compute_partial_kinematic_viscosity,
    compute_partial_pressure,
    compute_partial_speed_of_sound,
    compute_partial_temperature,
)
from ..state_parameters import (
    compute_density,
    compute_dynamic_viscosity,
    compute_kinematic_viscosity,
    compute_pressure,
    compute_speed_of_sound,
    compute_temperature,
)


def _get_fd_partial(altitude, parameter_name, step=1e-6):
    """Approximate d(parameter)/d(altitude) with a centered finite difference."""
    atm_minus = AtmosphereSI(altitude - step)
    atm_plus = AtmosphereSI(altitude + step)
    return (getattr(atm_plus, parameter_name) - getattr(atm_minus, parameter_name)) / (2.0 * step)


def _get_state_data(altitude):
    """Build state and partial values at a given altitude for derivative chain tests."""
    temperature = compute_temperature(altitude, 0.0)
    pressure = compute_pressure(altitude)
    density = compute_density(pressure, temperature)
    speed_of_sound = compute_speed_of_sound(temperature)
    dynamic_viscosity = compute_dynamic_viscosity(temperature)
    kinematic_viscosity = compute_kinematic_viscosity(dynamic_viscosity, density)

    partial_temperature = compute_partial_temperature(altitude)
    partial_pressure = compute_partial_pressure(altitude)
    partial_density = compute_partial_density(
        temperature, pressure, partial_temperature, partial_pressure
    )
    partial_speed_of_sound = compute_partial_speed_of_sound(temperature, partial_temperature)
    partial_dynamic_viscosity = compute_partial_dynamic_viscosity(temperature, partial_temperature)
    partial_kinematic_viscosity = compute_partial_kinematic_viscosity(
        dynamic_viscosity, density, partial_dynamic_viscosity, partial_density
    )

    return {
        "temperature": temperature,
        "pressure": pressure,
        "density": density,
        "speed_of_sound": speed_of_sound,
        "dynamic_viscosity": dynamic_viscosity,
        "kinematic_viscosity": kinematic_viscosity,
        "partial_temperature": partial_temperature,
        "partial_pressure": partial_pressure,
        "partial_density": partial_density,
        "partial_speed_of_sound": partial_speed_of_sound,
        "partial_dynamic_viscosity": partial_dynamic_viscosity,
        "partial_kinematic_viscosity": partial_kinematic_viscosity,
    }


def _validation_altitude_array():
    """Validation grid used for FD comparisons in array tests."""
    # Avoid exactly 11,000 m where finite-difference derivatives are not representative.
    return np.linspace(0.0, 20000.0, 101, endpoint=False)


def test_compute_partial_temperature_scalar():
    for altitude in [2000.0, 10000.0, 15000.0]:
        computed = compute_partial_temperature(altitude)
        expected = _get_fd_partial(altitude, "temperature")
        assert expected == pytest.approx(computed, rel=5e-5)


def test_compute_partial_temperature_array():
    altitude = _validation_altitude_array()
    computed = compute_partial_temperature(altitude)
    expected = _get_fd_partial(altitude, "temperature")
    assert_allclose(computed, expected, rtol=5e-5)


def test_compute_partial_pressure_scalar():
    for altitude in [2000.0, 10000.0, 15000.0]:
        computed = compute_partial_pressure(altitude)
        expected = _get_fd_partial(altitude, "pressure")
        assert expected == pytest.approx(computed, rel=5e-5)


def test_compute_partial_pressure_array():
    altitude = _validation_altitude_array()
    computed = compute_partial_pressure(altitude)
    expected = _get_fd_partial(altitude, "pressure")
    assert_allclose(computed, expected, rtol=5e-5)


def test_compute_partial_density_scalar():
    for altitude in [2000.0, 10000.0, 15000.0]:
        data = _get_state_data(altitude)
        computed = compute_partial_density(
            data["temperature"],
            data["pressure"],
            data["partial_temperature"],
            data["partial_pressure"],
        )
        expected = _get_fd_partial(altitude, "density")
        assert expected == pytest.approx(computed, rel=5e-5)


def test_compute_partial_density_array():
    altitude = _validation_altitude_array()
    data = _get_state_data(altitude)
    computed = compute_partial_density(
        data["temperature"], data["pressure"], data["partial_temperature"], data["partial_pressure"]
    )
    expected = _get_fd_partial(altitude, "density")
    assert_allclose(computed, expected, rtol=5e-5)


def test_compute_partial_speed_of_sound_scalar():
    for altitude in [2000.0, 10000.0, 15000.0]:
        data = _get_state_data(altitude)
        computed = compute_partial_speed_of_sound(data["temperature"], data["partial_temperature"])
        expected = _get_fd_partial(altitude, "speed_of_sound")
        assert expected == pytest.approx(computed, rel=5e-5)


def test_compute_partial_speed_of_sound_array():
    altitude = _validation_altitude_array()
    data = _get_state_data(altitude)
    computed = compute_partial_speed_of_sound(data["temperature"], data["partial_temperature"])
    expected = _get_fd_partial(altitude, "speed_of_sound")
    assert_allclose(computed, expected, rtol=5e-5)


def test_compute_partial_dynamic_viscosity_scalar():
    for altitude in [2000.0, 10000.0, 15000.0]:
        data = _get_state_data(altitude)
        computed = compute_partial_dynamic_viscosity(
            data["temperature"], data["partial_temperature"]
        )
        expected = _get_fd_partial(altitude, "dynamic_viscosity")
        assert expected == pytest.approx(computed, rel=5e-5)


def test_compute_partial_dynamic_viscosity_array():
    altitude = _validation_altitude_array()
    data = _get_state_data(altitude)
    computed = compute_partial_dynamic_viscosity(data["temperature"], data["partial_temperature"])
    expected = _get_fd_partial(altitude, "dynamic_viscosity")
    assert_allclose(computed, expected, rtol=1e-4)


def test_compute_partial_kinematic_viscosity_scalar():
    for altitude in [2000.0, 10000.0, 15000.0]:
        data = _get_state_data(altitude)
        computed = compute_partial_kinematic_viscosity(
            data["dynamic_viscosity"],
            data["density"],
            data["partial_dynamic_viscosity"],
            data["partial_density"],
        )
        expected = _get_fd_partial(altitude, "kinematic_viscosity")
        assert expected == pytest.approx(computed, rel=5e-5)


def test_compute_partial_kinematic_viscosity_array():
    altitude = _validation_altitude_array()
    data = _get_state_data(altitude)
    computed = compute_partial_kinematic_viscosity(
        data["dynamic_viscosity"],
        data["density"],
        data["partial_dynamic_viscosity"],
        data["partial_density"],
    )
    expected = _get_fd_partial(altitude, "kinematic_viscosity")
    assert_allclose(computed, expected, rtol=5e-5)


@pytest.fixture(scope="session")
def altitude():
    """Large altitude vector used by benchmark tests for array and scalar loops."""
    return np.linspace(0.0, 20000.0, int(1e6))


def test_performances_array_compute_partial_temperature(altitude, benchmark):
    def func():
        _ = compute_partial_temperature(altitude)

    benchmark(func)


def test_performances_scalar_compute_partial_temperature(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            _ = compute_partial_temperature(float(alt))

    benchmark(func)


def test_performances_array_compute_partial_pressure(altitude, benchmark):
    def func():
        _ = compute_partial_pressure(altitude)

    benchmark(func)


def test_performances_scalar_compute_partial_pressure(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            _ = compute_partial_pressure(float(alt))

    benchmark(func)


def test_performances_array_compute_partial_density(altitude, benchmark):
    temperature = compute_temperature(altitude, 0.0)
    pressure = compute_pressure(altitude)
    partial_temperature = compute_partial_temperature(altitude)
    partial_pressure = compute_partial_pressure(altitude)

    def func():
        _ = compute_partial_density(temperature, pressure, partial_temperature, partial_pressure)

    benchmark(func)


def test_performances_scalar_compute_partial_density(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            temperature = compute_temperature(float(alt), 0.0)
            pressure = compute_pressure(float(alt))
            partial_temperature = compute_partial_temperature(float(alt))
            partial_pressure = compute_partial_pressure(float(alt))
            _ = compute_partial_density(
                temperature, pressure, partial_temperature, partial_pressure
            )

    benchmark(func)


def test_performances_array_compute_partial_speed_of_sound(altitude, benchmark):
    temperature = compute_temperature(altitude, 0.0)
    partial_temperature = compute_partial_temperature(altitude)

    def func():
        _ = compute_partial_speed_of_sound(temperature, partial_temperature)

    benchmark(func)


def test_performances_scalar_compute_partial_speed_of_sound(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            temperature = compute_temperature(float(alt), 0.0)
            partial_temperature = compute_partial_temperature(float(alt))
            _ = compute_partial_speed_of_sound(temperature, partial_temperature)

    benchmark(func)


def test_performances_array_compute_partial_dynamic_viscosity(altitude, benchmark):
    temperature = compute_temperature(altitude, 0.0)
    partial_temperature = compute_partial_temperature(altitude)

    def func():
        _ = compute_partial_dynamic_viscosity(temperature, partial_temperature)

    benchmark(func)


def test_performances_scalar_compute_partial_dynamic_viscosity(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            temperature = compute_temperature(float(alt), 0.0)
            partial_temperature = compute_partial_temperature(float(alt))
            _ = compute_partial_dynamic_viscosity(temperature, partial_temperature)

    benchmark(func)


def test_performances_array_compute_partial_kinematic_viscosity(altitude, benchmark):
    temperature = compute_temperature(altitude, 0.0)
    pressure = compute_pressure(altitude)
    density = compute_density(pressure, temperature)
    dynamic_viscosity = compute_dynamic_viscosity(temperature)
    partial_temperature = compute_partial_temperature(altitude)
    partial_pressure = compute_partial_pressure(altitude)
    partial_density = compute_partial_density(
        temperature, pressure, partial_temperature, partial_pressure
    )
    partial_dynamic_viscosity = compute_partial_dynamic_viscosity(temperature, partial_temperature)

    def func():
        _ = compute_partial_kinematic_viscosity(
            dynamic_viscosity, density, partial_dynamic_viscosity, partial_density
        )

    benchmark(func)


def test_performances_scalar_compute_partial_kinematic_viscosity(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            temperature = compute_temperature(float(alt), 0.0)
            pressure = compute_pressure(float(alt))
            density = compute_density(pressure, temperature)
            dynamic_viscosity = compute_dynamic_viscosity(temperature)
            partial_temperature = compute_partial_temperature(float(alt))
            partial_pressure = compute_partial_pressure(float(alt))
            partial_density = compute_partial_density(
                temperature, pressure, partial_temperature, partial_pressure
            )
            partial_dynamic_viscosity = compute_partial_dynamic_viscosity(
                temperature, partial_temperature
            )
            _ = compute_partial_kinematic_viscosity(
                dynamic_viscosity, density, partial_dynamic_viscosity, partial_density
            )

    benchmark(func)
