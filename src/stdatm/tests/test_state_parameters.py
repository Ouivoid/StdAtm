"""Unit and benchmark tests for functions in `state_parameters.py`.

Ownership note:
- this file owns low-level state-parameter formula validation and scalar/array benchmarks
- class-level integration of these formulas through `Atmosphere` and `AtmosphereSI`
  belongs in `test_atmosphere.py`

The `EXPECTATIONS` table comes from the existing atmosphere reference values and is
used to validate each function with scalar and vectorized inputs.
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

from ..state_parameters import (
    compute_density,
    compute_dynamic_viscosity,
    compute_kinematic_viscosity,
    compute_pressure,
    compute_speed_of_sound,
    compute_temperature,
)

# Columns: altitude, delta-T, and expected outputs for state parameters.
EXPECTATIONS = np.array(
    [
        (0, 0, 288.15, 1.225, 101325, 1.789e-05, 1.460e-05, 340.29),
        (500, 0, 284.90, 1.1673, 95461, 1.773e-05, 1.519e-05, 338.37),
        (1000, 0, 281.65, 1.1117, 89874, 1.757e-05, 1.581e-05, 336.43),
        (1500, 0, 278.40, 1.0581, 84556, 1.742e-05, 1.646e-05, 334.49),
        (2000, 0, 275.15, 1.0065, 79495, 1.725e-05, 1.714e-05, 332.53),
        (2500, 0, 271.90, 0.9569, 74682, 1.710e-05, 1.787e-05, 330.56),
        (3000, 0, 268.65, 0.9091, 70108, 1.693e-05, 1.863e-05, 328.58),
        (3500, 0, 265.40, 0.8632, 65764, 1.677e-05, 1.943e-05, 326.58),
        (4000, 0, 262.15, 0.8191, 61640, 1.661e-05, 2.028e-05, 324.58),
        (4500, 0, 258.90, 0.7768, 57728, 1.644e-05, 2.117e-05, 322.56),
        (5000, 0, 255.65, 0.7361, 54020, 1.628e-05, 2.211e-05, 320.53),
        (5500, 0, 252.40, 0.6971, 50506, 1.611e-05, 2.311e-05, 318.48),
        (6000, 0, 249.15, 0.6597, 47181, 1.594e-05, 2.417e-05, 316.43),
        (6500, 0, 245.90, 0.6238, 44034, 1.578e-05, 2.529e-05, 314.36),
        (7000, 0, 242.65, 0.5895, 41060, 1.561e-05, 2.648e-05, 312.27),
        (7500, 0, 239.40, 0.5566, 38251, 1.544e-05, 2.773e-05, 310.17),
        (8000, 0, 236.15, 0.5252, 35599, 1.526e-05, 2.906e-05, 308.06),
        (8500, 0, 232.90, 0.4951, 33099, 1.509e-05, 3.048e-05, 305.93),
        (9000, 0, 229.65, 0.4663, 30742, 1.492e-05, 3.199e-05, 303.79),
        (9500, 0, 226.40, 0.4389, 28523, 1.474e-05, 3.359e-05, 301.63),
        (10000, 0, 223.15, 0.4127, 26436, 1.457e-05, 3.530e-05, 299.46),
        (10500, 0, 219.90, 0.3877, 24474, 1.439e-05, 3.712e-05, 297.27),
        (11000, 0, 216.65, 0.3639, 22632, 1.421e-05, 3.905e-05, 295.07),
        (12000, 0, 216.65, 0.3108, 19330, 1.421e-05, 4.573e-05, 295.07),
        (13000, 0, 216.65, 0.2655, 16510, 1.421e-05, 5.353e-05, 295.07),
        (14000, 0, 216.65, 0.2268, 14101, 1.421e-05, 6.266e-05, 295.07),
        (15000, 0, 216.65, 0.1937, 12044, 1.421e-05, 7.337e-05, 295.07),
        (16000, 0, 216.65, 0.1654, 10287, 1.421e-05, 8.592e-05, 295.07),
        (17000, 0, 216.65, 0.1413, 8786, 1.421e-05, 1.006e-04, 295.07),
        (18000, 0, 216.65, 0.1207, 7505, 1.421e-05, 1.177e-04, 295.07),
        (19000, 0, 216.65, 0.1031, 6410, 1.421e-05, 1.378e-04, 295.07),
        (20000, 0, 216.65, 0.088, 5475, 1.421e-05, 1.615e-04, 295.07),
        (0, 10, 298.15, 1.1839, 101325, 1.838e-05, 1.5527e-05, 346.15),
        (1000, 10, 291.65, 1.0735, 89875, 1.807e-05, 1.6829e-5, 342.36),
        (3000, 10, 278.65, 0.87650, 70108, 1.742e-05, 1.9877e-5, 334.64),
        (10000, 10, 233.15, 0.39500, 26436, 1.505e-05, 3.8106e-05, 306.10),
        (14000, 10, 226.65, 0.2167, 14102, 1.469e-05, 6.7808e-05, 301.80),
    ],
    dtype=[
        ("alt", "f8"),
        ("dT", "f4"),
        ("T", "f4"),
        ("rho", "f4"),
        ("P", "f4"),
        ("dyn_visc", "f4"),
        ("kin_visc", "f4"),
        ("SoS", "f4"),
    ],
)


def test_compute_temperature_scalar():
    for values in EXPECTATIONS:
        computed = compute_temperature(float(values["alt"]), float(values["dT"]))
        assert values["T"] == pytest.approx(computed, rel=1e-4)


def test_compute_temperature_array():
    computed = compute_temperature(EXPECTATIONS["alt"], EXPECTATIONS["dT"])
    assert_allclose(computed, EXPECTATIONS["T"], rtol=1e-4)


def test_compute_pressure_scalar():
    for values in EXPECTATIONS:
        computed = compute_pressure(float(values["alt"]))
        assert values["P"] == pytest.approx(computed, rel=1e-4)


def test_compute_pressure_array():
    computed = compute_pressure(EXPECTATIONS["alt"])
    assert_allclose(computed, EXPECTATIONS["P"], rtol=1e-4)


def test_compute_density_scalar():
    for values in EXPECTATIONS:
        computed = compute_density(float(values["P"]), float(values["T"]))
        assert values["rho"] == pytest.approx(computed, rel=1e-3)


def test_compute_density_array():
    computed = compute_density(EXPECTATIONS["P"], EXPECTATIONS["T"])
    assert_allclose(computed, EXPECTATIONS["rho"], rtol=1e-3)


def test_compute_speed_of_sound_scalar():
    for values in EXPECTATIONS:
        computed = compute_speed_of_sound(float(values["T"]))
        assert values["SoS"] == pytest.approx(computed, rel=1e-3)


def test_compute_speed_of_sound_array():
    computed = compute_speed_of_sound(EXPECTATIONS["T"])
    assert_allclose(computed, EXPECTATIONS["SoS"], rtol=1e-3)


def test_compute_dynamic_viscosity_scalar():
    for values in EXPECTATIONS:
        computed = compute_dynamic_viscosity(float(values["T"]))
        assert values["dyn_visc"] == pytest.approx(computed, rel=1e-2)


def test_compute_dynamic_viscosity_array():
    computed = compute_dynamic_viscosity(EXPECTATIONS["T"])
    assert_allclose(computed, EXPECTATIONS["dyn_visc"], rtol=1e-2)


def test_compute_kinematic_viscosity_scalar():
    for values in EXPECTATIONS:
        computed = compute_kinematic_viscosity(float(values["dyn_visc"]), float(values["rho"]))
        assert values["kin_visc"] == pytest.approx(computed, rel=1e-2)


def test_compute_kinematic_viscosity_array():
    computed = compute_kinematic_viscosity(EXPECTATIONS["dyn_visc"], EXPECTATIONS["rho"])
    assert_allclose(computed, EXPECTATIONS["kin_visc"], rtol=1e-2)


@pytest.fixture(scope="session")
def altitude():
    """Large altitude vector used by benchmark tests for array and scalar loops."""
    return np.linspace(0.0, 20000.0, int(1e6))


def test_performances_array_compute_temperature(altitude, benchmark):
    delta_t = np.zeros_like(altitude)

    def func():
        _ = compute_temperature(altitude, delta_t)

    benchmark(func)


def test_performances_scalar_compute_temperature(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            _ = compute_temperature(float(alt), 0.0)

    benchmark(func)


def test_performances_array_compute_pressure(altitude, benchmark):
    def func():
        _ = compute_pressure(altitude)

    benchmark(func)


def test_performances_scalar_compute_pressure(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            _ = compute_pressure(float(alt))

    benchmark(func)


def test_performances_array_compute_density(altitude, benchmark):
    pressure = compute_pressure(altitude)
    temperature = compute_temperature(altitude, np.zeros_like(altitude))

    def func():
        _ = compute_density(pressure, temperature)

    benchmark(func)


def test_performances_scalar_compute_density(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            pressure = compute_pressure(float(alt))
            temperature = compute_temperature(float(alt), 0.0)
            _ = compute_density(pressure, temperature)

    benchmark(func)


def test_performances_array_compute_speed_of_sound(altitude, benchmark):
    temperature = compute_temperature(altitude, np.zeros_like(altitude))

    def func():
        _ = compute_speed_of_sound(temperature)

    benchmark(func)


def test_performances_scalar_compute_speed_of_sound(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            temperature = compute_temperature(float(alt), 0.0)
            _ = compute_speed_of_sound(temperature)

    benchmark(func)


def test_performances_array_compute_dynamic_viscosity(altitude, benchmark):
    temperature = compute_temperature(altitude, np.zeros_like(altitude))

    def func():
        _ = compute_dynamic_viscosity(temperature)

    benchmark(func)


def test_performances_scalar_compute_dynamic_viscosity(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            temperature = compute_temperature(float(alt), 0.0)
            _ = compute_dynamic_viscosity(temperature)

    benchmark(func)


def test_performances_array_compute_kinematic_viscosity(altitude, benchmark):
    pressure = compute_pressure(altitude)
    temperature = compute_temperature(altitude, np.zeros_like(altitude))
    density = compute_density(pressure, temperature)
    dynamic_viscosity = compute_dynamic_viscosity(temperature)

    def func():
        _ = compute_kinematic_viscosity(dynamic_viscosity, density)

    benchmark(func)


def test_performances_scalar_compute_kinematic_viscosity(altitude, benchmark):
    def func():
        for alt in altitude[::1000]:
            pressure = compute_pressure(float(alt))
            temperature = compute_temperature(float(alt), 0.0)
            density = compute_density(pressure, temperature)
            dynamic_viscosity = compute_dynamic_viscosity(temperature)
            _ = compute_kinematic_viscosity(dynamic_viscosity, density)

    benchmark(func)
