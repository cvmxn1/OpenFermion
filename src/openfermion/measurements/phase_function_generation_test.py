#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
'''tests for phase_function_generation'''

import pytest
import numpy

from .phase_function_generation import make_phase_function


def test_no_sampling_noise():
    amplitudes = numpy.array([0.2, 0.8])
    energies = numpy.array([0, 1])
    times = numpy.array([0, 1])
    pf = make_phase_function(times, energies, amplitudes)
    assert len(pf) == 2
    assert numpy.isclose(pf[0], 1)
    assert numpy.isclose(pf[1], 0.2 + 0.8 * numpy.exp(1j))


def test_rng_repeats():
    amplitudes = numpy.array([0.2, 0.8])
    energies = numpy.array([0, 1])
    times = numpy.array([0, 1])
    rng = numpy.random.RandomState(seed=42)
    pf = make_phase_function(times,
                             energies,
                             amplitudes,
                             repetitions=100,
                             rng=rng)
    rng = numpy.random.RandomState(seed=42)
    pf2 = make_phase_function(times,
                              energies,
                              amplitudes,
                              repetitions=100,
                              rng=rng)
    assert len(pf) == 2
    assert len(pf2) == 2
    assert numpy.isclose(pf, pf2).all()


def test_sampling_noise():
    amplitudes = numpy.array([0.2, 0.8])
    energies = numpy.array([0, 1])
    times = numpy.array([0, 1])
    repetitions = 1000000
    pf = make_phase_function(times,
                             energies,
                             amplitudes,
                             repetitions=repetitions)
    err = 1 / numpy.sqrt(repetitions)
    assert len(pf) == 2
    assert numpy.isclose(numpy.real(pf[0]), 1)
    assert numpy.abs(pf[0] - 1) < 2 * err
    assert numpy.abs(pf[1] - (0.2 + 0.8 * numpy.exp(1j))) < 2 * err
