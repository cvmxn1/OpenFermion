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
"""Classes to assist in processing of VPE (ArXiv:2010.02538) signal data"""

import abc
from typing import Sequence, Optional
import numpy

import cirq

from openfermion.linalg import (
    fit_known_frequencies,
    fit_known_frequencies_in_phase,
    get_condition_number_generation_matrix,
)


class _VPEEstimator(metaclass=abc.ABCMeta):
    """Generic class for any VPE estimator"""

    @abc.abstractmethod
    def get_simulation_points(self) -> numpy.ndarray:
        """Generates time points for estimation

        VPE requires estimating the phase function g(t) at multiple points t,
        and some care in choosing these points is needed to prevent aliasing.
        This should be taken care of in the estimator.

        Returns:
            times: a set of times t that g(t) should be estimated at.
        """

    @abc.abstractmethod
    def get_expectation_value(self,
                              phase_function: numpy.ndarray,
                              times: numpy.ndarray = None) -> numpy.ndarray:
        """Estimates expectation values from an input phase function

        Given a phase function g(t), estimates the expectation value <H> of the
        operator used to generate it on the initial state rho

        $g(t) = Trace[e^{-iHt} (|psi_r><psi_r| + rho) e^{iHt} |0><1|])$

        Arguments:
            phase_function [numpy.ndarray] -- The phase function g(t)

        Returns:
            expectation value [numpy.ndarray] -- <H>.
        """


class PhaseFitEstimator(_VPEEstimator):
    """A VPE estimator that works by fitting a set of known frequencies.

    A Hamiltonian being fast-forwardable is equivalent to its spectral
    decomposition being known. This means that the only information to
    be obtained from QPE is the amplitudes. This estimator proceeds
    by a simple least-squares fit to obtain the amplitudes, and then
    outputs the expectation values.
    """

    def __init__(self,
                 evals: numpy.ndarray,
                 ref_eval: float = 0,
                 separation_tol: float = 1e-6):
        """
        Arguments:
            evals [numpy.ndarray] -- The (known) eigenvalues of the target
                operator
            ref_eval [numpy.ndarray] -- The eigenvalue of the reference state.
                When using a control qubit for QPE, this should be set to 0.
        """
        self.evals = [
            ev for n, ev in enumerate(evals)
            if ((n == 0) or
                (min([abs(ev - ev2) for ev2 in evals[:n]]) > separation_tol))
        ]
        self.ref_eval = ref_eval

    def get_simulation_points(self, safe: bool = True) -> numpy.ndarray:
        """Generates time points for estimation

        VPE requires estimating the phase function g(t) at multiple points t,
        and some care in choosing these points is needed to prevent aliasing.
        This should be taken care of in the estimator.

        In this case, we fit len(self.energies) complex amplitudes to a complex
        valued signal, we need precisely this number of points in the signal.

        However, it appears numerically that approximately twice as many points
        are needed to prevent aliasing, so we double this number here.

        Then, to prevent aliasing, we need to make sure that the time step
        dt < 2*pi / (E_max-E_min). Here, we choose dt = pi / (E_max-E_min).
        (Importantly, for Pauli operators this reproduces the H test.)

        Args:
            safe [bool, default True] -- numerical testing shows that taking
                approximately twice as many points is better for the stability
                of the estimator; this

        Returns:
            times: a set of times t that g(t) should be estimated at.
        """
        if safe:
            numsteps = len(self.evals) * 4
            step_size = numpy.pi / (max(self.evals) - min(self.evals)) / 2
        else:
            numsteps = len(self.evals)
            step_size = numpy.pi / (max(self.evals) - min(self.evals))
        maxtime = step_size * (numsteps - 1)
        times = numpy.linspace(0, maxtime, numsteps)
        return times

    def get_amplitudes(self,
                       phase_function: numpy.ndarray,
                       times: numpy.ndarray = None,
                       force_inphase: bool = False) -> numpy.ndarray:
        """Fits the amplitudes in the phase function to the input signal data.

        Arguments:
            phase_function [numpy.ndarray] -- Phase function input
            force_inphase [bool] -- Flag for whether the resulting amplitudes
                should be stuck at having the same amplitudes.

        Returns:
            amplitudes [numpy.ndarray] -- Fitted estimates of the amplitudes
                of the given frequencies (in the same order as in self.energies)
        """
        if times is None:
            times = self.get_simulation_points()
        phase_function_shifted = numpy.array(phase_function) *\
            numpy.exp(1j * times * self.ref_eval)
        if force_inphase:
            amplitudes = fit_known_frequencies_in_phase(phase_function_shifted,
                                                        times, self.evals)
        else:
            amplitudes = fit_known_frequencies(phase_function_shifted, times,
                                               self.evals)
        return amplitudes

    def get_expectation_value(self,
                              phase_function: numpy.ndarray,
                              times: Optional[numpy.ndarray] = None,
                              force_inphase: bool = False) -> numpy.ndarray:
        """Estates expectation values via amplitude fitting of known frequencies

        Arguments:
            phase_function [numpy.ndarray] -- The phase function obtained in
                experiment

        Returns:
            expectation_value [float] -- the estimated expectation value
        """
        amplitudes = self.get_amplitudes(phase_function, times, force_inphase)
        expectation_value = numpy.dot(numpy.real(amplitudes),
                                      self.evals) / numpy.sum(
                                          numpy.real(amplitudes))
        return expectation_value

    def get_condition_number(self, times: Optional[numpy.ndarray] = None):
        """Gets the condition number for the chosen problem and set of times
        
        Arguments:
            times [numpy.ndarray or None] -- the times to estimate g(t) at
                when solving this problem. When None, calls
                get_simulation_points to obtain appropriate values.

        Returns:
            cond_number [float] -- the condition number of the matrix B*B
                where B is the matrix that solves BA=g for our given problem.
        """
        return get_condition_number_generation_matrix(times, self.evals)


# disabling yapf here as its proposed formatting decreases readability
# yapf: disable
standard_rotation_set = [
    [0.25, cirq.ry(numpy.pi / 2), cirq.ry(-numpy.pi / 2)],
    [-0.25, cirq.ry(numpy.pi / 2), cirq.ry(numpy.pi / 2)],
    [-0.25j, cirq.ry(numpy.pi / 2), cirq.rx(-numpy.pi / 2)],
    [0.25j, cirq.ry(numpy.pi / 2), cirq.rx(numpy.pi / 2)],
    [0.25, cirq.rx(numpy.pi / 2), cirq.rx(-numpy.pi / 2)],
    [-0.25, cirq.rx(numpy.pi / 2), cirq.rx(numpy.pi / 2)],
    [0.25j, cirq.rx(numpy.pi / 2), cirq.ry(-numpy.pi / 2)],
    [-0.25j, cirq.rx(numpy.pi / 2), cirq.ry(numpy.pi / 2)],
]
# yapf: enable


def get_phase_function(results: Sequence[cirq.TrialResult],
                       qubits: Sequence[cirq.Qid],
                       target_qid: int,
                       rotation_set: Optional[Sequence] = None,
                       measurement_type: Optional[str] = 'shots'):
    """Generates an estimate of the phase function g(t) from circuit output

    The output from a VPE circuit is a set of measurements; from the frequency
    that these measurements occur, we can estimate the phase function.

    Arguments:
        measurements [Sequence[cirq.TrialResult]] -- A list of TrialResults
            from the different circuits to be run at each point. We assume that
            these circuits are correlated to the order of rotation_set, and the
            only difference should be the initial and final rotation (following)
            that data in rotation_set. We also assume that the final measurement
            is tagged with a label of 'msmt' (and that this is a measurement of
            all qubits, with the target qubit in the bit position indicated by
            target_qid)
        qubits [Sequence[cirq.Qid]] -- The list of qubits in the order that was
            passed to the final measurement call.

            Note: we flip from small endian to big endian notation within this
            function, no need to do this externally.

        target_qid [Int] -- The index of the target qubit in qubits.

            Note: we flip from small endian to big endian notation within this
            function, no need to do this externally.

        rotation_set [Sequence or None] -- The set of rotations performed to
            generate the input data in measurements. These in turn need to be
            summed together weighted by the first entry in the set (we do not
            use the other entries in the set here).

    Returns:
        phase_function [complex] -- An estimate of g(t).
    """
    hs_index = 2**(len(qubits) - target_qid - 1)
    if rotation_set is None:
        rotation_set = standard_vpe_rotation_set
    phase_function = 0
    if len(results) != len(rotation_set):
        raise ValueError("I have an incorrect number of TrialResults "
                         "in results. Correct length should be: {}".format(
                             len(rotation_set)))
    for result, rdata in zip(results, rotation_set):
        if measurement_type == 'wave_function':
            # Assumes no final rotation
            wf = result.final_state_vector
            phase_function += 2 * numpy.conj(wf[0]) * wf[hs_index] * rdata[0]
            continue
        total_shots = result.data['msmt'].count()
        msmt_counts = result.data['msmt'].value_counts()
        if 0 in msmt_counts:
            vprob0 = msmt_counts[0] / total_shots
        else:
            vprob0 = 0
        if hs_index in msmt_counts:
            vprob1 = msmt_counts[hs_index] / total_shots
        else:
            vprob1 = 0
        phase_function += rdata[0] * (vprob0 - vprob1)
    return phase_function
