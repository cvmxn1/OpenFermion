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
"""This module constructs Hamiltonians of the Richardson Gaudin type.
"""
import numpy

from openfermion.ops import QubitOperator
from openfermion.ops.representations import (PolynomialTensor,
                                             get_tensors_from_integrals)
from openfermion.ops.representations import DOCIHamiltonian

class RichardsonGaudin(DOCIHamiltonian):
    r"""Richardson Gaudin model.

    Class for storing and constructing Richardson Gaudin hamiltonians
    combining an equi-distant potential ladder like potential per
    qubit with a uniform coupling between any pair of
    qubits with coupling strength g, which can be either attractive
    (g<0) or repulsive (g>0).

    The operators represented by this class has the form:

    .. math::

        H = \sum_{p=0} (p + 1) N_p + g/2 \sum_{p < q} P_p^\dagger P_q,

    where

    .. math::

        \begin{align}
        N_p &= (1 - \sigma^Z_p)/2, \\
        P_p &= a_{p,\beta} a_{p,\alpha} = S^{-} = \sigma^X - i \sigma^Y, \\
        g &= constant coupling term
        \end{align}

    Note;
        The digonal of the Hamiltonian is composed of the values in
        range((n_qubits+1)*n_qubits//2+1).
    """
    def __init__(self, g, n_qubits):
        r"""Richardson Gaudin model on a given number of qubits.

        Args:
            g (float): Coupling strength
            n_qubits (int): Number of qubits
        """
        hc = numpy.zeros((n_qubits,))
        hr1 = numpy.zeros((n_qubits, n_qubits))
        hr2 = numpy.zeros((n_qubits, n_qubits))
        for p in range(n_qubits):
            hc[p] = p + 1
            for q in range(n_qubits):
                if p != q:
                    hr1[p, q] = g
        super().__init__(0.0, hc, hr1, hr2)

    @DOCIHamiltonian.constant.setter
    def constant(self, value):
        raise TypeError(
            'Raw edits of the constant of a RichardsonGaudin model'
            'is not allowed. Either adjust the g paramter '
            'or cast to another PolynomialTensor class.')

    @DOCIHamiltonian.n_body_tensors.setter
    def n_body_tensors(self, value):
        raise TypeError(
            'Raw edits of the n_body_tensors of a RichardsonGaudin model'
            'is not allowed. Either adjust the g paramter '
            'or cast to another PolynomialTensor class.')
