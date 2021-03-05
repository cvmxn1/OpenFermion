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

from openfermion.ops import QubitOperator
from openfermion.ops.representations import PolynomialTensor


class RichardsonGaudin:
    r"""Richardson Gaudin model.

    Class for storing and constructing Richardson Gaudin hamiltonians
    combining an equi-distant potential ladder like potential per
    qubit with a uniform coupling between any pair of
    qubits with coupling strength g, which can be either attractive
    (g<0) or repulsive (g>0).

    The operators represented by this class has the form:

    .. math::

        H = \sum_{p=0} (p+1) N_p + g/2 \sum_{p < q} P_p^\dagger P_q,

    where

    .. math::

        \begin{align}
        N_p &= (1 - \sigma^Z_p)/2, \\
        P_p &= a_{p,\beta} a_{p,\alpha} = S^{-} = \sigma^X - i \sigma^Y, \\
        g &= constant coupling term
        \end{align}
    """
    def __init__(self, g, n_qubits):
        r"""Richardson Gaudin model on a given number of qubits.

        Args:
            g (float): Coupling strength
            n_qubits (int): Number of qubits
        """
        self._g = g
        self._n_qubits = n_qubits

    @property
    def hamiltonian(self):
        """QubitOperator representation of the Hamiltonian.
        """
        return self.identity_part + self.z_part + self.xx_and_yy_part

    @property
    def identity_part(self):
        """Identity part of the QubitOperator representation of the Hamiltonian.
        """
        return QubitOperator((), (self._n_qubits+1)*self._n_qubits// 2 / 2)

    @property
    def xx_part(self):
        """XX part of the QubitOperator representation of the Hamiltonian.
        """
        return sum([
            QubitOperator("X" + str(p) + " X" + str(q), self._g / 2)
            for p in range(self.n_qubits)
            for q in range(p + 1, self.n_qubits)
        ])

    @property
    def yy_part(self):
        """YY part of the QubitOperator representation of the Hamiltonian.
        """
        return sum([
            QubitOperator("Y" + str(p) + " Y" + str(q), self._g / 2)
            for p in range(self.n_qubits)
            for q in range(p + 1, self.n_qubits)
        ])

    @property
    def z_part(self):
        """Z part of the QubitOperator representation of the Hamiltonian.
        """
        return sum([QubitOperator("Z" + str(p), (p+1)/2) for p in range(self.n_qubits)])

    @property
    def xx_and_yy_part(self):
        """XX and YY parts combined of the QubitOperator representation of the Hamiltonian.
        """
        return self.xx_part + self.yy_part

    @property
    def g(self):
        """The coupling strength g.

        Returns:
            float: The coupling strength g.
        """
        return self._g

    @g.setter
    def g(self, value):
        """Sets the coupling strength g.

        Args:
            g (float): The coupling strength g.
        """
        self._g = value

    @property
    def n_qubits(self):
        """The numner of qubits.

        Returns:
            int: The number of qubits.
        """
        return self._n_qubits
