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
"""Class and functions to store interaction operators."""
import numpy

from openfermion.ops import QubitOperator
from openfermion.ops.representations import (PolynomialTensor,
                                             get_tensors_from_integrals)

COEFFICIENT_TYPES = (int, float, complex)


class RichardsonGaudin:
    r"""Class for storing Richardson Gaudin hamiltonians within an equi-distant potential
    constant coupling term g which can be either attractive or repulsive indicated by its sign.

    Note that the operators stored in this class take the form:

        .. math::

            constant + \sum_{p} p N_p +
            g \sum_{p \neq q} P_p^\dagger P_q,

    where

        .. math::

            N_p = (1 - \sigma^Z_p)/2,
            P_p = a_{i,\beta} a_{i,\alpha},
            g = constant coupling term

    Attributes:
        constant: The constant offset.
        g = constant coupling term
    """

    def __init__(self, g, n_qubits):
        r"""
        Initialize the DOCIHamiltonian class.

        Args:
            g: The coefficients of (:math:`h^{(r2)}_{p, q}`).
                This is an n_qubits x n_qubits array of floats.
        """
        self._n_qubits = n_qubits 
        self._g = g

    @property
    def hamiltonian(self):
        """Return the QubitOperator representation of this DOCI Hamiltonian"""
        return self.identity_part + self.z_part + self.xy_part

    def xx_term(self, p, q):
        """Returns the XX term on a single pair of qubits as a QubitOperator
        Arguments:
            p, q [int] -- qubit indices
        Returns:
            [QubitOperator] -- XX term on the chosen qubits.
        """
        return QubitOperator("X" + str(p) + " X" + str(q), self._g / 2)

    def yy_term(self, p, q):
        """Returns the YY term on a single pair of qubits as a QubitOperator
        Arguments:
            p, q [int] -- qubit indices
        Returns:
            [QubitOperator] -- YY term on the chosen qubits.
        """
        return QubitOperator("Y" + str(p) + " Y" + str(q), self._g / 2)

    def z_term(self, p):
        """Returns the Z term on a single qubit as a QubitOperator
        Arguments:
            p [int] -- qubit index
        Returns:
            [QubitOperator] -- Z term on the chosen qubit.
        """
        return QubitOperator("Z" + str(p), p / 2)

    @property
    def identity_part(self):
        """Returns identity term of this operator (i.e. trace-ful term)
        in QubitOperator form.
        """
        return QubitOperator(
            (), (self._n_qubits+1)*self._n_qubits / 2 )

    @property
    def xx_part(self):
        """Returns the XX part of the QubitOperator representation of this
        DOCIHamiltonian
        """
        return sum([
            self.xx_term(p, q)
            for p in range(self.n_qubits)
            for q in range(p + 1, self.n_qubits)
        ])

    @property
    def yy_part(self):
        """Returns the YY part of the QubitOperator representation of this
        DOCIHamiltonian
        """
        return sum([
            self.yy_term(p, q)
            for p in range(self.n_qubits)
            for q in range(p + 1, self.n_qubits)
        ])

    @property
    def xy_part(self):
        """Returns the XX+YY part of the QubitOperator representation of this
        DOCIHamiltonian
        """
        return self.xx_part + self.yy_part

    @property
    def z_part(self):
        """Return the Z and ZZ part of the QubitOperator representation of this
        DOCI Hamiltonian"""
        return sum([self.z_term(p) for p in range(self.n_qubits)])


    @property
    def g(self):
        """The value of g"""
        return self._g

    @g.setter
    def g(self, value):
        self._g = value

