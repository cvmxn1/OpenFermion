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
"""Tests for Richardson Gaudin model module."""

import pytest
import numpy

from openfermion.hamiltonians import RichardsonGaudin
from openfermion.ops import QubitOperator
from openfermion.transforms import get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator


@pytest.mark.parametrize('g, n_qubits, expected', [
    (0.3, 2,
     QubitOperator('1.5 [] + 0.15 [X0 X1] + \
0.15 [Y0 Y1] + 0.5 [Z0] + 1.0 [Z1]')),
    (-0.1, 3,
     QubitOperator('3.0 [] - 0.05 [X0 X1] - 0.05 [X0 X2] - \
0.05 [Y0 Y1] - 0.05 [Y0 Y2] + 0.5 [Z0] - 0.05 [X1 X2] - \
0.05 [Y1 Y2] + 1.0 [Z1] + 1.5 [Z2]')),
])
def test_richardson_gaudin_hamiltonian(g, n_qubits, expected):
    rg = RichardsonGaudin(g, n_qubits)
    rg_qubit = rg.qubit_operator
    assert rg_qubit == expected

    assert numpy.array_equal(
        numpy.sort(numpy.unique(get_sparse_operator(rg_qubit).diagonal())),
        numpy.array(list(range((n_qubits + 1) * n_qubits // 2 + 1))))


def test_n_body_tensor_errors():
    rg = RichardsonGaudin(1.7, n_qubits=2)
    with pytest.raises(TypeError):
        rg.n_body_tensors = 0
    with pytest.raises(TypeError):
        rg.constant = 1.1
