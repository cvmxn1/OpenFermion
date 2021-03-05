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
"""Tests for pCCD."""

import os
import unittest
import numpy

from openfermion.config import DATA_DIRECTORY
from openfermion.chem import pCCD
from openfermion.chem.molecular_data import MolecularData


def test_pccd():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
    basis = 'sto-3g'
    multiplicity = 1
    filename = os.path.join(DATA_DIRECTORY, 'H2_sto-3g_singlet_0.7414')
    molecule = MolecularData(geometry, basis, multiplicity, filename=filename)
    molecule.load()

    pccd = pCCD(molecule, iter_max=20)
    pccd.setup_integrals(molecule)
    pccd_energy = pccd.compute_energy()
    assert numpy.isclose(pccd_energy , -1.13727009752064733838)
