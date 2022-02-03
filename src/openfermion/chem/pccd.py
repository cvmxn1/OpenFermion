"""
C-like implementaiton of of pCCD equations.

TODO: vectorize further with numpy.  current form is to compare with other codes
"""
from itertools import product
import warnings
import numpy as np
from openfermion.chem.molecular_data import spinorb_from_spatial


class pCCD:
    """Paired coupled cluster doubles
    """
    def __init__(self,
                 molecule,
                 *,
                 iter_max=100,
                 e_convergence=1.0E-6,
                 r_convergence=1.0E-6):
        self.molecule = molecule
        self.o = molecule.n_electrons // 2
        self.v = molecule.n_orbitals - self.o
        self.t0 = None
        self.sigma = None
        self.iter_max = iter_max
        self.e_convergence = e_convergence
        self.r_convergence = r_convergence

    def setup_integrals(self, molecule=None):
        if molecule is None:
            molecule = self.molecule
        oei, tei = molecule.get_integrals()
        o = molecule.n_electrons // 2
        v = molecule.n_orbitals - o

        self.v_iiaa = np.zeros(o * v)
        self.v_iaia = np.zeros(o * v)
        self.v_ijij = np.zeros(o * o)
        self.v_abab = np.zeros(v * v)
        self.f_o = np.zeros(o)
        self.f_v = np.zeros(v)

        for i in range(o):
            for a in range(v):
                self.v_iiaa[i * v + a] = tei[i, a + o, a + o, i]

        for i in range(o):
            for a in range(v):
                self.v_iaia[i * v + a] = tei[i, i, a + o, a + o]

        for i in range(o):
            for j in range(o):
                self.v_ijij[i * o + j] = tei[i, i, j, j]

        for a in range(v):
            for b in range(v):
                self.v_abab[a * v + b] = tei[a + o, a + o, b + o, b + o]

        for i in range(o):
            dum = oei[i, i]
            for k in range(o):
                dum += 2 * tei[i, k, k, i]
                dum -= tei[i, k, i, k]
            self.f_o[i] = dum

        for a in range(v):
            dum = oei[a + o, a + o]
            for k in range(o):
                dum += 2 * tei[a + o, k, k, a + o]
                dum -= tei[a + o, k, a + o, k]
            self.f_v[a] = dum

        self.escf = molecule.nuclear_repulsion
        for i in range(o):
            self.escf += oei[i, i] + self.f_o[i]

    def compute_energy(self):
        o, v = self.o, self.v
        en = 0

        # initialize amplitudes to zero
        self.t2 = np.zeros(o * v)
        self.setup_integrals()
        iter = 0

        while True:

            self.evaluate_residual()

            # update amplitudes
            for i in range(o):
                for a in range(v):
                    self.residual[i * v +
                                  a] *= -0.5 / (self.f_v[a] - self.f_o[i])

            self.t0 = self.residual.copy()
            self.t0 = self.t0 + -self.t2

            nrm = np.linalg.norm(self.t0)
            self.t2 = self.residual.copy()

            dE = en
            en = self.evaluate_projected_energy()
            dE -= en
            print("\t\t\t{}\t{: 5.10f}\t{: 5.10f}\t{: 5.10f}".format(
                iter, en, dE, nrm))

            if np.isnan(en) or np.isnan(dE) or np.isnan(nrm):
                raise TypeError('Unable to converge pCCD calculation. '
                                'Encountered nan value.')

            if np.abs(dE) < self.e_convergence and nrm < self.r_convergence:
                break

            iter += 1

            if iter >= self.iter_max:
                warnings.warn('Exceeded iter_max before converging '
                              'up to convergence criteria.')
                break

        self.correlation_energy = en
        self.total_energy = self.escf + en
        print("\t\tIterations Converged")
        print("\t\tTotal Energy {: 5.20f}".format(self.total_energy))
        return self.total_energy

    def evaluate_projected_energy(self):
        o, v = self.o, self.v
        energy = 0.
        # reset t0 to ones
        for i in range(o):
            for a in range(v):
                energy += self.t2[i * v + a] * self.v_iaia[i * v + a]
        return energy

    def normalize(self):
        self.t0 = np.ones(self.o * self.v)

    def evaluate_residual(self):
        o, v = self.o, self.v
        self.normalize()
        self.residual = np.zeros(o * v)
        self.residual = self.evaluate_sigma()

        VxT_v = np.zeros(v)
        VxT_o = np.zeros(o)
        VxT_oo = np.zeros(o * o)

        # print("VxT_v ")
        for a in range(v):
            # contract over the occupied space to get the virtual index
            VxT_v[a] = -2.0 * np.dot(self.v_iaia[a::v], self.t2[a::v])

        # print("VxT_o ")
        for i in range(o):
            # contract over the virtual index of the vectorized matrix
            VxT_o[i] = -2.0 * np.dot(self.v_iaia[i * v:(i + 1) * v],
                                     self.t2[i * v:(i + 1) * v])

        # print("VxT_oo(i,j) = (jb|jb) t(i,b)")
        for i, j in product(range(o), repeat=2):
            VxT_oo[i * o + j] = np.dot(self.v_iaia[j * v:(j + 1) * v],
                                       self.t2[i * v:(i + 1) * v])

        # // r2(i,a) += t(j,a) VxT_oo(i,j)
        for i in range(o):
            for a in range(v):
                # sum over j index
                self.residual[i * v + a] += np.dot(self.t2[a::v],
                                                   VxT_oo[i * o:(i + 1) * o])

        # print("VxT_v and o contraction")
        for i in range(o):
            for a in range(v):
                dum = 0.
                dum += VxT_v[a] * self.t2[i * v + a]
                dum += VxT_o[i] * self.t2[i * v + a]

                t_t2 = self.t2[i * v + a]
                dum += 2.0 * self.v_iaia[i * v + a] * t_t2 * t_t2
                self.residual[i * v + a] += dum

    def evaluate_sigma(self):
        """
        Evaluate
        """
        o, v = self.o, self.v
        sigma = self.v_iaia.copy()
        for i in range(o):
            for a in range(v):
                sigma[i * v +
                      a] -= 2 * (2 * self.v_iiaa[i * v + a] -
                                 self.v_iaia[i * v + a]) * self.t2[i * v + a]

        for i in range(o):
            for a in range(v):
                dum = 0
                for b in range(v):
                    dum += self.v_abab[a * v + b] * self.t2[i * v + b]
                for j in range(o):
                    dum += self.v_ijij[i * o + j] * self.t2[j * v + a]
                sigma[i * v + a] += dum
        return sigma


class CCD:
    """Coupled cluster doubles for spin hamiltonians
    """
    def __init__(self,
                 molecule=None,
                 *,
                 iter_max=100,
                 e_convergence=1.0E-6,
                 r_convergence=1.0E-6,
                 oei=None,
                 tei=None,
                 n_electrons=None,
                 escf=None,
                 orb_energies=None
    ):
        self.molecule = molecule

        self.t0 = None
        self.sigma = None
        self.iter_max = iter_max
        self.e_convergence = e_convergence
        self.r_convergence = r_convergence


        if molecule is not None:
            # if molecule is defined
            if self.molecule.multiplicity != 1:
                raise ValueError("We are only implementing for \
                closed shell RHF")

            self.n_electrons = molecule.n_electrons

            oei, tei = molecule.get_integrals()
            _, stei = spinorb_from_spatial(oei, tei)
            self.astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)

            self.orb_e = molecule.orbital_energies
            self.sorb_e = np.vstack((self.orb_e, self.orb_e)).flatten(order='F')
            self.scf_energy = molecule.hf_energy

        else:
            if not all(v is not None for v in
                       [oei, tei, n_electrons, escf, orb_energies]):
                raise ValueError("Not all inputs specified")

            self.n_electrons = n_electrons

            self.astei = tei
            self.oei = oei
            self.sorb_e = orb_energies
            self.scf_energy = escf

        self.norbs = self.astei.shape[0]//2
        self.nso = 2 * self.norbs

        self.nocc = self.n_electrons
        self.nvirt = self.nso - self.nocc

        self.n = np.newaxis
        self.o = slice(None, self.nocc)
        self.v = slice(self.nocc, None)
        n, o, v = self.n, self.o, self.v
        self.e_abij = 1 / (-self.sorb_e[v, n, n, n] - self.sorb_e[n, v, n, n] +
                           self.sorb_e[n, n, o, n] + self.sorb_e[n, n, n, o])
        self.t_amp = np.zeros((self.nvirt, self.nvirt, self.nocc, self.nocc))

    def solve_for_amplitudes(self):
        """
        Compute the CCD amplitudes

        Iteration code taken from https://github.com/psi4/psi4numpy/blob/ \
        master/Tutorials/08_CEPA0_and_CCD/8b_CEPA0_and_CCD.ipynb
        """
        t_amp = self.t_amp
        gmo = self.astei
        _, o, v = self.n, self.o, self.v
        e_abij = self.e_abij

        # Initialize energy
        E_CCD = 0.0

        for cc_iter in range(1, self.iter_max + 1):
            E_old = E_CCD

            # Collect terms
            mp2 = gmo[v, v, o, o]
            cepa1 = (1 / 2) * np.einsum('abcd, cdij -> abij', gmo[v, v, v, v],
                                        t_amp, optimize=True)
            cepa2 = (1 / 2) * np.einsum('klij, abkl -> abij', gmo[o, o, o, o],
                                        t_amp, optimize=True)
            cepa3a = np.einsum('akic, bcjk -> abij', gmo[v, o, o, v], t_amp,
                               optimize=True)
            cepa3b = -cepa3a.transpose(1, 0, 2, 3)
            cepa3c = -cepa3a.transpose(0, 1, 3, 2)
            cepa3d = cepa3a.transpose(1, 0, 3, 2)
            cepa3 = cepa3a + cepa3b + cepa3c + cepa3d

            ccd1a_tmp = np.einsum('klcd,bdkl->cb', gmo[o, o, v, v], t_amp,
                                  optimize=True)
            ccd1a = np.einsum("cb,acij->abij", ccd1a_tmp, t_amp, optimize=True)

            ccd1b = -ccd1a.transpose(1, 0, 2, 3)
            ccd1 = -(1 / 2) * (ccd1a + ccd1b)

            ccd2a_tmp = np.einsum('klcd,cdjl->jk', gmo[o, o, v, v], t_amp,
                                  optimize=True)
            ccd2a = np.einsum("jk,abik->abij", ccd2a_tmp, t_amp, optimize=True)

            ccd2b = -ccd2a.transpose(0, 1, 3, 2)
            ccd2 = -(1 / 2) * (ccd2a + ccd2b)


            ccd3_tmp = np.einsum("klcd,cdij->klij", gmo[o, o, v, v], t_amp,
                                 optimize=True)
            ccd3 = (1 / 4) * np.einsum("klij,abkl->abij", ccd3_tmp, t_amp,
                                       optimize=True)

            ccd4a_tmp = np.einsum("klcd,acik->laid", gmo[o, o, v, v], t_amp,
                                  optimize=True)
            ccd4a = np.einsum("laid,bdjl->abij", ccd4a_tmp, t_amp,
                              optimize=True)

            ccd4b = -ccd4a.transpose(0, 1, 3, 2)
            ccd4 = (ccd4a + ccd4b)

            # Update Amplitude
            t_amp_new = e_abij * (
                        mp2 + cepa1 + cepa2 + cepa3 + ccd1 + ccd2 + ccd3 + ccd4)

            # Evaluate Energy
            E_CCD = (1 / 4) * np.einsum('ijab, abij ->', gmo[o, o, v, v],
                                        t_amp_new, optimize=True)
            t_amp = t_amp_new
            dE = E_CCD - E_old
            print('CCD Iteration %3d: Energy = %4.12f dE = %1.5E' % (
            cc_iter, E_CCD, dE))

            if abs(dE) < self.e_convergence:
                print("\nCCD Iterations have converged!")
                break

        print('\nCCD Correlation Energy:    %15.12f' % (E_CCD))
        print('CCD Total Energy:         %15.12f' % (E_CCD + self.scf_energy))
        self.t_amp = t_amp
        self.ccd_energy = E_CCD

    def compute_energy(self, t_amplitudes=None):

        if t_amplitudes is None:
            self.solve_for_amplitudes()
            self.t_amp = self._amplitude_zero_nonpairs(self.t_amp)
            t_amplitudes = self.t_amp

        o, v = self.o, self.v
        gmo = self.astei
        ret = (1 / 4) * np.einsum('ijab, abij ->', gmo[o, o, v, v], \
                                  t_amplitudes, optimize=True) + self.scf_energy
        return ret

    def compute_energy_ncr(self):
        """Amplitude equations determined from pdaggerq"""
        pass

    def _amplitude_zero_nonpairs(self, tamps):
        """Zero out non-pair components of the amplitudes

        the pCCD ansatz is sum_{ia}t_{i}^{a}P_{a}^ P_{i} where
        P_{x}^ = a_{x alpha}^ a_{x beta}^

        :param tamps: (virt, virt, occ, occ) corresponding to t^{ab}_{ij} in
                      spin-orbitals {a,b,i,j}
        """
        pccd_t2_amps = np.zeros_like(tamps)
        amp_count = 0
        for a, b in product(range(self.nvirt), repeat=2):
            for i, j in product(range(self.nocc), repeat=2):
                a_spatial, b_spatial = a // 2, b // 2
                i_spatial, j_spatial = i // 2, j // 2
                if a % 2 == 0 and b % 2 == 1 and i % 2 == 0 and j % 2 == 1 and \
                   a_spatial == b_spatial and i_spatial == j_spatial:
                    pccd_t2_amps[a, b, i, j] = tamps[a, b, i, j]
                    pccd_t2_amps[b, a, j, i] = tamps[a, b, i, j]

                    pccd_t2_amps[a, b, j, i] = -tamps[a, b, i, j]
                    pccd_t2_amps[b, a, i, j] = -tamps[a, b, i, j]

                    amp_count += 1
        return pccd_t2_amps

    def get_pccd_amps(self, tamps):
        """
            for i in range(o):
                for a in range(v):
                    self.residual[i * v + a]
        :param tamps:
        :return:
        """
        pccd_t2_amps = np.zeros((self.nvirt//2, self.nocc//2))
        amp_count = 0
        for a, b in product(range(self.nvirt), repeat=2):
            for i, j in product(range(self.nocc), repeat=2):
                a_spatial, b_spatial = a // 2, b // 2
                i_spatial, j_spatial = i // 2, j // 2
                if a % 2 == 0 and b % 2 == 1 and i % 2 == 0 and j % 2 == 1 and \
                   a_spatial == b_spatial and i_spatial == j_spatial:
                    pccd_t2_amps[a_spatial, i_spatial] = tamps[a, b, i, j]
                    amp_count += 1
        assert np.isclose((self.nvirt//2) * (self.nocc//2), amp_count)
        return pccd_t2_amps

    def pccd_solve(self, starting_amps=None):
        if starting_amps is None:
            t_amp = self.t_amp
        else:
            t_amp = starting_amps

        gmo = self.astei
        _, o, v = self.n, self.o, self.v
        e_abij = self.e_abij

        # Initialize energy
        E_CCD = 0.0

        for cc_iter in range(1, self.iter_max + 1):
            E_old = E_CCD

            t_amp = self._amplitude_zero_nonpairs(t_amp)

            # Collect terms
            mp2 = gmo[v, v, o, o]

            cepa1 = (1 / 2) * np.einsum('abcd, cdij -> abij', gmo[v, v, v, v],
                                        t_amp, optimize=True)
            cepa2 = (1 / 2) * np.einsum('klij, abkl -> abij', gmo[o, o, o, o],
                                        t_amp, optimize=True)
            cepa3a = np.einsum('akic, bcjk -> abij', gmo[v, o, o, v], t_amp,
                               optimize=True)
            cepa3b = -cepa3a.transpose(1, 0, 2, 3)
            cepa3c = -cepa3a.transpose(0, 1, 3, 2)
            cepa3d = cepa3a.transpose(1, 0, 3, 2)
            cepa3 = cepa3a + cepa3b + cepa3c + cepa3d

            ccd1a_tmp = np.einsum('klcd,bdkl->cb', gmo[o, o, v, v], t_amp,
                                  optimize=True)
            ccd1a = np.einsum("cb,acij->abij", ccd1a_tmp, t_amp, optimize=True)

            ccd1b = -ccd1a.transpose(1, 0, 2, 3)
            ccd1 = -(1 / 2) * (ccd1a + ccd1b)

            ccd2a_tmp = np.einsum('klcd,cdjl->jk', gmo[o, o, v, v], t_amp,
                                  optimize=True)
            ccd2a = np.einsum("jk,abik->abij", ccd2a_tmp, t_amp, optimize=True)

            ccd2b = -ccd2a.transpose(0, 1, 3, 2)
            ccd2 = -(1 / 2) * (ccd2a + ccd2b)


            ccd3_tmp = np.einsum("klcd,cdij->klij", gmo[o, o, v, v], t_amp,
                                 optimize=True)
            ccd3 = (1 / 4) * np.einsum("klij,abkl->abij", ccd3_tmp, t_amp,
                                       optimize=True)

            ccd4a_tmp = np.einsum("klcd,acik->laid", gmo[o, o, v, v], t_amp,
                                  optimize=True)
            ccd4a = np.einsum("laid,bdjl->abij", ccd4a_tmp, t_amp,
                              optimize=True)

            ccd4b = -ccd4a.transpose(0, 1, 3, 2)
            ccd4 = (ccd4a + ccd4b)

            # Update Amplitude
            residual = self._amplitude_zero_nonpairs(mp2 + cepa1 + cepa2 + \
                                                     cepa3 + ccd1 + ccd2 + \
                                                     ccd3 + ccd4)
            t_amp_new = -e_abij * residual
            t_amp_new = self._amplitude_zero_nonpairs(t_amp_new)

            # Evaluate Energy
            E_CCD = (1 / 4) * np.einsum('ijab, abij ->', gmo[o, o, v, v],
                                        -t_amp_new, optimize=True)
            t_amp = t_amp_new
            dE = E_CCD - E_old
            print('CCD Iteration %3d: Energy = %4.12f dE = %1.5E' % (
            cc_iter, E_CCD, dE))

            if abs(dE) < self.e_convergence:
                print("\nCCD Iterations have converged!")
                break

        print('\nCCD Correlation Energy:    %15.12f' % (E_CCD))
        print('CCD Total Energy:         %15.12f' % (E_CCD + self.scf_energy))
        self.t_amp = t_amp


# if __name__ == "__main__":
#     import copy
#     np.set_printoptions(linewidth=500)
#     molecule = get_h2o()
#     ccd = CCD(molecule=molecule, iter_max=20)
#     ccd.solve_for_amplitudes()

#     ccd.t_amp = ccd._amplitude_zero_nonpairs(ccd.t_amp)
#     print("Correlation energy from just pCCD amps ", \
#           ccd.compute_energy(ccd.t_amp))
#     print("Correlation energy from just pCCD amps ", \
#           ccd.compute_energy(ccd.t_amp) + ccd.molecule.hf_energy)

#     pccd = CCD(molecule=molecule, iter_max=20)
#     pccd.pccd_solve()

#     pccd_amps = pccd.get_pccd_amps(pccd.t_amp)
#     t2_test = np.zeros((pccd.nocc//2) * (pccd.nvirt//2))
#     for i in range(pccd.nocc//2):
#         for a in range(pccd.nvirt//2):
#             t2_test[i * (pccd.nvirt//2) + a] = pccd_amps[a, i]
#     print(t2_test.shape)
