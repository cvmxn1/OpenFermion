"""
C-like implementaiton of of pCCD equations.

TODO: vectorize further with numpy.  current form is to compare with other codes
"""
from itertools import product
import warnings
import numpy as np

class pCCD:

    def __init__(self,
                 molecule,
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
