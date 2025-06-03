import numpy as np

class IsingModel:
    def __init__(self, size, temperature, interaction_strength, external_field):
        '''
        Initialises the class by saving size, temperature, interaction_strength, external_field as class attributes
        And initialises a size by size matrix of random +1s and -1s to represent the lattice of spins
        '''
        self.size = size
        if temperature == 0:
            self.T = 1e-6
        else:
            self.T = temperature
        self.J = interaction_strength
        self.h = external_field
        # Initialise the lattice of spins
        self.lattice = np.random.choice([1,-1], size=(size, size))

    def _site_spin_interaction(self, site: tuple) -> float:
        '''
        Finds the spin interaction of a given site of spins with given coordinate in the lattice of spins using:
        site_interaction = sum_{<i,j>}(S_i * S_j)
        <i,j> is a sum over nearest neighbours, but I will only be considering contributions from the spin directly right and directly below to avoid unnecessary sums when this function is used elsewhere.
        Note that this asserts periodic boundary conditions
        Inputs:
            - site: the index of the centre of the site
        Outputs:
            - site_interaction: a float determining the local interaction term of the site
        '''
        # Get the spin state of the spin at the centre of the site
        spin = self.lattice[*site]

        # Compare spin alignment with neighbouring spins
        site_interaction = spin*self.lattice[(site[0]+1)%len(self.lattice), site[1]] + spin*self.lattice[site[0], (site[1]+1)%len(self.lattice[0])]
        return site_interaction

    def _site_metropolis_step(self, site: tuple) -> None:
        '''
        For a given site, this function decides whether the spin should be flipped or not based on the energy difference between the flipped and un-flipped site using the Metropolis algorithm.
        Inputs:
            - site: The site to decide to flip
        Outputs:
            - None: however self.lattice is amended based on the spin at the site being flipped or not 
        '''
        # Calculate the relevant energy terms involving the given site
        E_same = self.J*(self._site_spin_interaction(site) + self._site_spin_interaction((site[0]-1, site[1])) + self._site_spin_interaction((site[0], site[1]-1))) + self.h*self.lattice[*site]

        # Flip the given site
        self.lattice[*site] = -1*self.lattice[*site]

        # Calculate the relevant energy terms involving the given flipped site
        E_flipped = self.J*(self._site_spin_interaction(site) + self._site_spin_interaction((site[0]-1, site[1])) + self._site_spin_interaction((site[0], site[1]-1))) + self.h*self.lattice[*site]

        # Calculate the energy change
        dE = E_flipped - E_same

        # If it is not energetically preferable, do not flip the spin
        # However, with probability exp(-dE/(k_B*T)), flip the spin anyway
        if dE > 0 or np.random.rand() > np.exp(-dE/self.T): # Taking k_B = 1
            self.lattice[*site] = -1*self.lattice[*site]
            
    def step(self):
        '''
        Picks a random site and applies a step of the Metropolis algorithm to it
        '''
        # Randomly choose a site
        i,j = np.random.randint(0, self.size, size=2)

        # Step the Metropolis algorithm on that site
        self._site_metropolis_step((i,j))

    def total_energy(self) -> float:
        '''
        Calculates the total energy of the system using the Ising model hamiltonian:
        H = J*sum_{<i,j>} S_i * S_j  + h*sum_{i} S_i
        _site_spin_interaction is designed so that the sum over neighbours sum_{<i,j>} is simply a sum over all sites.
        Output:
            - energy: a float determining the total energy of the system
        '''
        # Sum over all spin interactions and all spins for each term in the hamiltonian
        total_interaction = 0
        total_spins = 0
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice[0])):
                total_interaction += self._site_spin_interaction(lattice=self.lattice, site=(i,j))
                site_spins += self.lattice[i,j]

        # Evaluate the total energy using the total spin interactions and total spins
        energy = self.J*total_interaction + self.h*total_spins
        return energy


if __name__ == "__main__":
    SIZE = 100
    INTERACTION_STRENGTH = J = 1
    TEMPERATURE = T = 0
    EXTERNAL_FIELD = H = 0

    Model = IsingModel(SIZE, T, J, H)
    Model.step()