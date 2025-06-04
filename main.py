import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

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
        <i,j> is a sum over nearest neighbours, considering contributions from the spin directly right, left, below, and above.
        Note that this asserts periodic boundary conditions.
        Inputs:
            - site: the index of the centre of the site
        Outputs:
            - site_interaction: a float determining the local interaction term of the site
        '''
        # Get the spin state of the spin at the centre of the site
        spin = self.lattice[*site]

        # Compare spin alignment with neighbouring spins
        site_interaction = spin*(self.lattice[(site[0] + 1)%self.size, site[1]] + self.lattice[site[0], (site[1] + 1)%self.size] + self.lattice[(site[0] - 1)%self.size, site[1]] + self.lattice[site[0], (site[1] - 1)%self.size])
        return site_interaction

    def _site_metropolis_step(self, site: tuple) -> None:
        '''
        For a given site, this function decides whether the spin should be flipped or not based on the energy difference between the flipped and un-flipped site using the Metropolis algorithm.
        Inputs:
            - site: The site to decide to flip
        Outputs:
            - None: however self.lattice is amended based on the spin at the site being flipped or not 
        '''
        # Calculate the energy change if the spin was flipped
        # The change in energy is only due to the four interaction terms of neighbouring spins, and the external field term
        dE = 2*(self.J*self._site_spin_interaction(site) + self.h*self.lattice[*site])

        # If it is energetically preferable (dE < 0), flip the spin
        # However, even if dE > 0, with probability exp(-dE/(k_B*T)), flip the spin anyway
        if dE < 0 :
            self.lattice[*site] = -1*self.lattice[*site]
        elif np.random.rand() < np.exp(-dE/self.T):
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
        # Factor of half to account for double counting spin interactions
        energy = -self.J*total_interaction/2 - self.h*total_spins
        return energy



def animate_ising_model(model):
    '''
    Animates the Ising model as a grid of coloured tiles with blue representing spin 1 and red spin -1.
    Each frame runs model.size Metropolis steps.
    Matplotlib isn't very good at rendering this very quickly for large number of tiles, so it's essentially bounded to a 50x50 grid for reasonably fast rendering.
    '''
    fig, ax = plt.subplots()
    ax.set_xlim(0, model.size)
    ax.set_ylim(0, model.size)

    tiles = np.empty((model.size, model.size), dtype=object)
    for i in range(len(tiles)):
        for j in range(len(tiles[0])):
            tiles[i,j] = patches.Rectangle((j,model.size-1-i), 1, 1, edgecolor="none", facecolor=("red" if Model.lattice[i,j] == -1 else "blue"))
            ax.add_patch(tiles[i,j])
    
    def update(frame):
        for _ in range(model.size**3):
            Model.step()
        
        for i in range(len(tiles)):
            for j in range(len(tiles[0])):
                tiles[i,j].set_facecolor(("red" if Model.lattice[i,j] == -1 else "blue"))
        
        return tiles.flatten()
    
    ani = FuncAnimation(fig, update, frames=1000, interval=1, blit=True)
    plt.show()

if __name__ == "__main__":
    SIZE = 50
    INTERACTION_STRENGTH = J = 1
    TEMPERATURE = T = 2.5
    EXTERNAL_FIELD = H = 0

    # Generate model
    Model = IsingModel(SIZE, T, J, H)
    
    # Animate model
    animate_ising_model(Model)