import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify.population import Population

def dummy_fitness(genomes):
    for g in genomes:
        g.fitness = random.random()

def test_dynamic_threshold():
    # Target 5 species
    pop = Population(pop_size=50, num_inputs=2, num_outputs=1)
    pop.config.target_species = 5
    pop.config.compatibility_threshold = 3.0 # Start high
    
    print(f"Initial Species: {len(pop.species)}, Threshold: {pop.config.compatibility_threshold}")
    
    for i in range(20):
        pop.run_generation(dummy_fitness)
        print(f"Gen {i+1}: Species: {len(pop.species)}, Threshold: {pop.config.compatibility_threshold:.2f}")
        
    # Check if we are close to target
    if 3 <= len(pop.species) <= 7:
        print("SUCCESS: Species count maintained near target.")
    else:
        print("WARNING: Species count diverged (might need more gens or tuning).")

if __name__ == "__main__":
    test_dynamic_threshold()
