"""
NEATify Example: Solving the XOR problem.

This script demonstrates the most basic usage of NEATify:
1. Defining a fitness function.
2. Initializing a Population.
3. Running the evolutionary loop until a solution is found.
4. Visualizing the resulting neural network.
"""

import torch
import sys
import os

# Add parent directory to path to import neatify
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify import Population, NeatModule, visualize_genome

def xor_fitness(genomes):
    """
    Evaluates a list of genomes on the XOR problem.
    
    Fitness is calculated as (4.0 - Sum Squared Error).
    A perfect solver will have a fitness close to 4.0.
    
    Args:
        genomes (List[Genome]): The current population members.
    """
    # XOR inputs and outputs
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    
    for genome in genomes:
        model = NeatModule(genome)
        
        try:
            outputs = model(inputs)
            # MSE-based fitness calculation
            loss = torch.sum((outputs - targets) ** 2).item()
            genome.fitness = 4.0 - loss
        except Exception as e:
            # Handle potential runtime errors (e.g. invalid topologies)
            genome.fitness = 0.0

def main():
    """Main execution loop for XOR solver."""
    # Initialize population: 150 members, 2 inputs (A, B), 1 output (Result)
    pop = Population(pop_size=150, num_inputs=2, num_outputs=1)
    
    generations = 50
    print(f"Starting evolution for {generations} generations...")
    
    for gen in range(generations):
        pop.run_generation(xor_fitness)
        
        best_genome = max(pop.genomes, key=lambda g: g.fitness)
        print(f"Generation {gen}: Best Fitness = {best_genome.fitness:.4f}")
        
        # Stop if we found a good enough solution
        if best_genome.fitness > 3.9:
            print("\nSolution Found!")
            print(f"Best Genome ID: {best_genome.id}")
            visualize_genome(best_genome, save_path="xor_solution.png")
            print("Visualization saved to 'xor_solution.png'")
            break
            
    print("Evolution complete.")

if __name__ == "__main__":
    main()
