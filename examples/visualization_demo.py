"""
NEATify Utility: Visualization Suite Demonstration.

This script showcases the full analytical and visual capabilities of the 
NEATify library by evolving a XOR solver and generating comprehensive
reports on its progress and structure.

Generated Reports:
1. 'viz_species_distribution.png': Population dynamics and fitness trends.
2. 'viz_best_genome.png': Neural architecture of the champion individual.
3. 'viz_complexity.png': Growth patterns of nodes and connections.
4. 'viz_activations.png': Diversity of activation functions used.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import copy
from neatify.core import Genome, ActivationType
from neatify.evolution import EvolutionConfig
from neatify.population import Population
from neatify.pytorch_adapter import NeatModule
from neatify.visualization import (
    plot_species_distribution,
    visualize_genome,
    plot_complexity_evolution,
    plot_activation_distribution
)

# Reference data
XOR_INPUTS = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_OUTPUTS = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

def evaluate_genome(genome):
    """
    Score a genome's fitness for XOR.
    
    Returns:
        float: Fitness score (higher is better).
    """
    try:
        model = NeatModule(genome, use_sparse=False, trainable=False)
        with torch.no_grad():
            predictions = model(XOR_INPUTS)
        mse = torch.mean((predictions - XOR_OUTPUTS) ** 2).item()
        return 1.0 / (1.0 + mse)
    except Exception:
        return 0.0

def fitness_function(genomes):
    """Batch evaluation wrapper."""
    for genome in genomes:
        genome.fitness = evaluate_genome(genome)

def main():
    """Run a short evolution and export all supported plot types."""
    print("=" * 60)
    print("NEATify VISUALIZATION DEMO")
    print("=" * 60)
    
    config = EvolutionConfig()
    config.population_size = 100
    config.prob_mutate_weight = 0.8
    config.prob_add_connection = 0.3
    config.prob_add_node = 0.1
    config.elitism_count = 3
    
    print("\n[1] Evolution Phase")
    pop = Population(pop_size=100, num_inputs=2, num_outputs=1, config=config)
    
    # Track population snapshots for historical plotting
    history = []
    
    for gen in range(30):
        pop.run_generation(fitness_function)
        # Store copies to see evolution over time
        history.append(copy.deepcopy(pop))
        
        if gen % 5 == 0:
            best = max(pop.genomes, key=lambda g: g.fitness)
            print(f"  Gen {gen:2d} | Best Fitness: {best.fitness:.4f}")
    
    best_genome = max(pop.genomes, key=lambda g: g.fitness)
    
    print("\n[2] Visualization Phase")
    print("  Creating Species Distribution...")
    plot_species_distribution(history, "viz_species_distribution.png")
    
    print("  Rendering Best Architecture...")
    visualize_genome(best_genome, "viz_best_genome.png", layout='layered')
    
    print("  Plotting Complexity Growth...")
    plot_complexity_evolution(history, "viz_complexity.png")
    
    print("  Analyzing Activation Functions...")
    plot_activation_distribution(best_genome, "viz_activations.png")
    
    print("\n" + "=" * 60)
    print("GALLERY EXPORTED SUCCESSFULLY")
    print("=" * 60)
    print("Check current directory for the following assets:")
    print("  - viz_species_distribution.png")
    print("  - viz_best_genome.png")
    print("  - viz_complexity.png")
    print("  - viz_activations.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
