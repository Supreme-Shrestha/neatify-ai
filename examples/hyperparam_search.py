"""
NEATify Utility: Hyperparameter Optimization.

This script provides tools to search for optimal NEAT configurations using:
1. Grid Search: Exhaustive search over a predefined parameter space.
2. Random Search: Stochastic sampling of parameters for faster discovery.

The script targets the XOR problem by default and ranks configurations
based on their 'Solve Rate' and efficiency (generations to converge).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import itertools
from neatify.core import Genome, ActivationType
from neatify.evolution import EvolutionConfig
from neatify.population import Population
from neatify.pytorch_adapter import NeatModule

# Evaluation dataset
XOR_INPUTS = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_OUTPUTS = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

def evaluate_genome(genome):
    """
    Standard XOR evaluation logic.
    
    Returns:
        float: Inverse MSE fitness.
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
    """Fitness wrapper for multiple genomes."""
    for genome in genomes:
        genome.fitness = evaluate_genome(genome)

def run_experiment(config, max_generations=50):
    """
    Execute a single NEAT run with a specific config.
    
    Args:
        config (EvolutionConfig): The settings to test.
        max_generations (int): Cutoff limit for evolution.
        
    Returns:
        dict: Results including best fitness, speed, and success status.
    """
    pop = Population(pop_size=config.population_size, num_inputs=2, num_outputs=1, config=config)
    
    best_fitness = 0.0
    generations_to_solve = max_generations
    
    for gen in range(max_generations):
        pop.run_generation(fitness_function)
        
        best = max(pop.genomes, key=lambda g: g.fitness)
        if best.fitness > best_fitness:
            best_fitness = best.fitness
        
        if best.fitness > 0.95:  # Solved threshold
            generations_to_solve = gen
            break
    
    return {
        'best_fitness': best_fitness,
        'generations_to_solve': generations_to_solve,
        'solved': best_fitness > 0.95
    }

def grid_search():
    """
    Perform an exhaustive search over a cross-product of parameters.
    
    Best used when the search space is small and you need deterministic results.
    """
    print("=" * 70)
    print("HYPERPARAMETER TUNING - GRID SEARCH")
    print("=" * 70)
    
    param_grid = {
        'population_size': [50, 100, 150],
        'prob_add_connection': [0.2, 0.3, 0.4],
        'prob_add_node': [0.05, 0.1, 0.15],
        'elitism_count': [2, 3, 5],
        'target_species': [3, 5, 7]
    }
    
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"\nEvaluating {len(combinations)} parameter sets...")
    
    results = []
    
    for i, combo in enumerate(combinations):
        config = EvolutionConfig()
        for key, value in zip(keys, combo):
            setattr(config, key, value)
        
        # Average results over 3 trials to reduce stochastic noise
        trial_results = [run_experiment(config, max_generations=50) for _ in range(3)]
        
        avg_fitness = sum(r['best_fitness'] for r in trial_results) / 3
        avg_gens = sum(r['generations_to_solve'] for r in trial_results) / 3
        solve_rate = sum(1 for r in trial_results if r['solved']) / 3
        
        results.append({
            'config': dict(zip(keys, combo)),
            'avg_fitness': avg_fitness,
            'avg_generations': avg_gens,
            'solve_rate': solve_rate
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(combinations)}")
    
    # Ranking
    results.sort(key=lambda x: (-x['solve_rate'], x['avg_generations']))
    
    print("\n" + "=" * 70)
    print("RANKED LEADERBOARD (Top 5)")
    print("=" * 70)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}. Solve Rate: {res['solve_rate']:.1%} | Avg Gens: {res['avg_generations']:.1f}")
        print(f"   Config: {res['config']}")
    
    return results

def random_search(num_trials=50):
    """
    Sample configurations randomly from distributions.
    
    More efficient for high-dimensional parameter spaces.
    """
    import random
    
    print("=" * 70)
    print("HYPERPARAMETER TUNING - RANDOM SEARCH")
    print("=" * 70)
    
    results = []
    
    for i in range(num_trials):
        config = EvolutionConfig()
        config.population_size = random.choice([50, 75, 100, 150])
        config.prob_add_connection = random.uniform(0.1, 0.5)
        config.prob_add_node = random.uniform(0.01, 0.2)
        config.elitism_count = random.choice([1, 2, 3, 5])
        config.target_species = random.choice([2, 5, 10])
        config.prob_mutate_weight = random.uniform(0.5, 0.9)
        config.weight_mutation_power = random.uniform(0.1, 1.0)
        
        result = run_experiment(config, max_generations=50)
        
        results.append({
            'config': {k: getattr(config, k) for k in ['population_size', 'prob_add_connection', 'prob_add_node', 'elitism_count', 'target_species', 'prob_mutate_weight', 'weight_mutation_power']},
            'best_fitness': result['best_fitness'],
            'generations_to_solve': result['generations_to_solve'],
            'solved': result['solved']
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_trials} samples.")
    
    results.sort(key=lambda x: (-x['solved'], x['generations_to_solve']))
    
    print("\n" + "=" * 70)
    print("GLOBAL BEST RESULTS")
    print("=" * 70)
    for i, res in enumerate(results[:3], 1):
        print(f"{i}. Fitness: {res['best_fitness']:.4f}, Gens: {res['generations_to_solve']}")
        print(f"   Config: {res['config']}")
    
    return results

def main():
    """CLI Entry point for tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated NEAT Tuning')
    parser.add_argument('--method', choices=['grid', 'random'], default='random')
    parser.add_argument('--trials', type=int, default=20)
    
    args = parser.parse_args()
    
    if args.method == 'grid':
        grid_search()
    else:
        random_search(num_trials=args.trials)

if __name__ == "__main__":
    main()
