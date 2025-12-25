"""
NEATify Benchmark: Function Approximation (Regression).

This script tests NEAT's ability to evolve neural networks that approximate
various mathematical functions of increasing complexity:
1. Sine Wave: Continuous, periodic function.
2. Polynomial: 3rd degree polynomial with local minima/maxima.
3. Multi-modal: Composition of sine and cosine waves.

Key Features:
- Dataset generation for training and validation.
- Fitness based on inverse Mean Squared Error (MSE).
- Visualization: Overlaying NEAT predictions against ground truth.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from neatify.core import Genome, ActivationType
from neatify.evolution import EvolutionConfig
from neatify.population import Population
from neatify.pytorch_adapter import NeatModule

def generate_data(function_type='sine', num_samples=100):
    """
    Create a dataset for the specified mathematical function.
    
    Args:
        function_type (str): 'sine', 'polynomial', or 'multimodal'.
        num_samples (int): Number of points to sample between [-3, 3].
        
    Returns:
        tuple: (x_tensor, y_tensor) for training.
    """
    x = np.linspace(-3, 3, num_samples)
    
    if function_type == 'sine':
        y = np.sin(x)
    elif function_type == 'polynomial':
        y = 0.5 * x**3 - 2 * x**2 + x + 1
    elif function_type == 'multimodal':
        y = np.sin(x) * np.cos(2*x) + 0.5 * x
    else:
        raise ValueError(f"Unsupported function type: {function_type}")
    
    # Return as column vectors for PyTorch compatibility
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

def evaluate_genome_regression(genome, x_train, y_train):
    """
    Evaluate a genome based on its regression accuracy.
    
    Fitness = 1.0 / (1.0 + MSE).
    
    Args:
        genome (Genome): Network to test.
        x_train (Tensor): Input coordinates.
        y_train (Tensor): Target outputs.
        
    Returns:
        float: Fitness score.
    """
    try:
        model = NeatModule(genome, use_sparse=False, trainable=False)
        
        with torch.no_grad():
            predictions = model(x_train)
        
        mse = torch.mean((predictions - y_train) ** 2).item()
        return 1.0 / (1.0 + mse)
    except Exception:
        return 0.0

def test_function_approximation(function_type='sine', generations=50):
    """
    Orchestrate a complete evolution run for a specific function.
    
    Outputs a comparison plot to a .png file.
    
    Args:
        function_type (str): Target function name.
        generations (int): Maximum evolutionary iterations.
        
    Returns:
        tuple: (best_genome, final_mse)
    """
    print(f"\nTargeting: {function_type.upper()}")
    
    x_train, y_train = generate_data(function_type, num_samples=100)
    
    config = EvolutionConfig()
    config.population_size = 150
    config.prob_mutate_weight = 0.8
    config.prob_add_connection = 0.3
    config.prob_add_node = 0.1
    config.elitism_count = 3
    
    pop = Population(pop_size=150, num_inputs=1, num_outputs=1, config=config)
    
    def fitness_function(genomes):
        for genome in genomes:
            genome.fitness = evaluate_genome_regression(genome, x_train, y_train)
    
    best_fitness_history = []
    for gen in range(generations):
        pop.run_generation(fitness_function)
        best = max(pop.genomes, key=lambda g: g.fitness)
        best_fitness_history.append(best.fitness)
        
        if gen % 10 == 0:
            avg_fitness = sum(g.fitness for g in pop.genomes) / len(pop.genomes)
            print(f"  Gen {gen:3d} | Best Fitness: {best.fitness:.4f} | Avg: {avg_fitness:.4f}")
    
    best_genome = max(pop.genomes, key=lambda g: g.fitness)
    
    # Validation Phase
    x_test = torch.linspace(-3, 3, 200).unsqueeze(1)
    if function_type == 'sine':
        y_true = torch.sin(x_test)
    elif function_type == 'polynomial':
        y_true = 0.5 * x_test**3 - 2 * x_test**2 + x_test + 1
    elif function_type == 'multimodal':
        y_true = torch.sin(x_test) * torch.cos(2*x_test) + 0.5 * x_test
    
    model = NeatModule(best_genome, use_sparse=False, trainable=False)
    with torch.no_grad():
        y_pred = model(x_test)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_test.numpy(), y_true.numpy(), 'b-', linewidth=2, label='Ground Truth')
    plt.plot(x_test.numpy(), y_pred.numpy(), 'r--', linewidth=2, label='NEAT (id: {})'.format(best_genome.id))
    plt.scatter(x_train.numpy(), y_train.numpy(), c='gray', s=10, alpha=0.3, label='Samples')
    plt.title(f'{function_type.capitalize()} Approximation')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.subplot(1, 2, 2)
    plt.plot(best_fitness_history, 'g-')
    plt.title('Fitness Convergence')
    plt.xlabel('Generation')
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'function_approx_{function_type}.png')
    plt.close()
    
    final_mse = torch.mean((y_pred - y_true) ** 2).item()
    print(f"✓ Results saved to 'function_approx_{function_type}.png'")
    print(f"✓ Performance: MSE = {final_mse:.6f}")
    
    return best_genome, final_mse

def main():
    """Benchmark NEAT against a suite of functions."""
    print("=" * 60)
    print("FUNCTION APPROXIMATION BENCHMARK")
    print("=" * 60)
    
    results = {}
    for func_type in ['sine', 'polynomial', 'multimodal']:
        best_genome, mse = test_function_approximation(func_type, generations=50)
        results[func_type] = mse
    
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 60)
    for func, mse in results.items():
        print(f"{func.capitalize():12s} | MSE: {mse:.6f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
