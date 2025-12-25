"""
NEATify Example: Hybrid Evolution and Gradient Descent on XOR.

This advanced demonstration showcases:
1. Pure NEAT evolution: Evolving a topology to solve the XOR problem.
2. PyTorch Integration: Converting a NEAT genome into a trainable NeatModule.
3. Hybrid Learning: Fine-tuning weights of the evolved topology using Adam optimizer.
4. Performance Benchmarking: Comparing evolved performance vs. fine-tuned performance.

Usage:
    python examples/xor_demo.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from neatify.core import Genome, ActivationType
from neatify.evolution import EvolutionConfig, mutate_add_connection, mutate_add_node, mutate_weight
from neatify.population import Population
from neatify.pytorch_adapter import NeatModule

# XOR truth table as PyTorch tensors
XOR_INPUTS = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_OUTPUTS = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

def evaluate_genome(genome):
    """
    Evaluate genome performance on XOR.
    
    Returns:
        float: Fitness score (1.0 / (1.0 + MSE)).
    """
    try:
        model = NeatModule(genome, use_sparse=False, trainable=False)
        with torch.no_grad():
            predictions = model(XOR_INPUTS)
        
        mse = torch.mean((predictions - XOR_OUTPUTS) ** 2).item()
        return 1.0 / (1.0 + mse)
    except Exception:
        return 0.0

def fine_tune_genome(genome, epochs=50, lr=0.1):
    """
    Apply gradient descent to a NEAT genome's weights.
    
    Args:
        genome (Genome): The genome topology to tune.
        epochs (int): Number of training iterations.
        lr (float): Learning rate for Adam.
        
    Returns:
        tuple: (initial_loss, final_loss, model)
    """
    # Create a trainable PyTorch module from the genome
    model = NeatModule(genome, use_sparse=False, trainable=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    initial_loss = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(XOR_INPUTS)
        loss = torch.nn.functional.mse_loss(predictions, XOR_OUTPUTS)
        
        if epoch == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
    
    return initial_loss, loss.item(), model

def print_predictions(model, title):
    """Log model outputs for the XOR truth table."""
    print(f"\n{title}")
    print("=" * 50)
    with torch.no_grad():
        predictions = model(XOR_INPUTS)
    
    for i in range(4):
        inp = XOR_INPUTS[i].tolist()
        target = XOR_OUTPUTS[i].item()
        pred = predictions[i].item()
        print(f"  Input: {inp} -> Target: {target:.1f}, Predicted: {pred:.4f}")

def main():
    """Execute the full evolution + fine-tuning pipeline."""
    print("=" * 70)
    print("XOR PROBLEM SOLVER - NEATify Hybrid Demo")
    print("=" * 70)
    
    # 1. Setup Evolution
    config = EvolutionConfig()
    config.population_size = 150
    config.prob_mutate_weight = 0.8
    config.prob_add_connection = 0.3
    config.prob_add_node = 0.1
    config.elitism_count = 3
    
    print("\n[1] Initializing population...")
    population = Population(
        pop_size=config.population_size,
        num_inputs=2,
        num_outputs=1,
        config=config
    )
    
    # 2. Pure Evolution Phase
    print("\n[2] Running NEAT Evolution...")
    print("-" * 70)
    
    def fitness_function(genomes):
        for genome in genomes:
            genome.fitness = evaluate_genome(genome)
    
    best_gen = 0
    for generation in range(50):
        population.run_generation(fitness_function)
        best = max(population.genomes, key=lambda g: g.fitness)
        
        if generation % 10 == 0:
            avg_fitness = sum(g.fitness for g in population.genomes) / len(population.genomes)
            print(f"  Gen {generation:3d} | Best Fitness: {best.fitness:.4f} | Avg: {avg_fitness:.4f}")
        
        if best.fitness > 0.95:
            print(f"\n✓ Target fitness reached at generation {generation}!")
            best_gen = generation
            break
    
    # 3. Analyze Best Evolved Genome
    best_evolved = max(population.genomes, key=lambda g: g.fitness)
    evolved_model = NeatModule(best_evolved, use_sparse=False, trainable=False)
    
    print(f"\n[3] Best Evolved Topology:")
    print(f"  Nodes: {len(best_evolved.nodes)}")
    print(f"  Connections: {len([c for c in best_evolved.connections.values() if c.enabled])}")
    
    print_predictions(evolved_model, "[4] Evolved Network Results")
    
    # 4. Fine-Tuning Phase
    print("\n[5] Fine-Tuning with Adam Optimizer...")
    print("-" * 70)
    
    initial_loss, final_loss, tuned_model = fine_tune_genome(best_evolved, epochs=100, lr=0.1)
    
    print(f"  Initial MSE: {initial_loss:.6f}")
    print(f"  Final MSE:   {final_loss:.6f}")
    print(f"  Accuracy Gain: {((initial_loss - final_loss) / initial_loss * 100):.1f}% reduction in error")
    
    print_predictions(tuned_model, "[6] Fine-Tuned Network Results")
    
    # 5. Conclusion
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"Topology Search: COMPLETED ({best_gen} generations)")
    print(f"Gradient Polish: COMPLETED (100 epochs)")
    print("\nInsight: NEAT effectively discovers the structure, while gradient descent")
    print("rapidly converges on optimal weights within that structure.")
    print("=" * 70)

if __name__ == "__main__":
    main()
