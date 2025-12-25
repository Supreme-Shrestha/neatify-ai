import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify.core import Genome, ConnectionGene
from neatify.evolution import EvolutionConfig, mutate_weight

def test_weight_mutation():
    config = EvolutionConfig()
    config.prob_mutate_weight = 1.0 # Always mutate
    config.prob_replace_weight = 0.0 # Never replace, only perturb
    config.weight_mutation_power = 1.0
    
    # Test Gaussian
    config.weight_perturbation_type = 'gaussian'
    genome = Genome(0, 1, 1)
    # Add a connection
    conn = ConnectionGene(0, 1, 0.0, True, 1)
    genome.add_connection(conn)
    
    print("Testing Gaussian Perturbation...")
    values = []
    for _ in range(1000):
        conn.weight = 0.0
        mutate_weight(genome, config)
        values.append(conn.weight)
        
    avg = sum(values) / len(values)
    print(f"Average perturbation (should be ~0): {avg:.4f}")
    # Check if we have values outside uniform range [-1, 1] (gaussian can go beyond)
    outliers = [v for v in values if abs(v) > 1.0]
    print(f"Number of values > 1.0 (expected some): {len(outliers)}")
    
    # Test Uniform
    config.weight_perturbation_type = 'uniform'
    print("\nTesting Uniform Perturbation...")
    values = []
    for _ in range(1000):
        conn.weight = 0.0
        mutate_weight(genome, config)
        values.append(conn.weight)
        
    avg = sum(values) / len(values)
    print(f"Average perturbation (should be ~0): {avg:.4f}")
    # Check if we have values outside uniform range [-1, 1] (should be 0)
    outliers = [v for v in values if abs(v) > 1.000001]
    print(f"Number of values > 1.0 (expected 0): {len(outliers)}")
    
    if len(outliers) == 0:
        print("SUCCESS: Uniform perturbation respected bounds.")
    else:
        print("FAILURE: Uniform perturbation exceeded bounds.")

if __name__ == "__main__":
    test_weight_mutation()
