import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from neatify.core import Genome, ActivationType
from neatify.evolution import EvolutionConfig
from neatify.population import Population
from neatify.checkpoint import Checkpoint

def simple_fitness(genomes):
    """Simple fitness function for testing"""
    for genome in genomes:
        genome.fitness = len(genome.connections)

def test_checkpoint_system():
    """Test checkpoint save/load functionality"""
    print("Testing Checkpoint System...")
    
    # Create and evolve a population
    config = EvolutionConfig()
    config.population_size = 50
    
    pop = Population(pop_size=50, num_inputs=2, num_outputs=1, config=config)
    
    # Run a few generations
    for _ in range(5):
        pop.run_generation(simple_fitness)
    
    original_gen = pop.generation
    original_best_fitness = max(g.fitness for g in pop.genomes)
    
    print(f"\nOriginal state:")
    print(f"  Generation: {original_gen}")
    print(f"  Best fitness: {original_best_fitness}")
    print(f"  Population size: {len(pop.genomes)}")
    
    # Save checkpoint
    checkpoint_path = "test_checkpoint.pkl"
    Checkpoint.save(pop, checkpoint_path, metadata={"test": "checkpoint_test"})
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")
    
    # Continue evolution
    for _ in range(3):
        pop.run_generation(simple_fitness)
    
    print(f"\nAfter continuing:")
    print(f"  Generation: {pop.generation}")
    
    # Load checkpoint
    loaded_pop = Checkpoint.load(checkpoint_path)
    
    print(f"\nLoaded state:")
    print(f"  Generation: {loaded_pop.generation}")
    print(f"  Best fitness: {max(g.fitness for g in loaded_pop.genomes)}")
    print(f"  Population size: {len(loaded_pop.genomes)}")
    
    # Verify
    if loaded_pop.generation != original_gen:
        print(f"FAIL: Generation mismatch: {loaded_pop.generation} != {original_gen}")
        return False
    
    if len(loaded_pop.genomes) != len(pop.genomes):
        print(f"FAIL: Population size mismatch")
        return False
    
    print("PASS: Checkpoint restore successful")
    
    # Test best genome save/load
    print("\nTesting best genome save/load...")
    best_genome = max(loaded_pop.genomes, key=lambda g: g.fitness)
    best_path = "test_best.pkl"
    
    Checkpoint.save_best(best_genome, best_path, metadata={"fitness": best_genome.fitness})
    loaded_best = Checkpoint.load_best(best_path)
    
    if loaded_best.fitness != best_genome.fitness:
        print(f"FAIL: Best genome fitness mismatch")
        return False
    
    if len(loaded_best.connections) != len(best_genome.connections):
        print(f"FAIL: Best genome structure mismatch")
        return False
    
    print("PASS: Best genome save/load successful")
    
    # Test JSON export/import
    print("\nTesting JSON export/import...")
    json_path = "test_genome.json"
    
    Checkpoint.export_genome_json(best_genome, json_path)
    imported_genome = Checkpoint.import_genome_json(json_path)
    
    if len(imported_genome.nodes) != len(best_genome.nodes):
        print(f"FAIL: JSON import node count mismatch")
        return False
    
    if len(imported_genome.connections) != len(best_genome.connections):
        print(f"FAIL: JSON import connection count mismatch")
        return False
    
    print("PASS: JSON export/import successful")
    
    # Cleanup
    os.remove(checkpoint_path)
    os.remove(best_path)
    os.remove(json_path)
    
    return True

if __name__ == "__main__":
    success = test_checkpoint_system()
    if success:
        print("\n✓ All checkpoint tests passed!")
    else:
        print("\n✗ Checkpoint tests failed!")
        sys.exit(1)
