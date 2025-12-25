import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify.core import ActivationType, NodeType
from neatify.population import Population
from neatify.evolution import mutate_activation

def test_activation_mutation():
    # Create a population
    pop = Population(pop_size=10, num_inputs=2, num_outputs=1)
    
    # Force add some hidden nodes so we have candidates for mutation
    for g in pop.genomes:
        # Manually add a hidden node
        from neatify.core import NodeGene
        g.add_node(NodeGene(100, NodeType.HIDDEN, ActivationType.SIGMOID))
    
    # Check initial activations
    initial_activations = [g.nodes[100].activation for g in pop.genomes]
    print("Initial activations:", initial_activations)
    
    # Mutate with high probability
    pop.config.prob_mutate_activation = 1.0
    
    for g in pop.genomes:
        mutate_activation(g, pop.config)
        
    final_activations = [g.nodes[100].activation for g in pop.genomes]
    print("Final activations:  ", final_activations)
    
    # Check if any changed
    changed = any(i != f for i, f in zip(initial_activations, final_activations))
    if changed:
        print("SUCCESS: Activations mutated.")
    else:
        print("FAILURE: Activations did not mutate (might be bad luck but unlikely with prob 1.0).")

if __name__ == "__main__":
    test_activation_mutation()
