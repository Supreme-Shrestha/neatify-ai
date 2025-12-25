import sys
import os
import torch
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify.core import Genome, ConnectionGene, NodeGene, NodeType, ActivationType
from neatify.pytorch_adapter import NeatModule

def test_activations():
    print("Testing Activation Mapping...")
    
    # Create a genome with one node of each type
    # Input -> Node(Act) -> Output
    # We'll test each activation type individually
    
    activations_to_test = [
        (ActivationType.SIGMOID, torch.sigmoid),
        (ActivationType.RELU, torch.relu),
        (ActivationType.TANH, torch.tanh),
        (ActivationType.IDENTITY, lambda x: x),
        (ActivationType.LEAKY_RELU, torch.nn.functional.leaky_relu),
        (ActivationType.ELU, torch.nn.functional.elu)
    ]
    
    for act_type, torch_func in activations_to_test:
        print(f"Testing {act_type.name}...")
        
        genome = Genome(0, 1, 1)
        # Hidden node with specific activation
        genome.add_node(NodeGene(2, NodeType.HIDDEN, act_type))
        # Set Output to IDENTITY to observe Hidden output directly
        genome.nodes[1].activation = ActivationType.IDENTITY
        
        # 0 -> 2 -> 1
        genome.add_connection(ConnectionGene(0, 2, 1.0, True, 1))
        genome.add_connection(ConnectionGene(2, 1, 1.0, True, 2))
        
        # Test Standard Mode
        model = NeatModule(genome, use_sparse=False)
        x = torch.tensor([[-1.0], [0.0], [1.0]])
        out = model(x)
        
        # Expected: f(x) because weights are 1.0 and output is identity
        expected = torch_func(x)
        
        diff = torch.abs(out - expected).max().item()
        if diff < 1e-5:
            print(f"  Standard: PASS (diff={diff:.6f})")
        else:
            print(f"  Standard: FAIL (diff={diff:.6f})")
            
        # Test Sparse Mode
        model_sparse = NeatModule(genome, use_sparse=True)
        out_sparse = model_sparse(x)
        
        diff_sparse = torch.abs(out_sparse - expected).max().item()
        if diff_sparse < 1e-5:
            print(f"  Sparse:   PASS (diff={diff_sparse:.6f})")
        else:
            print(f"  Sparse:   FAIL (diff={diff_sparse:.6f})")

if __name__ == "__main__":
    test_activations()
