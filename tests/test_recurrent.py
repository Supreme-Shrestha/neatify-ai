import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify.core import Genome, ConnectionGene, NodeGene, NodeType, ActivationType
from neatify.pytorch_adapter import NeatModule

def test_recurrent_network():
    # Create a simple recurrent network: Input -> Hidden <-> Hidden -> Output
    genome = Genome(0, 1, 1)
    # Input 0, Output 1
    # Add Hidden 2
    genome.add_node(NodeGene(2, NodeType.HIDDEN, ActivationType.IDENTITY))
    # Set Output 1 to IDENTITY for easier testing
    genome.nodes[1].activation = ActivationType.IDENTITY
    
    # Connections
    # 0 -> 2 (weight 1.0)
    genome.add_connection(ConnectionGene(0, 2, 1.0, True, 1))
    # 2 -> 2 (weight 0.5) - Self loop / Recurrent
    genome.add_connection(ConnectionGene(2, 2, 0.5, True, 2))
    # 2 -> 1 (weight 1.0)
    genome.add_connection(ConnectionGene(2, 1, 1.0, True, 3))
    
    model = NeatModule(genome)
    
    print(f"Is Recurrent: {model.is_recurrent}")
    if not model.is_recurrent:
        print("FAILURE: Did not detect recurrence.")
        return

    # Test inputs (Gauss-Seidel updates)
    # Step 1: Input 1.0. 
    #   Hidden (2) sees Input(0)=1.0 + Hidden(2)=0.0 (prev). Total=1.0. New Hidden=1.0.
    #   Output (1) sees Hidden(2)=1.0 (new). Total=1.0. New Output=1.0.
    # Step 2: Input 0.0.
    #   Hidden (2) sees Input(0)=0.0 + Hidden(2)=1.0 (prev). Total=0.5. New Hidden=0.5.
    #   Output (1) sees Hidden(2)=0.5 (new). Total=0.5. New Output=0.5.
    # Step 3: Input 0.0.
    #   Hidden (2) sees Input(0)=0.0 + Hidden(2)=0.5 (prev). Total=0.25. New Hidden=0.25.
    #   Output (1) sees Hidden(2)=0.25 (new). Total=0.25. New Output=0.25.
    
    inputs = torch.tensor([[1.0], [0.0], [0.0]])
    
    # Run step by step manually
    print("\nRunning step-by-step:")
    model.reset()
    for i in range(3):
        inp = inputs[i:i+1] # Batch size 1
        out = model(inp, steps=1)
        print(f"Step {i+1}: Input={inp.item()}, Output={out.item():.4f}")
        
    # Verify values
    model.reset()
    out1 = model(inputs[0:1], steps=1).item()
    out2 = model(inputs[1:2], steps=1).item()
    out3 = model(inputs[2:3], steps=1).item()
    
    expected = [1.0, 0.5, 0.25]
    actual = [out1, out2, out3]
    
    print(f"\nExpected: {expected}")
    print(f"Actual:   {actual}")
    
    if all(abs(a - e) < 1e-5 for a, e in zip(actual, expected)):
        print("SUCCESS: Recurrent dynamics correct.")
    else:
        print("FAILURE: Values mismatch.")

if __name__ == "__main__":
    test_recurrent_network()
