import sys
import os
import torch
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify.core import Genome, ConnectionGene, NodeGene, NodeType, ActivationType
from neatify.pytorch_adapter import NeatModule

def test_fine_tuning():
    print("Testing Parameter Registration and Fine-Tuning...")
    
    # Create a simple genome: Input(0) -> Hidden(2) -> Output(1)
    genome = Genome(0, 1, 1)
    genome.add_node(NodeGene(2, NodeType.HIDDEN, ActivationType.SIGMOID))
    genome.add_connection(ConnectionGene(0, 2, 1.0, True, 1))
    genome.add_connection(ConnectionGene(2, 1, 1.0, True, 2))
    
    # Target: Output should be 0.5 for input 1.0
    # Current: Sigmoid(1*1) = 0.73, Sigmoid(0.73*1) = 0.67
    # So gradients should flow to reduce weights.
    
    x = torch.tensor([[1.0]])
    target = torch.tensor([[0.5]])
    
    print("\n--- Standard Mode ---")
    model = NeatModule(genome, use_sparse=False, trainable=True)
    
    # Check if parameters exist
    params = list(model.parameters())
    print(f"Number of parameters: {len(params)}")
    if len(params) > 0:
        print("PASS: Parameters registered.")
    else:
        print("FAIL: No parameters found.")
        
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Training loop
    initial_loss = None
    for i in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss.item()
            print(f"Initial Loss: {initial_loss:.6f}")
            
    final_loss = loss.item()
    print(f"Final Loss: {final_loss:.6f}")
    
    if final_loss < initial_loss:
        print("PASS: Loss decreased (Standard Mode).")
    else:
        print("FAIL: Loss did not decrease (Standard Mode).")
        
    print("\n--- Sparse Mode ---")
    model_sparse = NeatModule(genome, use_sparse=True, trainable=True)
    
    optimizer_sparse = optim.SGD(model_sparse.parameters(), lr=0.1)
    
    initial_loss_sparse = None
    for i in range(10):
        optimizer_sparse.zero_grad()
        out = model_sparse(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer_sparse.step()
        
        if i == 0:
            initial_loss_sparse = loss.item()
            print(f"Initial Loss: {initial_loss_sparse:.6f}")
            
    final_loss_sparse = loss.item()
    print(f"Final Loss: {final_loss_sparse:.6f}")
    
    if final_loss_sparse < initial_loss_sparse:
        print("PASS: Loss decreased (Sparse Mode).")
    else:
        print("FAIL: Loss did not decrease (Sparse Mode).")

if __name__ == "__main__":
    test_fine_tuning()
