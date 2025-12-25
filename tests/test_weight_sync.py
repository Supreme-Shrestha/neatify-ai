import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from neatify.core import Genome, ConnectionGene, NodeGene, NodeType, ActivationType
from neatify.pytorch_adapter import NeatModule

def test_weight_synchronization():
    """Test bidirectional weight synchronization between Genome and PyTorch"""
    print("Testing Weight Synchronization...")
    
    # Create a simple genome
    genome = Genome(0, 2, 1)
    genome.add_node(NodeGene(3, NodeType.HIDDEN, ActivationType.SIGMOID))
    genome.add_connection(ConnectionGene(0, 3, 0.5, True, 1))
    genome.add_connection(ConnectionGene(1, 3, 0.5, True, 2))
    genome.add_connection(ConnectionGene(3, 2, 0.5, True, 3))
    
    # Store original weights
    original_weights = [conn.weight for conn in genome.connections.values() if conn.enabled]
    print(f"\nOriginal genome weights: {original_weights}")
    
    # Convert to PyTorch and train
    model = NeatModule(genome, trainable=True)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Simple training loop
    x = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([[0.0]])
    
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    
    # Get trained weights
    trained_weights = model.weight_values.detach().tolist()
    print(f"Trained PyTorch weights: {trained_weights}")
    
    # Sync back to genome
    model.update_genome_weights()
    
    # Verify synchronization
    synced_weights = [conn.weight for conn in genome.connections.values() if conn.enabled]
    print(f"Synced genome weights: {synced_weights}")
    
    # Check that weights changed
    weights_changed = any(abs(orig - synced) > 0.01 for orig, synced in zip(original_weights, synced_weights))
    if not weights_changed:
        print("FAIL: Weights did not change during training")
        return False
    
    # Check that synced weights match trained weights
    weights_match = all(abs(trained - synced) < 1e-6 for trained, synced in zip(trained_weights, synced_weights))
    if not weights_match:
        print("FAIL: Synced weights do not match trained weights")
        return False
    
    print("PASS: Weights successfully synchronized")
    
    # Test round-trip: Genome -> PyTorch -> train -> Genome -> PyTorch
    print("\nTesting round-trip...")
    model2 = NeatModule(genome, trainable=False)
    
    with torch.no_grad():
        output1 = model(x)
        output2 = model2(x)
    
    if abs(output1.item() - output2.item()) < 1e-6:
        print("PASS: Round-trip successful - outputs match")
        return True
    else:
        print(f"FAIL: Round-trip failed - outputs differ: {output1.item()} vs {output2.item()}")
        return False

if __name__ == "__main__":
    success = test_weight_synchronization()
    if success:
        print("\n✓ All weight synchronization tests passed!")
    else:
        print("\n✗ Weight synchronization tests failed!")
        sys.exit(1)
