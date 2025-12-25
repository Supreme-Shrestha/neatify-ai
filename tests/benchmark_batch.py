import sys
import os
import time
import torch
import networkx as nx
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatify.core import Genome, ConnectionGene, NodeGene, NodeType, ActivationType
from neatify.pytorch_adapter import NeatModule

def create_large_genome(num_inputs=10, num_outputs=10, num_hidden=500, density=0.1):
    genome = Genome(0, num_inputs, num_outputs)
    
    # Add hidden nodes
    for i in range(num_hidden):
        genome.add_node(NodeGene(num_inputs + num_outputs + i, NodeType.HIDDEN, ActivationType.SIGMOID))
        
    # Add random connections
    nodes = list(genome.nodes.keys())
    num_connections = int(len(nodes) * len(nodes) * density)
    print(f"Creating genome with {len(nodes)} nodes and ~{num_connections} connections...")
    
    # To ensure feedforward (for fair comparison initially), we only connect i -> j where i < j
    # (assuming IDs are sorted roughly by depth, or we just enforce ID order)
    # Actually, let's just create random connections and let the adapter handle cycles (it might fall back to recurrent)
    # But for "Batch Optimization" we mostly care about the matrix op speedup which applies to both.
    # Let's try to make it feedforward to test the layered approach.
    
    sorted_nodes = sorted(nodes)
    
    count = 0
    attempts = 0
    while count < num_connections and attempts < num_connections * 2:
        attempts += 1
        idx1 = random.randint(0, len(nodes)-2)
        idx2 = random.randint(idx1+1, len(nodes)-1)
        
        in_node = sorted_nodes[idx1]
        out_node = sorted_nodes[idx2]
        
        # Avoid input->input or output->input (already handled by sorting if inputs are first)
        if genome.nodes[in_node].type == NodeType.OUTPUT: continue
        if genome.nodes[out_node].type == NodeType.INPUT: continue
        
        genome.add_connection(ConnectionGene(in_node, out_node, random.uniform(-1, 1), True, count))
        count += 1
        
    return genome

def benchmark():
    genome = create_large_genome(num_inputs=20, num_outputs=5, num_hidden=200, density=0.05)
    
    print("\n--- Standard Mode ---")
    model = NeatModule(genome, use_sparse=False)
    print(f"Model created. Recurrent: {model.is_recurrent}")
    
    batch_size = 128
    inputs = torch.randn(batch_size, 20)
    
    # Warmup
    with torch.no_grad():
        model(inputs)
        
    start_time = time.time()
    iterations = 50
    with torch.no_grad():
        for _ in range(iterations):
            model(inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    print(f"Average forward pass time: {avg_time*1000:.2f} ms")
    
    print("\n--- Sparse Mode ---")
    model_sparse = NeatModule(genome, use_sparse=True)
    print(f"Model created. Recurrent: {model_sparse.is_recurrent}")
    
    # Warmup
    with torch.no_grad():
        model_sparse(inputs)
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model_sparse(inputs)
    end_time = time.time()
    
    avg_time_sparse = (end_time - start_time) / iterations
    print(f"Average forward pass time: {avg_time_sparse*1000:.2f} ms")
    
    # Verify correctness (approximate)
    out_std = model(inputs)
    out_sparse = model_sparse(inputs)
    diff = torch.abs(out_std - out_sparse).max().item()
    print(f"\nMax difference between modes: {diff:.6f}")
    if diff < 1e-4:
        print("SUCCESS: Outputs match.")
    else:
        print("WARNING: Outputs differ (might be due to float precision or order).")

if __name__ == "__main__":
    benchmark()
