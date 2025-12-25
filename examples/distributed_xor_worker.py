"""
Distributed NEAT Example: XOR Problem - Worker Node
"""

import torch
import sys
import os
import argparse

# Add src directory to path to import neatify
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from neatify.distributed import WorkerNode
from neatify import NeatModule

def xor_fitness(genomes):
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    for genome in genomes:
        model = NeatModule(genome)
        outputs = model(inputs)
        loss = torch.sum((outputs - targets) ** 2).item()
        genome.fitness = 4.0 - loss

def main():
    parser = argparse.ArgumentParser(description='Distributed NEAT XOR Solver - Worker Node')
    parser.add_argument('--master-host', type=str, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--worker-id', type=int, required=True)
    args = parser.parse_args()
    
    worker = WorkerNode(args.master_host, args.master_port, args.worker_id, xor_fitness)
    worker.start()

if __name__ == "__main__":
    main()
