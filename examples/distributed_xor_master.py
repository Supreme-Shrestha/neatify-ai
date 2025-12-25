"""
Distributed NEAT Example: XOR Problem - Master Node
"""

import sys
import os
import argparse

# Add src directory to path to import neatify
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from neatify.distributed import DistributedPopulation, DistributedConfig
from neatify import visualize_genome

def main():
    parser = argparse.ArgumentParser(description='Distributed NEAT XOR Solver - Master Node')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--pop-size', type=int, default=150)
    args = parser.parse_args()
    
    config = DistributedConfig(host=args.host, port=args.port, min_workers=args.workers)
    pop = DistributedPopulation(args.pop_size, 2, 1, distributed_config=config)
    
    print(f"Starting evolution for {args.generations} generations...")
    try:
        for gen in range(args.generations):
            pop.run_generation(lambda g: None)
            best = max(pop.genomes, key=lambda g: g.fitness)
            print(f"Gen {gen}: Best Fitness = {best.fitness:.4f}")
            if best.fitness > 3.9: break
    finally:
        pop.shutdown()

if __name__ == "__main__":
    main()
