"""
NEATify - A modern NEAT (NeuroEvolution of Augmenting Topologies) implementation

Features:
- Pure NEAT evolution with speciation
- PyTorch integration for inference and fine-tuning
- Checkpoint system for long-running experiments
- Visualization tools
- Benchmark environments
- Distributed computing support (optional)
"""

from .core import Genome, NodeGene, ConnectionGene, NodeType, ActivationType
from .evolution import EvolutionConfig, mutate_weight, mutate_add_connection, mutate_add_node, mutate_activation, crossover
from .population import Population, Species
from .pytorch_adapter import NeatModule
from .checkpoint import Checkpoint
from .visualization import (
    plot_species_distribution,
    visualize_genome,
    plot_complexity_evolution,
    plot_activation_distribution,
    draw_genome
)

# Optional distributed computing support
try:
    from .distributed import DistributedPopulation, DistributedConfig, WorkerNode, SystemCoordinator
    _DISTRIBUTED_AVAILABLE = True
except ImportError:
    _DISTRIBUTED_AVAILABLE = False

__version__ = "0.1.3"
__all__ = [
    'Genome', 'NodeGene', 'ConnectionGene', 'NodeType', 'ActivationType',
    'EvolutionConfig', 'mutate_weight', 'mutate_add_connection', 'mutate_add_node', 'mutate_activation', 'crossover',
    'Population', 'Species',
    'NeatModule',
    'Checkpoint',
    'plot_species_distribution', 'visualize_genome', 'plot_complexity_evolution', 'plot_activation_distribution', 'draw_genome'
]

# Add distributed components if available
if _DISTRIBUTED_AVAILABLE:
    __all__.extend(['DistributedPopulation', 'DistributedConfig', 'WorkerNode', 'SystemCoordinator'])
