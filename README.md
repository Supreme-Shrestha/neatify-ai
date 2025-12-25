# NEATify

A modern, production-ready implementation of NEAT (NeuroEvolution of Augmenting Topologies) with PyTorch integration.

## Features

> [!TIP]
> **New to NEAT?** Check out our [Comprehensive User Guide](GUIDE.md) for a step-by-step tutorial!

### Core Evolution
- ✅ **NEAT Algorithm**: Complete implementation with speciation and innovation tracking
- ✅ **Dynamic Speciation**: Automatic compatibility threshold adjustment
- ✅ **Advanced Mutations**: Weight perturbation, topology mutations, activation mutations
- ✅ **Elitism**: Preserve best genomes across generations
- ✅ **Configurable**: Extensive hyperparameter control

### PyTorch Integration
- ✅ **Efficient Inference**: Convert genomes to PyTorch modules
- ✅ **Recurrent Networks**: Full support for cyclic topologies
- ✅ **Sparse Mode**: Optimized batch processing for large networks
- ✅ **Gradient Fine-Tuning**: Register weights as `nn.Parameter` for hybrid evolution + learning
- ✅ **Bidirectional Sync**: Update genomes with trained weights

### Production Features
- ✅ **Checkpointing**: Save/load population state
- ✅ **Serialization**: JSON and Pickle export/import
- ✅ **Visualization**: Species plots, genome graphs, complexity tracking
- ✅ **Benchmarks**: XOR, LunarLander, function approximation
- ✅ **Hyperparameter Tuning**: Grid and random search utilities

## Installation

```bash
pip install torch numpy matplotlib networkx gymnasium
```

## Quick Start

```python
from neatify import Population, EvolutionConfig, NeatModule
import torch

# Configure evolution
config = EvolutionConfig()
config.population_size = 150
config.prob_add_connection = 0.3
config.prob_add_node = 0.1

# Create population
pop = Population(pop_size=150, num_inputs=2, num_outputs=1, config=config)

# Define fitness function
def fitness_fn(genomes):
    for genome in genomes:
        model = NeatModule(genome)
        # ... evaluate model ...
        genome.fitness = score

# Evolve
for generation in range(100):
    pop.run_generation(fitness_fn)
    best = max(pop.genomes, key=lambda g: g.fitness)
    print(f"Gen {generation}: Best fitness = {best.fitness:.4f}")
```

## Examples

### XOR Problem
```bash
python examples/xor_simple.py
```

### LunarLander-v2
```bash
python examples/lunarlander_solve.py
```

### Function Approximation
```bash
python examples/function_approx.py
```

### Visualization Demo
```bash
python examples/visualization_demo.py
```

### Hyperparameter Tuning
```bash
python examples/hyperparam_search.py --method random --trials 50
```

## Hybrid Evolution + Learning

```python
from neatify import NeatModule, Checkpoint
import torch.optim as optim

# 1. Evolve topology
pop.run_generation(fitness_fn)
best_genome = max(pop.genomes, key=lambda g: g.fitness)

# 2. Fine-tune weights with gradient descent
model = NeatModule(best_genome, trainable=True)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# 3. Sync weights back to genome
model.update_genome_weights()

# 4. Continue evolution with improved weights
pop.genomes[0] = best_genome
pop.run_generation(fitness_fn)
```

## Checkpointing

```python
from neatify import Checkpoint

# Save population state
Checkpoint.save(pop, "checkpoint_gen100.pkl")

# Load and resume
pop = Checkpoint.load("checkpoint_gen100.pkl")
pop.run_generation(fitness_fn)

# Save best genome
best = max(pop.genomes, key=lambda g: g.fitness)
Checkpoint.save_best(best, "best_genome.pkl")

# Export to JSON (human-readable)
Checkpoint.export_genome_json(best, "best_genome.json")
```

## Visualization

```python
from neatify.visualization import (
    plot_species_distribution,
    visualize_genome,
    plot_complexity_evolution
)

# Track evolution history
history = []
for gen in range(100):
    pop.run_generation(fitness_fn)
    history.append(copy.deepcopy(pop))

# Generate plots
plot_species_distribution(history, "species.png")
visualize_genome(best_genome, "network.png")
plot_complexity_evolution(history, "complexity.png")
```

## Configuration

```python
from neatify import EvolutionConfig

config = EvolutionConfig()

# Population
config.population_size = 150
config.elitism_count = 3
config.target_species = 5

# Mutations
config.prob_mutate_weight = 0.8
config.prob_add_connection = 0.3
config.prob_add_node = 0.1
config.prob_mutate_activation = 0.1

# Weight mutations
config.prob_replace_weight = 0.1
config.weight_mutation_power = 0.5
config.weight_perturbation_type = 'gaussian'  # or 'uniform'
config.weight_min_value = -30.0
config.weight_max_value = 30.0

# Speciation
config.compatibility_threshold = 3.0
config.c1 = 1.0  # Excess genes coefficient
config.c2 = 1.0  # Disjoint genes coefficient
config.c3 = 0.4  # Weight difference coefficient
```

## Testing

Run all tests:
```bash
python tests/test_dynamic_threshold.py
python tests/test_elitism.py
python tests/test_weight_mutation.py
python tests/test_recurrent.py
python tests/test_activations_adapter.py
python tests/test_fine_tuning.py
python tests/test_weight_sync.py
python tests/test_checkpoint.py
```

## Architecture

```
neatify/
├── core.py              # Genome, NodeGene, ConnectionGene
├── evolution.py         # Mutation operators, crossover, config
├── population.py        # Population management, speciation
├── pytorch_adapter.py   # PyTorch integration
├── checkpoint.py        # Save/load utilities
└── visualization.py     # Plotting functions

examples/
├── xor_simple.py        # Basic XOR solver
├── lunarlander_solve.py # Continuous control
├── function_approx.py   # Regression tasks
├── visualization_demo.py
└── hyperparam_search.py

tests/
├── test_*.py            # Unit tests
└── benchmark_batch.py   # Performance benchmarks
```

## Performance

- **Sparse Mode**: 3-5x faster for large, sparse networks
- **Batch Processing**: Efficient GPU utilization
- **Checkpointing**: Minimal overhead (~1% of evolution time)

## Documentation

- [User Guide](GUIDE.md): Step-by-step tutorials, advanced usage, and best practices.
- [Walkthrough](walkthrough.md): Technical details and implementation history.
- [Examples](examples/): Practical demonstration scripts.

## Citation

If you use NEATify in your research, please cite:

```bibtex
@software{neatify2024,
  title = {NEATify: Modern NEAT Implementation with PyTorch},
  year = {2024},
  url = {https://github.com/yourusername/neatify}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
