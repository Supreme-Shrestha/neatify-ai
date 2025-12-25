# NEATify User Guide

Welcome to the comprehensive guide for **NEATify**. This document will take you from a complete beginner to an advanced user, capable of evolving complex neural architectures and integrating them with modern deep learning workflows.

---

## 🛠️ Installation

To get started, install the library via pip:

```bash
pip install neatify-ai
```

---

## 📑 Table of Contents
1. [What is NEATify?](#-what-is-neatify)
2. [Basic Concepts](#-basic-concepts)
3. [Quick Start: Solving XOR](#-quick-start-solving-xor)
4. [Mastering the EvolutionConfig](#-mastering-the-evolutionconfig)
5. [Advanced: Hybrid Evolution + PyTorch Learning](#-advanced-hybrid-evolution--pytorch-learning)
6. [Visualization & Debugging](#-visualization--debugging)
7. [Checkpointing & Persistence](#-checkpointing--persistence)
8. [Best Practices](#-best-practices)

---

## 🚀 What is NEATify?

**NEATify** is an implementation of **NEAT (NeuroEvolution of Augmenting Topologies)**, a popular algorithm for evolving neural network structures. Unlike traditional deep learning, where you define the network shape yourself, NEAT "invents" the architecture, adding nodes and connections over time as needed to solve a problem.

**Why use NEATify?**
- **Architecture Discovery**: Finds the smallest network that can solve your task.
- **No Backprop Needed**: Can solve problems where gradients are unavailable (e.g., reinforcement learning).
- **PyTorch Native**: Seamlessly convert evolved networks into PyTorch modules for ultra-fast inference and hybrid training.

---

## 🧠 Basic Concepts

Before diving into code, let's understand the core entities:

- **Genome**: A blueprint for a neural network. It contains a list of **Node Genes** (neurons) and **Connection Genes** (weights between neurons).
- **Innovation Number**: A historical marker that allows NEAT to safely cross-breed two different network structures without breaking them.
- **Speciation**: The population is divided into "species" based on structural similarity. This prevents new, innovative topologies from being killed off by older, more optimized ones before they have a chance to prove themselves.
- **Fitness**: A score assigned to a genome representing how well it performed a task. **Higher is always better.**

---

## ⚡ Quick Start: Solving XOR

Let's solve the classic XOR problem. We want a network that takes two inputs and outputs 1 if they are different, and 0 if they are the same.

### 1. Define your Fitness Function
The `fitness_fn` is where the magic happens. It takes a list of genomes and assigns a `fitness` value to each.

```python
import torch
from neatify import NeatModule

# X: Inputs, Y: Targets
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

def eval_fitness(genomes):
    for genome in genomes:
        # 1. Convert Genome to a PyTorch Module
        model = NeatModule(genome)
        
        # 2. Run Inference
        with torch.no_grad():
            predictions = model(X)
            
        # 3. Calculate Error (MSE)
        mse = torch.mean((predictions - Y)**2).item()
        
        # 4. Assign Fitness (Higher is better, so we use inverse MSE)
        genome.fitness = 1.0 / (1.0 + mse)
```

### 2. Set Up and Run Evolution

```python
from neatify import Population, EvolutionConfig

# Create configuration
config = EvolutionConfig()
config.population_size = 150

# Initialize population: 2 inputs, 1 output
pop = Population(pop_size=150, num_inputs=2, num_outputs=1, config=config)

# Run for 50 generations
for gen in range(50):
    pop.run_generation(eval_fitness)
    best = max(pop.genomes, key=lambda g: g.fitness)
    print(f"Gen {gen} | Best Fitness: {best.fitness:.4f}")
    
    if best.fitness > 0.95:
        print("Solved!")
        break
```

---

## ⚙️ Mastering the EvolutionConfig

The `EvolutionConfig` object controls every aspect of the evolutionary process. Tuning these parameters is key to solving hard problems.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 150 | Number of individuals in each generation. |
| `prob_add_connection` | 0.05 | Chance to add a new link between nodes. |
| `prob_add_node` | 0.03 | Chance to split a connection and insert a new node. |
| `prob_mutate_weight`| 0.80 | Chance to change the weights of existing connections. |
| `elitism_count` | 2 | Number of top individuals to preserve unchanged. |
| `target_species` | 5 | NEAT will adjust speciation thresholds to aim for this many species. |

**Example: Encouraging complex topologies**
```python
config.prob_add_connection = 0.2  # Add connections much more often
config.prob_add_node = 0.1        # Add nodes more often
```

---

## 🛠️ Advanced: Hybrid Evolution + PyTorch Learning

One of NEATify's strongest features is the ability to combine **Evolution** (finding topology) with **Backpropagation** (refining weights).

```python
import torch.optim as optim

# 1. Get the best genome from evolution
best_genome = max(pop.genomes, key=lambda g: g.fitness)

# 2. Convert to a TRAINABLE PyTorch module
model = NeatModule(best_genome, trainable=True)

# 3. Use standard PyTorch optimizers
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(X), Y)
    loss.backward()
    optimizer.step()

# 4. CRITICAL: Sync the trained weights back to the Genome
model.update_genome_weights()

# Now 'best_genome' has the optimized weights and can be saved or evolved further!
```

---

## 📊 Visualization & Debugging

Don't guess what your networks are doing—see them!

```python
from neatify.visualization import visualize_genome, plot_species_distribution

# 1. Visualize a specific network
visualize_genome(best_genome, "best_network.png", layout='layered')

# 2. Plot population dynamics (requires history of population objects)
# history = [pop_gen0, pop_gen1, ...]
plot_species_distribution(history, "species_growth.png")
```

---

## 💾 Checkpointing & Persistence

NEAT runs can take a long time. Use the `Checkpoint` system to save your progress.

```python
from neatify import Checkpoint

# Save the entire state (Population + Random states)
Checkpoint.save(pop, "my_run_gen10.pkl")

# Resume later
pop = Checkpoint.load("my_run_gen10.pkl")

# Export a single genome to human-readable JSON
Checkpoint.export_genome_json(best_genome, "champion.json")
```

---

## 💡 Best Practices

1. **Fitness Scaling**: Always ensure your fitness is positive and higher values are better. Use `1.0 / (1.0 + error)` for regression tasks.
2. **Batch Processing**: When evaluating genomes, pass all data points through the model at once (e.g., `model(full_dataset_tensor)`) rather than looping. NEATify is optimized for batch tensors.
3. **Speciation Tuning**: If you have too many species, the population gets fragmented and evolution slows down. If you have too few, you might lose innovation. Aim for `target_species` between 3 and 10.
4. **Hardware**: For large populations or complex networks, enable sparse mode: `NeatModule(genome, use_sparse=True)`.

---

Happy Evolving! 🧬🚀
