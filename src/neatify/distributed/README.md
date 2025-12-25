# Distributed NEAT Computing

This directory contains the distributed computing implementation for NEATify, enabling parallel genome evaluation across multiple worker nodes.

## Architecture

The distributed system uses a **Master-Worker architecture**:

- **Master Node**: Coordinates evolution (crossover, mutation, speciation) and distributes genome batches to workers
- **Worker Nodes**: Evaluate genome fitness in parallel and report results back to master
- **Communication**: TCP/IP with pickle serialization for efficient data transfer

## Components

### Core Modules

- **`config.py`**: Configuration parameters for distributed system
- **`protocol.py`**: Message types, data structures, and serialization utilities
- **`master.py`**: Master node implementation (`DistributedPopulation`, `SystemCoordinator`)
- **`worker.py`**: Worker node implementation (`WorkerNode`, `GenomeEvaluator`)

### Key Classes

- **`DistributedPopulation`**: Extends `Population` to distribute fitness evaluation
- **`WorkerNode`**: Worker process that evaluates genomes
- **`SystemCoordinator`**: Manages worker connections and task distribution
- **`DistributedConfig`**: Configuration for network, timeouts, and capacity

## Usage

### Master Node Example

```python
from neatify.distributed import DistributedPopulation, DistributedConfig

# Configure distributed system
config = DistributedConfig(
    host='0.0.0.0',
    port=5000,
    min_workers=2
)

# Create distributed population
pop = DistributedPopulation(
    pop_size=150,
    num_inputs=2,
    num_outputs=1,
    distributed_config=config
)

# Run evolution (fitness evaluation happens on workers)
for gen in range(50):
    pop.run_generation(lambda genomes: None)  # Dummy function
    best = max(pop.genomes, key=lambda g: g.fitness)
    print(f"Gen {gen}: Best Fitness = {best.fitness:.4f}")

pop.shutdown()
```

### Worker Node Example

```python
from neatify.distributed import WorkerNode
from neatify import NeatModule
import torch

def xor_fitness(genomes):
    """Fitness function executed on worker."""
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    
    for genome in genomes:
        model = NeatModule(genome)
        outputs = model(inputs)
        loss = torch.sum((outputs - targets) ** 2).item()
        genome.fitness = 4.0 - loss

# Create and start worker
worker = WorkerNode(
    master_host='192.168.1.100',
    master_port=5000,
    worker_id=1,
    fitness_function=xor_fitness,
    capacity=50
)

worker.start()  # Blocks until shutdown
```

## Running the Examples

### Terminal 1 - Start Master

```bash
cd /home/jegree/Documents/NEATify
python examples/distributed_xor_master.py --host 0.0.0.0 --port 5000 --workers 2
```

### Terminal 2 - Start Worker 1

```bash
cd /home/jegree/Documents/NEATify
python examples/distributed_xor_worker.py --master-host localhost --master-port 5000 --worker-id 1
```

### Terminal 3 - Start Worker 2

```bash
cd /home/jegree/Documents/NEATify
python examples/distributed_xor_worker.py --master-host localhost --master-port 5000 --worker-id 2
```

## Configuration Options

### DistributedConfig Parameters

- **`host`**: Master node host address (default: '0.0.0.0')
- **`port`**: Master node port (default: 5000)
- **`min_workers`**: Minimum workers required to start (default: 1)
- **`heartbeat_interval`**: Seconds between heartbeat checks (default: 5.0)
- **`task_timeout`**: Maximum seconds for task completion (default: 300.0)
- **`batch_size_per_worker`**: Genomes per batch, 0=auto (default: 0)
- **`enable_fault_tolerance`**: Enable task redistribution on failure (default: True)

## Design Principles

1. **Backward Compatibility**: Existing `Population` class unchanged, distributed is opt-in
2. **Minimal Dependencies**: Uses standard library (socket, pickle, threading)
3. **LAN Optimized**: Designed for low-latency local network communication
4. **Fault Tolerance**: Handles worker failures gracefully (future enhancement)

## Future Enhancements

- Dynamic worker discovery and auto-scaling
- Advanced distribution strategies (weighted, species-based)
- Checkpoint/resume for long-running experiments
- GPU acceleration support
- Multi-master fault tolerance

## Protocol Details

### Message Types

- `WORKER_REGISTRATION`: Worker announces itself to master
- `TASK_ASSIGNMENT`: Master sends genome batch to worker
- `FITNESS_REPORT`: Worker returns fitness results
- `HEARTBEAT_REQUEST/RESPONSE`: Health monitoring
- `SHUTDOWN_SIGNAL`: Graceful termination

### Message Format

```
[4 bytes length][1 byte message_type][N bytes pickled data]
```

## Performance Considerations

- **Serialization Overhead**: Pickle is fast but adds ~10-20% overhead
- **Network Latency**: LAN typically <1ms, minimal impact
- **Batch Size**: Larger batches reduce communication overhead
- **Scalability**: Linear speedup up to ~10 workers for typical problems

## Troubleshooting

**Workers not connecting:**
- Check firewall settings
- Verify master host/port are correct
- Ensure master is started before workers

**Slow performance:**
- Increase batch size per worker
- Check network bandwidth
- Verify workers have sufficient CPU/memory

**Connection timeouts:**
- Increase `task_timeout` in config
- Check network stability
- Reduce population size or batch size
