"""
Quick test to verify distributed module imports and basic functionality.
"""

import sys
import os
# Ensure we import from the local codebase
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all distributed components can be imported."""
    print("Testing imports...")
    try:
        from neatify.distributed import (
            DistributedConfig,
            DistributedPopulation,
            WorkerNode,
            SystemCoordinator
        )
        print("✓ All distributed components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test DistributedConfig creation."""
    print("\nTesting DistributedConfig...")
    try:
        from neatify.distributed import DistributedConfig
        config = DistributedConfig(host='localhost', port=5000, min_workers=1)
        assert config.host == 'localhost'
        assert config.port == 5000
        print(f"✓ DistributedConfig created: {config}")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_protocol():
    """Test protocol message types and data structures."""
    print("\nTesting protocol...")
    try:
        from neatify.distributed.protocol import (
            MessageType, WorkerState, GenomePackage,
            BatchContainer, FitnessResult, WorkerStatus
        )
        assert MessageType.WORKER_REGISTRATION.value == 1
        status = WorkerStatus(worker_id=1, worker_address='localhost:5000', capacity=50)
        assert status.worker_id == 1
        print("✓ Protocol components working correctly")
        return True
    except Exception as e:
        print(f"✗ Protocol test failed: {e}")
        return False

def test_genome_serialization():
    """Test genome serialization/deserialization."""
    print("\nTesting genome serialization...")
    try:
        from neatify import Genome
        from neatify.distributed.protocol import serialize_genome, deserialize_genome
        genome = Genome(id=1, num_inputs=2, num_outputs=1)
        genome.fitness = 2.5
        serialized = serialize_genome(genome)
        deserialized = deserialize_genome(serialized)
        assert deserialized.id == genome.id
        assert deserialized.fitness == genome.fitness
        print(f"✓ Genome serialization working (size: {len(serialized)} bytes)")
        return True
    except Exception as e:
        print(f"✗ Serialization test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Distributed NEAT Module - Quick Tests")
    print("=" * 60)
    results = [test_imports(), test_config(), test_protocol(), test_genome_serialization()]
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
