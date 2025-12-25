"""
Distributed NEAT Computing Module

This module provides distributed computing capabilities for NEAT evolution
using a Master-Worker architecture optimized for LAN environments.
"""

from .config import DistributedConfig
from .master import DistributedPopulation, SystemCoordinator
from .worker import WorkerNode

__all__ = [
    'DistributedConfig',
    'DistributedPopulation',
    'SystemCoordinator',
    'WorkerNode',
]
