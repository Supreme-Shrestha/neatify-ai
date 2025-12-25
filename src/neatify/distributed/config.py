"""
Configuration for distributed NEAT computing.
"""

class DistributedConfig:
    """
    Configuration parameters for distributed NEAT evolution.
    """
    
    def __init__(
        self,
        host='0.0.0.0',
        port=5000,
        heartbeat_interval=5.0,
        heartbeat_timeout=2.0,
        task_timeout=300.0,
        connection_timeout=10.0,
        max_retries=3,
        batch_size_per_worker=0,
        min_workers=1,
        enable_fault_tolerance=True
    ):
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.task_timeout = task_timeout
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self.batch_size_per_worker = batch_size_per_worker
        self.min_workers = min_workers
        self.enable_fault_tolerance = enable_fault_tolerance
        
    def __repr__(self):
        return (
            f"DistributedConfig(host={self.host}, port={self.port}, "
            f"min_workers={self.min_workers}, fault_tolerance={self.enable_fault_tolerance})"
        )
