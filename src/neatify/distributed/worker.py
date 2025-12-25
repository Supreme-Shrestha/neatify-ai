"""
Worker node implementation for distributed NEAT.
"""

import socket
import threading
import time
import logging
from typing import Callable, List

from .protocol import (
    MessageType, WorkerState, GenomePackage, BatchContainer, 
    FitnessResult, WorkerStatus, send_message, receive_message,
    serialize_genome, deserialize_genome
)

logging.basicConfig(level=logging.INFO)

class WorkerNode:
    def __init__(self, master_host: str, master_port: int, worker_id: int, 
                 fitness_function: Callable, capacity: int = 50):
        self.master_host = master_host
        self.master_port = master_port
        self.worker_id = worker_id
        self.capacity = capacity
        self.fitness_function = fitness_function
        self.socket = None
        self.running = False
        self.logger = logging.getLogger(f'Worker-{worker_id}')
        
    def start(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.master_host, self.master_port))
            self._register_with_master()
            self.running = True
            self._main_loop()
        except Exception as e:
            self.logger.error(f"Worker error: {e}")
        finally:
            self._shutdown()
    
    def _register_with_master(self):
        status = WorkerStatus(worker_id=self.worker_id, worker_address="", capacity=self.capacity)
        send_message(self.socket, MessageType.WORKER_REGISTRATION, status)
        msg_type, data = receive_message(self.socket)
        if msg_type != MessageType.REGISTRATION_ACK:
            raise RuntimeError(f"Unexpected response: {msg_type}")
    
    def _main_loop(self):
        while self.running:
            try:
                self.socket.settimeout(1.0)
                try:
                    msg_type, data = receive_message(self.socket)
                except socket.timeout:
                    continue
                
                if msg_type == MessageType.TASK_ASSIGNMENT:
                    logging.debug(f"Worker {self.worker_id} received task assignment")
                    self._handle_task_assignment(data)
                elif msg_type == MessageType.HEARTBEAT_REQUEST:
                    self._handle_heartbeat()
                elif msg_type == MessageType.SHUTDOWN_SIGNAL:
                    logging.info("Worker received shutdown signal")
                    self.running = False
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                import traceback
                traceback.print_exc()
                self.running = False
    
    def _handle_task_assignment(self, batch: BatchContainer):
        logging.debug(f"Processing batch {batch.batch_id} with {len(batch.genomes)} genomes")
        evaluator = GenomeEvaluator()
        results = evaluator.evaluate_batch(batch.genomes, self.fitness_function)
        data = {'batch_id': batch.batch_id, 'results': results}
        logging.debug(f"Sending fitness report for batch {batch.batch_id}")
        send_message(self.socket, MessageType.FITNESS_REPORT, data)
    
    def _handle_heartbeat(self):
        status = WorkerStatus(worker_id=self.worker_id, worker_address="", capacity=self.capacity)
        send_message(self.socket, MessageType.HEARTBEAT_RESPONSE, status)
    
    def _shutdown(self):
        self.running = False
        if self.socket: self.socket.close()

class GenomeEvaluator:
    def evaluate_batch(self, genome_packages: List[GenomePackage], 
                      fitness_function: Callable) -> List[FitnessResult]:
        results = []
        genomes = []
        for pkg in genome_packages:
            try:
                genome = deserialize_genome(pkg.serialized_genome)
                genomes.append(genome)
            except:
                results.append(FitnessResult(pkg.genome_id, 0.0, evaluation_successful=False))
        
        try:
            fitness_function(genomes)
            for genome in genomes:
                results.append(FitnessResult(genome.id, genome.fitness))
        except:
            for genome in genomes:
                results.append(FitnessResult(genome.id, 0.0, evaluation_successful=False))
        return results
