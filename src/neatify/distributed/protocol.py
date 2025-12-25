"""
Communication protocol for distributed NEAT.
"""

import pickle
import struct
from enum import Enum
from typing import List, Dict, Any, Tuple
import time

class MessageType(Enum):
    WORKER_REGISTRATION = 1
    REGISTRATION_ACK = 2
    TASK_ASSIGNMENT = 3
    FITNESS_REPORT = 4
    HEARTBEAT_REQUEST = 5
    HEARTBEAT_RESPONSE = 6
    SHUTDOWN_SIGNAL = 7
    ERROR_REPORT = 8
    STATUS_UPDATE = 9

class WorkerState(Enum):
    IDLE = 0
    BUSY = 1
    UNRESPONSIVE = 2
    FAILED = 3

class GenomePackage:
    def __init__(self, genome_id, serialized_genome, species_id=0, generation=0):
        self.genome_id = genome_id
        self.serialized_genome = serialized_genome
        self.species_id = species_id
        self.generation = generation

class BatchContainer:
    def __init__(self, batch_id, worker_id, genomes, fitness_function_name):
        self.batch_id = batch_id
        self.worker_id = worker_id
        self.genomes = genomes
        self.fitness_function_name = fitness_function_name
        self.dispatch_time = time.time()

class FitnessResult:
    def __init__(self, genome_id, fitness_score, evaluation_duration_ms=0, 
                 evaluation_successful=True, error_message=""):
        self.genome_id = genome_id
        self.fitness_score = fitness_score
        self.evaluation_duration_ms = evaluation_duration_ms
        self.evaluation_successful = evaluation_successful
        self.error_message = error_message

class WorkerStatus:
    def __init__(self, worker_id, worker_address, capacity=50):
        self.worker_id = worker_id
        self.worker_address = worker_address
        self.state = WorkerState.IDLE
        self.active_evaluations = 0
        self.completed_batches = 0
        self.last_heartbeat = time.time()
        self.capacity = capacity

def serialize_genome(genome):
    return pickle.dumps(genome)

def deserialize_genome(data):
    return pickle.loads(data)

def send_message(sock, message_type: MessageType, data: Any):
    serialized_data = pickle.dumps(data)
    message_type_byte = struct.pack('B', message_type.value)
    message = message_type_byte + serialized_data
    length = len(message)
    length_prefix = struct.pack('!I', length)
    sock.sendall(length_prefix + message)

def receive_message(sock) -> Tuple[MessageType, Any]:
    length_data = _recv_exact(sock, 4)
    if not length_data:
        raise ConnectionError("Connection closed")
    length = struct.unpack('!I', length_data)[0]
    message = _recv_exact(sock, length)
    if not message or len(message) != length:
        raise ConnectionError("Incomplete message received")
    message_type_value = struct.unpack('B', message[0:1])[0]
    message_type = MessageType(message_type_value)
    data = pickle.loads(message[1:])
    return message_type, data

def _recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data
