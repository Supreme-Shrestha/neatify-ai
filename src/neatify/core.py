"""
Core data structures for NEAT (NeuroEvolution of Augmenting Topologies).

This module defines the fundamental building blocks of a neural network in NEAT:
- NodeType: Enum for input, hidden, and output nodes.
- ActivationType: Enum for available activation functions.
- NodeGene: Representation of a single neuron.
- ConnectionGene: Representation of a synapse between neurons with innovation tracking.
- Genome: A collection of nodes and connections forming a complete network topology.
"""

import random
import copy
from enum import Enum

class NodeType(Enum):
    """Enumeration of possible node types in a NEAT genome."""
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2

class ActivationType(Enum):
    """Enumeration of supported activation functions for nodes."""
    SIGMOID = 0
    RELU = 1
    TANH = 2
    IDENTITY = 3
    LEAKY_RELU = 4
    ELU = 5

class NodeGene:
    """
    Representation of a single node (neuron) in the genome.
    
    Attributes:
        id (int): Unique identifier for the node.
        type (NodeType): The role of the node (input, hidden, or output).
        activation (ActivationType): The activation function applied at this node.
    """
    def __init__(self, id, type, activation=ActivationType.SIGMOID):
        """
        Initialize a new NodeGene.
        
        Args:
            id (int): Unique node ID.
            type (NodeType): Type of the node.
            activation (ActivationType, optional): Initial activation function. Defaults to SIGMOID.
        """
        self.id = id
        self.type = type
        self.activation = activation

    def __repr__(self):
        return f"NodeGene(id={self.id}, type={self.type}, act={self.activation.name})"

    def copy(self):
        """
        Create a deep copy of the node gene.
        
        Returns:
            NodeGene: A new NodeGene instance with the same properties.
        """
        return NodeGene(self.id, self.type, self.activation)

class ConnectionGene:
    """
    Representation of a connection (synapse) between two nodes.
    
    Attributes:
        in_node (int): ID of the source node.
        out_node (int): ID of the destination node.
        weight (float): Scalar weight of the connection.
        enabled (bool): Whether the connection is active during inference.
        innovation_number (int): Historical marking for tracking evolutionary lineage.
    """
    def __init__(self, in_node, out_node, weight, enabled, innovation_number):
        """
        Initialize a new ConnectionGene.
        
        Args:
            in_node (int): Source node ID.
            out_node (int): Destination node ID.
            weight (float): Initial weight value.
            enabled (bool): Initial enabled state.
            innovation_number (int): Global innovation ID.
        """
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

    def __repr__(self):
        return f"ConnectionGene(in={self.in_node}, out={self.out_node}, weight={self.weight:.3f}, enabled={self.enabled}, innov={self.innovation_number})"

    def copy(self):
        """
        Create a deep copy of the connection gene.
        
        Returns:
            ConnectionGene: A new ConnectionGene instance with the same properties.
        """
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled, self.innovation_number)

class Genome:
    """
    A collection of NodeGenes and ConnectionGenes representing a neural network topology.
    
    Attributes:
        id (int): Unique identifier for the genome.
        nodes (dict): Mapping of node IDs to NodeGene objects.
        connections (dict): Mapping of innovation numbers to ConnectionGene objects.
        fitness (float): Raw performance score assigned during evaluation.
        adjusted_fitness (float): Fitness score adjusted for speciation.
    """
    def __init__(self, id, num_inputs, num_outputs):
        """
        Initialize a new Genome with basic input and output nodes.
        
        Args:
            id (int): Unique genome ID.
            num_inputs (int): Number of input nodes to create.
            num_outputs (int): Number of output nodes to create.
        """
        self.id = id
        self.nodes = {} # id -> NodeGene
        self.connections = {} # innovation_number -> ConnectionGene
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        
        # Initialize inputs
        for i in range(num_inputs):
            self.nodes[i] = NodeGene(i, NodeType.INPUT, ActivationType.IDENTITY)
            
        # Initialize outputs
        for i in range(num_outputs):
            self.nodes[num_inputs + i] = NodeGene(num_inputs + i, NodeType.OUTPUT, ActivationType.SIGMOID)

    def add_node(self, node):
        """Add a NodeGene to the genome."""
        self.nodes[node.id] = node

    def add_connection(self, connection):
        """Add a ConnectionGene to the genome."""
        self.connections[connection.innovation_number] = connection

    def copy(self):
        """
        Create a deep copy of the entire genome.
        
        Returns:
            Genome: A new Genome instance with cloned nodes and connections.
        """
        new_genome = Genome(self.id, 0, 0) # Empty init
        new_genome.nodes = {k: v.copy() for k, v in self.nodes.items()}
        new_genome.connections = {k: v.copy() for k, v in self.connections.items()}
        new_genome.fitness = self.fitness
        return new_genome
        
    def __repr__(self):
        return f"Genome(id={self.id}, nodes={len(self.nodes)}, connections={len(self.connections)}, fitness={self.fitness})"
