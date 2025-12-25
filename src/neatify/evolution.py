"""
Evolutionary operators and tracking for NEAT.

This module implements the evolutionary mechanisms described in the original NEAT paper:
- Weight and activation mutation.
- Topology mutation (adding nodes and connections).
- Crossover (mating) between genomes.
- Compatibility distance calculation for speciation.
- Innovation tracking to ensure consistency of topological changes.
"""

import random
from .core import Genome, ConnectionGene, NodeGene, NodeType, ActivationType

class EvolutionConfig:
    """
    Configuration parameters for the evolutionary process.
    
    Attributes:
        prob_add_node (float): Probability of adding a new node by splitting a connection.
        prob_add_connection (float): Probability of adding a new connection between existing nodes.
        prob_mutate_weight (float): Probability of a single connection weight being mutated.
        weight_mutation_power (float): Standard deviation for Gaussian weight mutations.
        prob_toggle_link (float): Probability of toggling the enabled state of a connection.
        c1, c2, c3 (float): Coefficients for excess, disjoint, and weight difference in compatibility distance.
        prob_mutate_activation (float): Probability of a node's activation function being changed.
        allowed_activations (list): List of ActivationType values choices for mutation.
        compatibility_threshold (float): Initial distance threshold for speciation.
        target_species (int): Desired number of species to maintain via dynamic thresholding.
        compatibility_threshold_delta (float): Adjust step size for the compatibility threshold.
        elitism_count (int): Number of top genomes preserved unchanged in each generation.
        prob_replace_weight (float): Probability of a weight being completely replaced rather than perturbed.
        weight_min_value, weight_max_value (float): Clipping range for connection weights.
        weight_perturbation_type (str): Type of mutation ('gaussian' or 'uniform').
    """
    def __init__(self):
        self.prob_add_node = 0.03
        self.prob_add_connection = 0.05
        self.prob_mutate_weight = 0.8
        self.weight_mutation_power = 0.5
        self.prob_toggle_link = 0.01
        self.c1 = 1.0 # Excess
        self.c2 = 1.0 # Disjoint
        self.c3 = 0.4 # Weight difference
        self.prob_mutate_activation = 0.1
        self.allowed_activations = [ActivationType.SIGMOID, ActivationType.RELU, ActivationType.TANH, ActivationType.IDENTITY]
        self.compatibility_threshold = 3.0
        self.target_species = 5
        self.compatibility_threshold_delta = 0.3
        self.elitism_count = 2
        self.prob_replace_weight = 0.1
        self.weight_min_value = -30.0
        self.weight_max_value = 30.0
        self.weight_perturbation_type = 'gaussian' # 'gaussian' or 'uniform'

class InnovationTracker:
    """
    Global tracker for historical markings (innovation numbers) in NEAT.
    
    Used to ensure that identical topological mutations across different genomes
    receive the same innovation number, enabling alignment during crossover.
    """
    def __init__(self):
        self.current_innovation = 0
        self.global_innovations = {} # (in_node, out_node) -> innovation_number
        self.node_id_counter = 0

    def get_innovation(self, in_node, out_node):
        """
        Get or create an innovation number for a connection between two nodes.
        
        Args:
            in_node (int): Source node ID.
            out_node (int): Destination node ID.
            
        Returns:
            int: The innovation number for this connection.
        """
        if (in_node, out_node) in self.global_innovations:
            return self.global_innovations[(in_node, out_node)]
        else:
            self.current_innovation += 1
            self.global_innovations[(in_node, out_node)] = self.current_innovation
            return self.current_innovation

    def get_new_node_id(self):
        """Generates a unique node ID."""
        self.node_id_counter += 1
        return self.node_id_counter

    def set_initial_node_id(self, max_id):
        """Sets the starting point for node ID generation."""
        self.node_id_counter = max_id

def mutate_weight(genome, config):
    """
    Mutate connection weights in a genome based on configuration.
    
    Weights are either perturbed (added value) or replaced (new random value).
    
    Args:
        genome (Genome): The genome to mutate.
        config (EvolutionConfig): Configuration parameters.
    """
    for conn in genome.connections.values():
        if random.random() < config.prob_mutate_weight:
            if random.random() < config.prob_replace_weight:
                # New random weight
                conn.weight = random.uniform(-1, 1)
            else:
                # Perturb weight
                if config.weight_perturbation_type == 'uniform':
                    perturbation = random.uniform(-config.weight_mutation_power, config.weight_mutation_power)
                else:
                    perturbation = random.gauss(0, config.weight_mutation_power)
                    
                conn.weight += perturbation
                # Clamp weights
                conn.weight = max(config.weight_min_value, min(config.weight_max_value, conn.weight))

def mutate_activation(genome, config):
    """
    Mutate node activation functions in a genome.
    
    Args:
        genome (Genome): The genome to mutate.
        config (EvolutionConfig): Configuration parameters.
    """
    candidates = [n for n in genome.nodes.values() if n.type != NodeType.INPUT]
    if not candidates:
        return
        
    for node in candidates:
        if random.random() < config.prob_mutate_activation:
            node.activation = random.choice(config.allowed_activations)

def mutate_add_connection(genome, tracker, config, max_attempts=20):
    """
    Attempt to add a new connection between two randomly selected nodes.
    
    Args:
        genome (Genome): The genome to mutate.
        tracker (InnovationTracker): Global innovation tracker.
        config (EvolutionConfig): Configuration parameters.
        max_attempts (int, optional): Max retries to find a valid new link. Defaults to 20.
        
    Returns:
        bool: True if a connection was added, False otherwise.
    """
    # Try to find two nodes to connect
    nodes = list(genome.nodes.keys())
    for _ in range(max_attempts):
        in_node_id = random.choice(nodes)
        out_node_id = random.choice(nodes)
        
        if in_node_id == out_node_id:
            continue
            
        # Check if connection already exists
        exists = False
        for conn in genome.connections.values():
            if conn.in_node == in_node_id and conn.out_node == out_node_id:
                exists = True
                break
        
        if exists:
            continue
            
        innov = tracker.get_innovation(in_node_id, out_node_id)
        new_conn = ConnectionGene(in_node_id, out_node_id, random.uniform(-1, 1), True, innov)
        genome.add_connection(new_conn)
        return True
    return False

def mutate_add_node(genome, tracker, config):
    """
    Add a new node by splitting an existing enabled connection.
    
    The old connection is disabled, and two new connections are created:
    - One from source to new node with weight 1.0.
    - One from new node to destination with the original connection's weight.
    
    Args:
        genome (Genome): The genome to mutate.
        tracker (InnovationTracker): Global innovation tracker.
        config (EvolutionConfig): Configuration parameters.
        
    Returns:
        bool: True if a node was added, False otherwise.
    """
    if not genome.connections:
        return False
        
    # Pick a random enabled connection
    enabled_conns = [c for c in genome.connections.values() if c.enabled]
    if not enabled_conns:
        return False
        
    conn_to_split = random.choice(enabled_conns)
    conn_to_split.enabled = False
    
    new_node_id = tracker.get_new_node_id()
    # Ensure we don't collide with existing IDs (though tracker should handle this globally)
    while new_node_id in genome.nodes:
         new_node_id = tracker.get_new_node_id()

    new_node = NodeGene(new_node_id, NodeType.HIDDEN, random.choice(config.allowed_activations))
    genome.add_node(new_node)
    
    # Add two new connections
    # 1. In -> New (weight = 1.0)
    innov1 = tracker.get_innovation(conn_to_split.in_node, new_node_id)
    conn1 = ConnectionGene(conn_to_split.in_node, new_node_id, 1.0, True, innov1)
    
    # 2. New -> Out (weight = old weight)
    innov2 = tracker.get_innovation(new_node_id, conn_to_split.out_node)
    conn2 = ConnectionGene(new_node_id, conn_to_split.out_node, conn_to_split.weight, True, innov2)
    
    genome.add_connection(conn1)
    genome.add_connection(conn2)
    return True

def crossover(parent1, parent2, config):
    """
    Perform crossover (mating) between two parent genomes.
    
    The child inherits:
    - All matching genes (randomly picked from either parent).
    - Disjoint/excess genes from the fitter parent.
    
    Args:
        parent1 (Genome): First parent.
        parent2 (Genome): Second parent.
        config (EvolutionConfig): Configuration parameters.
        
    Returns:
        Genome: The resulting child genome.
    """
    # Parent1 should be the fitter one
    if parent2.fitness > parent1.fitness:
        parent1, parent2 = parent2, parent1
        
    child = Genome(parent1.id, 0, 0) # ID will be set by population
    
    # Inherit nodes from fitter parent
    for node_id, node in parent1.nodes.items():
        child.add_node(node.copy())
        
    # Match connections
    for innov, conn1 in parent1.connections.items():
        if innov in parent2.connections:
            # Matching gene
            conn2 = parent2.connections[innov]
            # Randomly pick weight
            child_conn = conn1.copy() if random.random() < 0.5 else conn2.copy()
            # If either disabled, 75% chance to disable in child
            if not conn1.enabled or not conn2.enabled:
                if random.random() < 0.75:
                    child_conn.enabled = False
            child.add_connection(child_conn)
        else:
            # Disjoint/Excess gene from fitter parent
            child.add_connection(conn1.copy())
            
    return child

def calculate_compatibility_distance(genome1, genome2, config):
    """
    Calculate the compatibility distance between two genomes.
    
    Distance = (c1 * E / N) + (c2 * D / N) + (c3 * W_diff)
    Where E=excess, D=disjoint, N=max genome size, W_diff=avg weight difference.
    
    Args:
        genome1 (Genome): First genome.
        genome2 (Genome): Second genome.
        config (EvolutionConfig): Configuration parameters.
        
    Returns:
        float: Compatibility distance.
    """
    conns1 = genome1.connections
    conns2 = genome2.connections
    
    all_innovs = set(conns1.keys()) | set(conns2.keys())
    max_innov1 = max(conns1.keys()) if conns1 else 0
    max_innov2 = max(conns2.keys()) if conns2 else 0
    
    matching = 0
    disjoint = 0
    excess = 0
    weight_diff_sum = 0
    
    for innov in all_innovs:
        in1 = innov in conns1
        in2 = innov in conns2
        
        if in1 and in2:
            matching += 1
            weight_diff_sum += abs(conns1[innov].weight - conns2[innov].weight)
        elif in1:
            if innov > max_innov2:
                excess += 1
            else:
                disjoint += 1
        elif in2:
            if innov > max_innov1:
                excess += 1
            else:
                disjoint += 1
                
    N = max(len(conns1), len(conns2))
    if N < 20: N = 1.0 # Normalize only for larger networks
    
    avg_weight_diff = weight_diff_sum / matching if matching > 0 else 0
    
    distance = (config.c1 * excess / N) + (config.c2 * disjoint / N) + (config.c3 * avg_weight_diff)
    return distance
