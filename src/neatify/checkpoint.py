"""
Checkpointing and Persistence for NEATify.

This module provides a robust system for saving and loading the state
of an evolutionary run, including population demographics, innovation
tracking, and individual genomes. It supports both binary pickle
format for full restoration and JSON for human-readable export.
"""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any
from .population import Population
from .evolution import EvolutionConfig
from .core import Genome, NodeGene, ConnectionGene, NodeType, ActivationType

class Checkpoint:
    """
    Utility class for population and genome persistence.
    
    Provides static methods to serialize and deserialize evolutionary state,
    enabling resumption of training from previous states and deployment
    of evolved models.
    """
    
    @staticmethod
    def save(population: Population, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save the entire population state to a binary file.
        
        Args:
            population (Population): The population object to save.
            filepath (str): Destination path (typically .pkl).
            metadata (dict, optional): Arbitrary dictionary of experiment metadata.
        """
        checkpoint_data = {
            'generation': population.generation,
            'pop_size': population.pop_size,
            'num_inputs': population.num_inputs,
            'num_outputs': population.num_outputs,
            'config': population.config,
            'genomes': population.genomes,
            'species': population.species,
            'species_id_counter': population.species_id_counter,
            'tracker': population.tracker,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    @staticmethod
    def load(filepath: str) -> Population:
        """
        Load a population and restore the evolutionary state.
        
        Args:
            filepath (str): Path to the saved checkpoint file.
            
        Returns:
            Population: A fully restored population object ready to continue evolution.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Manually reconstruct to bypass default __init__ logic
        pop = Population.__new__(Population)
        pop.generation = data['generation']
        pop.pop_size = data['pop_size']
        pop.num_inputs = data['num_inputs']
        pop.num_outputs = data['num_outputs']
        pop.config = data['config']
        pop.genomes = data['genomes']
        pop.species = data['species']
        pop.species_id_counter = data['species_id_counter']
        pop.tracker = data['tracker']
        
        return pop
    
    @staticmethod
    def save_best(genome: Genome, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save a single genome (usually the champion) to a file.
        
        Args:
            genome (Genome): The genome to save.
            filepath (str): Destination path.
            metadata (dict, optional): Arbitrary metadata (e.g. fitness value).
        """
        data = {
            'genome': genome,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_best(filepath: str) -> Genome:
        """
        Load a single genome from a binary file.
        
        Args:
            filepath (str): Path to the saved genome file.
            
        Returns:
            Genome: The restored genome object.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data['genome']
    
    @staticmethod
    def export_genome_json(genome: Genome, filepath: str):
        """
        Export a genome's topology and parameters to a human-readable JSON.
        
        Useful for auditing, manual editing, or cross-language deployment.
        
        Args:
            genome (Genome): The genome to export.
            filepath (str): Destination JSON path.
        """
        num_inputs = sum(1 for node in genome.nodes.values() if node.type == NodeType.INPUT)
        num_outputs = sum(1 for node in genome.nodes.values() if node.type == NodeType.OUTPUT)
        
        data = {
            'genome_id': genome.id,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'fitness': genome.fitness,
            'nodes': [
                {
                    'id': node_id,
                    'type': node.type.name,
                    'activation': node.activation.name
                }
                for node_id, node in genome.nodes.items()
            ],
            'connections': [
                {
                    'in_node': conn.in_node,
                    'out_node': conn.out_node,
                    'weight': conn.weight,
                    'enabled': conn.enabled,
                    'innovation': conn.innovation_number
                }
                for conn in genome.connections.values()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def import_genome_json(filepath: str) -> Genome:
        """
        Import a genome from a JSON file.
        
        Args:
            filepath (str): Path to the source JSON file.
            
        Returns:
            Genome: Reconstructed Genome object.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        genome = Genome(
            id=data['genome_id'],
            num_inputs=data['num_inputs'],
            num_outputs=data['num_outputs']
        )
        genome.fitness = data.get('fitness', 0.0)
        
        for node_data in data['nodes']:
            node_id = node_data['id']
            # Reconstruct hidden nodes that aren't part of base topology
            if node_id >= data['num_inputs'] + data['num_outputs']:
                node = NodeGene(
                    id=node_id,
                    type=NodeType[node_data['type']],
                    activation=ActivationType[node_data['activation']]
                )
                genome.nodes[node_id] = node
        
        for conn_data in data['connections']:
            conn = ConnectionGene(
                in_node=conn_data['in_node'],
                out_node=conn_data['out_node'],
                weight=conn_data['weight'],
                enabled=conn_data['enabled'],
                innovation_number=conn_data['innovation']
            )
            genome.add_connection(conn)
        
        return genome
