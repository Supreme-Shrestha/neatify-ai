"""
Management of Genome populations and speciation.

This module provides the Population class which handles the high-level
evolutionary loop, and the Species class for grouping similar genomes.
It coordinates speciation, reproduction, and fitness-based selection.
"""

import math
import random
from .core import Genome
from .evolution import (
    EvolutionConfig, 
    InnovationTracker, 
    mutate_add_node, 
    mutate_add_connection, 
    mutate_weight, 
    mutate_activation, 
    crossover, 
    calculate_compatibility_distance
)

class Species:
    """
    A collection of similar genomes sharing a common evolutionary lineage.
    
    Speciation protects innovation by allowing new structures to optimize
    within their own niche before competing with the entire population.
    
    Attributes:
        id (int): Unique identifier for the species.
        representative (Genome): A member used as a reference for distance calculations.
        members (list): List of all genomes currently belonging to this species.
        average_fitness (float): Mean fitness of all members.
        staleness (int): Number of generations without improvement in best fitness.
    """
    def __init__(self, id, representative):
        """
        Initialize a new Species.
        
        Args:
            id (int): Unique species ID.
            representative (Genome): Initial member to represent the species.
        """
        self.id = id
        self.representative = representative
        self.members = [representative]
        self.average_fitness = 0.0
        self.staleness = 0

    def add_member(self, genome):
        """Add a genome to the species."""
        self.members.append(genome)

    def reset(self):
        """
        Prepare species for a new generation by clearing members.
        A random current member is selected as the new representative.
        """
        # Pick a random new representative from current members before clearing
        if self.members:
            self.representative = random.choice(self.members)
        self.members = []
        self.average_fitness = 0.0

    def calculate_average_fitness(self):
        """Calculate the arithmetic mean fitness of all current members."""
        if not self.members:
            self.average_fitness = 0.0
            return
        total = sum(m.fitness for m in self.members)
        self.average_fitness = total / len(self.members)

class Population:
    """
    The main coordinator for the NEAT evolutionary process.
    
    Manages a collection of genomes, organizes them into species, and
    executes the generational loop of evaluation, selection, and reproduction.
    
    Attributes:
        pop_size (int): Total number of individuals in the population.
        config (EvolutionConfig): Parameters controlling the evolution.
        tracker (InnovationTracker): Global tracker for topological mutations.
        genomes (list): List of all current individuals (Genome objects).
        species (list): List of active Species objects.
        generation (int): Current generation counter.
    """
    def __init__(self, pop_size, num_inputs, num_outputs, config=None):
        """
        Initialize a new Population with a base topology.
        
        Args:
            pop_size (int): Number of genomes to create.
            num_inputs (int): Number of input nodes.
            num_outputs (int): Number of output nodes.
            config (EvolutionConfig, optional): Custom configuration. Defaults to None.
        """
        self.pop_size = pop_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config if config else EvolutionConfig()
        self.tracker = InnovationTracker()
        self.tracker.set_initial_node_id(num_inputs + num_outputs - 1)
        
        self.species = []
        self.genomes = []
        self.generation = 0
        self.species_id_counter = 0
        
        # Initialize population with minimal linear connections
        for i in range(pop_size):
            g = Genome(i, num_inputs, num_outputs)
            # Standard NEAT starts with no hidden nodes.
            # Let's add initial connections between all inputs and outputs
            for inp in range(num_inputs):
                for out in range(num_outputs):
                    out_node_idx = num_inputs + out
                    innov = self.tracker.get_innovation(inp, out_node_idx)
                    # Random weights
                    from .core import ConnectionGene
                    conn = ConnectionGene(inp, out_node_idx, random.uniform(-1, 1), True, innov)
                    g.add_connection(conn)
            self.genomes.append(g)
            
        self.speciate()

    def speciate(self):
        """
        Partition the current genome list into species based on compatibility distance.
        Also handles dynamic thresholding to maintain the target number of species.
        """
        # Clear species members but keep representatives
        for s in self.species:
            s.reset()
            
        for g in self.genomes:
            found = False
            for s in self.species:
                dist = calculate_compatibility_distance(g, s.representative, self.config)
                if dist < self.config.compatibility_threshold:
                    s.add_member(g)
                    found = True
                    break
            
            if not found:
                self.species_id_counter += 1
                new_species = Species(self.species_id_counter, g)
                self.species.append(new_species)
                
        # Remove empty species
        self.species = [s for s in self.species if s.members]
        
        # Adjust compatibility threshold to hit target species count
        if len(self.species) < self.config.target_species:
            self.config.compatibility_threshold = max(0.3, self.config.compatibility_threshold - self.config.compatibility_threshold_delta)
        elif len(self.species) > self.config.target_species:
            self.config.compatibility_threshold += self.config.compatibility_threshold_delta

    def run_generation(self, fitness_function):
        """
        Execute one complete generation of evolution.
        
        Steps:
        1. Evaluate all genomes using the provided fitness function.
        2. Update species metrics.
        3. Apply Reproduction.
        4. Re-speciate the new population.
        
        Args:
            fitness_function (callable): A function that accepts a list of Genomes and sets their .fitness attribute.
        """
        # 1. Evaluate fitness
        # fitness_function takes a list of genomes and assigns fitness to them
        fitness_function(self.genomes)
        
        # 2. Update species metrics
        total_avg_fitness = 0
        for s in self.species:
            s.calculate_average_fitness()
            total_avg_fitness += s.average_fitness
            
        # 3. Apply Reproduction
        new_genomes = []
        
        # Elitism: preserve best performers
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        elites = self.genomes[:self.config.elitism_count]
        for elite in elites:
            new_genomes.append(elite.copy())
        
        if total_avg_fitness == 0:
            # Guard against dead populations
            total_avg_fitness = 1.0 

        for s in self.species:
            # Species-wise proportionate reproduction
            # offspring_count = (s.average_fitness / total_avg_fitness) * self.pop_size
            # Simple integer partitioning
            offspring_count = int((s.average_fitness / total_avg_fitness) * self.pop_size)
            
            # Sort members by fitness
            s.members.sort(key=lambda g: g.fitness, reverse=True)
            
            # Tournament/Culling selection: only top half survive to mate
            survivors = s.members[:max(1, len(s.members)//2)]
            
            for _ in range(offspring_count):
                if len(new_genomes) >= self.pop_size:
                    break
                    
                parent1 = random.choice(survivors)
                # Interspecies mating chance
                if random.random() < 0.001: 
                    # Pick random species
                    other_species = random.choice(self.species)
                    if other_species.members:
                        parent2 = random.choice(other_species.members)
                    else:
                        parent2 = random.choice(survivors)
                else:
                    parent2 = random.choice(survivors)
                    
                child = crossover(parent1, parent2, self.config)
                
                # Topological Mutations
                if random.random() < self.config.prob_add_node:
                    mutate_add_node(child, self.tracker, self.config)
                if random.random() < self.config.prob_add_connection:
                    mutate_add_connection(child, self.tracker, self.config)
                # Parameter Mutations
                mutate_weight(child, self.config)
                mutate_activation(child, self.config)
                
                new_genomes.append(child)
                
        # Fill rounding gaps
        while len(new_genomes) < self.pop_size:
            # Pick random species
            if not self.species: # Should not happen
                break
            s = random.choice(self.species)
            if not s.members: continue
            parent1 = random.choice(s.members)
            child = parent1.copy()
            mutate_weight(child, self.config) # Just mutate
            mutate_activation(child, self.config)
            new_genomes.append(child)
            
        self.genomes = new_genomes
        self.speciate()
        self.generation += 1
