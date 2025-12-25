"""
Visualization tools for NEAT evolution and genome analysis.

This module provides plotting functions to help users analyze:
- Species distribution and fitness trends over generations.
- Neural network topologies of individual genomes.
- Evolutionary complexity (nodes and connections growth).
- Activation function diversity within a genome.

Dependencies:
    - matplotlib: For all 2D plotting.
    - networkx: For graph layout and rendering.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Optional
from .core import Genome, NodeType, ActivationType
from .population import Population

def plot_species_distribution(population_history: List[Population], save_path: Optional[str] = None):
    """
    Plot the count of species and fitness trends over multiple generations.
    
    Creates a two-panel figure:
    1. Upper panel: Number of active species per generation.
    2. Lower panel: Best and average fitness per generation.
    
    Args:
        population_history (List[Population]): Collection of population snapshots.
        save_path (str, optional): File path to save the image. If None, calls plt.show().
    """
    generations = [pop.generation for pop in population_history]
    species_counts = [len(pop.species) for pop in population_history]
    
    # Get fitness datasets
    best_fitness = [max(g.fitness for g in pop.genomes) if pop.genomes else 0 for pop in population_history]
    avg_fitness = [sum(g.fitness for g in pop.genomes) / len(pop.genomes) if pop.genomes else 0 for pop in population_history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Diversity Subplot
    ax1.plot(generations, species_counts, 'b-', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Number of Species', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Species Diversity Over Generations')
    
    # Performance Subplot
    ax2.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness')
    ax2.plot(generations, avg_fitness, 'orange', linewidth=2, label='Average Fitness')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Fitness Trends')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

def visualize_genome(genome: Genome, save_path: Optional[str] = None, show_weights: bool = True, layout: str = 'layered'):
    """
    Render a genome's neural network topology.
    
    Supports two layout strategies:
    - 'layered': Inputs on the left, outputs on the right, hidden nodes in between.
    - 'spring': Force-directed placement using NetworkX spring layout.
    
    Args:
        genome (Genome): The genome to visualize.
        save_path (str, optional): File path to save the image (.png, .jpg, .pdf).
        show_weights (bool, optional): If True, renders numeric weight values on edges.
        layout (str, optional): 'layered' or 'spring'. Defaults to 'layered'.
    """
    G = nx.DiGraph()
    
    node_colors = []
    node_labels = {}
    
    inputs = [n_id for n_id, n in genome.nodes.items() if n.type == NodeType.INPUT]
    outputs = [n_id for n_id, n in genome.nodes.items() if n.type == NodeType.OUTPUT]
    hidden = [n_id for n_id, n in genome.nodes.items() if n.type == NodeType.HIDDEN]
    
    for node_id, node in genome.nodes.items():
        G.add_node(node_id)
        # Label with ID and truncated activation name
        node_labels[node_id] = f"{node_id}\n{node.activation.name[:3]}"
        
        if node.type == NodeType.INPUT:
            node_colors.append('lightblue')
        elif node.type == NodeType.OUTPUT:
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightcoral')
    
    edge_labels = {}
    for conn in genome.connections.values():
        if conn.enabled:
            G.add_edge(conn.in_node, conn.out_node, weight=conn.weight)
            if show_weights:
                edge_labels[(conn.in_node, conn.out_node)] = f"{conn.weight:.2f}"
    
    # Coordinate calculation
    if layout == 'layered':
        pos = {}
        # Left column
        for i, n_id in enumerate(sorted(inputs)):
            pos[n_id] = (0, -i)
        # Right column
        for i, n_id in enumerate(sorted(outputs)):
            pos[n_id] = (1, -i)
        # Initial middle vertical stripe
        for i, n_id in enumerate(sorted(hidden)):
            pos[n_id] = (0.5, -i)
            
        # Refine hidden placement with spring logic while keeping I/O fixed
        fixed_nodes = inputs + outputs
        try:
            pos = nx.spring_layout(G, pos=pos, fixed=fixed_nodes, k=1.0, iterations=50)
        except:
            pass # Keep hardcoded layers if spring fails
    else:
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            pos = nx.shell_layout(G)
    
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, width=2)
    
    if show_weights:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
    
    plt.title(f"Genome {genome.id} - Fitness: {genome.fitness:.3f}\n"
              f"Nodes: {len(genome.nodes)}, Connections: {len([c for c in genome.connections.values() if c.enabled])}")
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def draw_genome(genome: Genome, filename: Optional[str] = None):
    """
    Alias for visualize_genome for backward compatibility.
    
    Defaults to the 'layered' layout as found in original implementation.
    """
    return visualize_genome(genome, save_path=filename, layout='layered')

def plot_complexity_evolution(population_history: List[Population], save_path: Optional[str] = None):
    """
    Track growth of network structures (nodes and edges) over time.
    
    Args:
        population_history (List[Population]): Multi-generation history.
        save_path (str, optional): Target image path.
    """
    generations = [pop.generation for pop in population_history]
    
    avg_nodes = []
    avg_connections = []
    max_nodes = []
    max_connections = []
    
    for pop in population_history:
        nodes_counts = [len(g.nodes) for g in pop.genomes]
        conn_counts = [len([c for c in g.connections.values() if c.enabled]) for g in pop.genomes]
        
        avg_nodes.append(sum(nodes_counts) / len(nodes_counts) if nodes_counts else 0)
        avg_connections.append(sum(conn_counts) / len(conn_counts) if conn_counts else 0)
        max_nodes.append(max(nodes_counts) if nodes_counts else 0)
        max_connections.append(max(conn_counts) if conn_counts else 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Nodes Growth
    ax1.plot(generations, avg_nodes, 'b-', linewidth=2, label='Average')
    ax1.plot(generations, max_nodes, 'b--', linewidth=1, label='Maximum')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Number of Nodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Node Count Evolution')
    
    # Connections Growth
    ax2.plot(generations, avg_connections, 'g-', linewidth=2, label='Average')
    ax2.plot(generations, max_connections, 'g--', linewidth=1, label='Maximum')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Number of Connections')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Connection Count Evolution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

def plot_activation_distribution(genome: Genome, save_path: Optional[str] = None):
    """
    Analyze and plot usage frequency of different activation functions in a genome.
    
    Args:
        genome (Genome): The genome to analyze.
        save_path (str, optional): Target image path.
    """
    activation_counts = {}
    for node in genome.nodes.values():
        if node.type != NodeType.INPUT:  # Inputs are passthrough (fixed)
            act_name = node.activation.name
            activation_counts[act_name] = activation_counts.get(act_name, 0) + 1
    
    if not activation_counts:
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(activation_counts.keys(), activation_counts.values(), color='skyblue', edgecolor='black')
    plt.xlabel('Activation Function')
    plt.ylabel('Count')
    plt.title(f'Activation Function Distribution - Genome {genome.id}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
