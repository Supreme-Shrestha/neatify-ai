"""
PyTorch Adapter for NEAT Genomes.

This module provides the NeatModule class, which converts a NEAT genome
into a standard PyTorch nn.Module. It supports:
- Efficient vectorized forward passes.
- Sparse matrix operations for large networks.
- Recurrent network handling with state propagation.
- Gradient-based optimization of genome weights.
"""

import torch
import torch.nn as nn
import networkx as nx
from .core import NodeType, ActivationType

class NeatModule(nn.Module):
    """
    Standard PyTorch wrapper for a NEAT Genome.
    
    This class allows a NEAT network to be used as a standard PyTorch module,
    enabling integration with PyTorch optimizers, loss functions, and datasets.
    
    Attributes:
        genome (Genome): The source NEAT genome.
        use_sparse (bool): Whether to use sparse matrix operations for the forward pass.
        trainable (bool): Whether connection weights are registered as trainable parameters.
        weight_values (nn.Parameter): Vector of all enabled connection weights.
        is_recurrent (bool): Whether the neural network contains cycles.
    """
    def __init__(self, genome, use_sparse=False, trainable=True):
        """
        Initialize the NeatModule.
        
        Args:
            genome (Genome): The genome to convert.
            use_sparse (bool, optional): Use sparse ops for performance. Defaults to False.
            trainable (bool, optional): Allow weight training via Autograd. Defaults to True.
        """
        super(NeatModule, self).__init__()
        self.genome = genome
        self.use_sparse = use_sparse
        self.trainable = trainable
        self.node_eval_order = []
        self.connections_map = {} # out_node -> [(in_node, weight_index)]
        
        # Consistent ordering of connections
        self.connection_list = []
        for conn in self.genome.connections.values():
            if conn.enabled:
                self.connection_list.append(conn)
        
        # Register weights as an nn.Parameter for gradient flow
        initial_weights = torch.tensor([c.weight for c in self.connection_list], dtype=torch.float32)
        self.weight_values = nn.Parameter(initial_weights)
        if not self.trainable:
            self.weight_values.requires_grad = False
            
        if self.use_sparse:
            self._build_sparse_graph()
        else:
            self._build_graph()
        
    def _build_graph(self):
        """Build the internal graph representation for standard dense forward pass."""
        G = nx.DiGraph()
        
        inputs = []
        outputs = []
        hidden = []
        
        for node_id, node in self.genome.nodes.items():
            G.add_node(node_id)
            if node.type == NodeType.INPUT:
                inputs.append(node_id)
            elif node.type == NodeType.OUTPUT:
                outputs.append(node_id)
            else:
                hidden.append(node_id)
                
        for i, conn in enumerate(self.connection_list):
             G.add_edge(conn.in_node, conn.out_node)
             if conn.out_node not in self.connections_map:
                 self.connections_map[conn.out_node] = []
             self.connections_map[conn.out_node].append((conn.in_node, i))

        # Determine evaluation order
        try:
            self.node_eval_order = list(nx.topological_sort(G))
            self.is_recurrent = False
        except nx.NetworkXUnfeasible:
            self.is_recurrent = True
            self.node_eval_order = sorted(list(self.genome.nodes.keys()), key=lambda n: (self.genome.nodes[n].type.value, n))

        # Inputs are fed directly, so filter them out of computation order
        self.node_eval_order = [n for n in self.node_eval_order if self.genome.nodes[n].type != NodeType.INPUT]
        
        self.input_nodes = sorted(inputs)
        self.output_nodes = sorted(outputs)
        self.node_states = {}

    def _build_sparse_graph(self):
        """Prepare indices and masks for vectorized sparse matrix forward pass."""
        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.genome.nodes.keys())}
        self.idx_to_node = {i: node_id for node_id, i in self.node_to_idx.items()}
        self.num_nodes = len(self.genome.nodes)
        
        indices = []
        for i, conn in enumerate(self.connection_list):
            in_idx = self.node_to_idx[conn.in_node]
            out_idx = self.node_to_idx[conn.out_node]
            indices.append([in_idx, out_idx])
                
        if indices:
            self.weight_indices = torch.LongTensor(indices).t()
        else:
            self.weight_indices = torch.LongTensor(torch.empty(2, 0))
            
        self.input_indices = [self.node_to_idx[n] for n in self.genome.nodes if self.genome.nodes[n].type == NodeType.INPUT]
        self.output_indices = [self.node_to_idx[n] for n in self.genome.nodes if self.genome.nodes[n].type == NodeType.OUTPUT]
        
        G = nx.DiGraph()
        for conn in self.genome.connections.values():
            if conn.enabled:
                G.add_edge(conn.in_node, conn.out_node)
                
        try:
            layers = list(nx.topological_generations(G))
            self.is_recurrent = False
            self.layers = [[self.node_to_idx[n] for n in layer] for layer in layers]
            self.layers = self.layers[1:] # Skip inputs
        except nx.NetworkXUnfeasible:
            self.is_recurrent = True
            self.layers = []
            
        self.activation_masks = {}
        for act_type in ActivationType:
            indices = [self.node_to_idx[n] for n in self.genome.nodes if self.genome.nodes[n].activation == act_type]
            if indices:
                self.activation_masks[act_type] = torch.tensor(indices, dtype=torch.long)
        
    def reset(self):
        """Reset internal node states (useful for recurrent networks)."""
        self.node_states = {}

    def forward(self, x, steps=1):
        """
        Execute forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_inputs).
            steps (int, optional): Number of propagation steps for recurrent networks. Defaults to 1.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, num_outputs).
        """
        batch_size = x.size(0)
        
        if self.use_sparse:
            return self._forward_sparse(x, batch_size, steps)
            
        if not self.node_states or list(self.node_states.values())[0].size(0) != batch_size:
             self.node_states = {n: torch.zeros(batch_size, device=x.device) for n in self.genome.nodes}
        else:
            for n in self.node_states:
                self.node_states[n] = self.node_states[n].detach()
        
        if not self.is_recurrent and steps == 1:
            # Optimized Feedforward
            activations = {}
            for i, node_id in enumerate(self.input_nodes):
                activations[node_id] = x[:, i]
                
            for node_id in self.node_eval_order:
                total_input = torch.zeros(batch_size, device=x.device)
                if node_id in self.connections_map:
                    for in_node, weight_idx in self.connections_map[node_id]:
                        if in_node in activations:
                            weight = self.weight_values[weight_idx]
                            total_input += activations[in_node] * weight
                
                node = self.genome.nodes[node_id]
                val = self._apply_activation(node, total_input)
                activations[node_id] = val
                
            output_tensors = [activations[n] for n in self.output_nodes]
            return torch.stack(output_tensors, dim=1)
            
        else:
            # Recurrent/Stateful
            current_outputs = None
            for _ in range(steps):
                for i, node_id in enumerate(self.input_nodes):
                    self.node_states[node_id] = x[:, i]
                
                new_states = {}
                for node_id in self.node_eval_order:
                    total_input = torch.zeros(batch_size, device=x.device)
                    if node_id in self.connections_map:
                        for in_node, weight_idx in self.connections_map[node_id]:
                            val = new_states[in_node] if in_node in new_states else self.node_states.get(in_node, 0.0)
                            weight = self.weight_values[weight_idx]
                            total_input += val * weight
                    
                    node = self.genome.nodes[node_id]
                    new_states[node_id] = self._apply_activation(node, total_input)
                
                for node_id, val in new_states.items():
                    self.node_states[node_id] = val
                    
                output_tensors = [self.node_states[n] for n in self.output_nodes]
                current_outputs = torch.stack(output_tensors, dim=1)
                
            return current_outputs

    def _forward_sparse(self, x, batch_size, steps):
        """Optimized sparse matrix implementation of the forward pass."""
        state = torch.zeros(batch_size, self.num_nodes, device=x.device)
        for i, idx in enumerate(self.input_indices):
            state[:, idx] = x[:, i]
            
        if not self.is_recurrent and steps == 1:
            for layer_indices in self.layers:
                if self.weight_indices.size(1) > 0:
                    weight_matrix = torch.sparse_coo_tensor(self.weight_indices, self.weight_values, (self.num_nodes, self.num_nodes))
                    total_input = torch.sparse.mm(weight_matrix.t(), state.t()).t()
                else:
                    total_input = torch.zeros(batch_size, self.num_nodes, device=x.device)
                
                new_state = torch.zeros_like(state)
                for act_type, mask in self.activation_masks.items():
                    val = self._vectorized_activation(act_type, total_input)
                    new_state.index_copy_(1, mask, val.index_select(1, mask))
                    
                layer_tensor = torch.tensor(layer_indices, device=x.device, dtype=torch.long)
                state = state.clone()
                state.index_copy_(1, layer_tensor, new_state.index_select(1, layer_tensor))
                
            output_indices = torch.tensor(self.output_indices, device=x.device, dtype=torch.long)
            return state.index_select(1, output_indices)
        else:
            # Recurrent Sparse (Jacobi approximation)
            current_outputs = None
            for _ in range(steps):
                state = state.clone()
                for i, idx in enumerate(self.input_indices):
                    state[:, idx] = x[:, i]
                    
                if self.weight_indices.size(1) > 0:
                    weight_matrix = torch.sparse_coo_tensor(self.weight_indices, self.weight_values, (self.num_nodes, self.num_nodes))
                    total_input = torch.sparse.mm(weight_matrix.t(), state.t()).t()
                else:
                    total_input = torch.zeros(batch_size, self.num_nodes, device=x.device)
                
                new_state = torch.zeros_like(state)
                for act_type, mask in self.activation_masks.items():
                    val = self._vectorized_activation(act_type, total_input)
                    new_state.index_copy_(1, mask, val.index_select(1, mask))
                state = new_state
                
                output_indices = torch.tensor(self.output_indices, device=x.device, dtype=torch.long)
                out = state.index_select(1, output_indices)
                
                if current_outputs is None:
                    current_outputs = out.unsqueeze(1)
                else:
                    current_outputs = torch.cat([current_outputs, out.unsqueeze(1)], dim=1)
                    
            return current_outputs.squeeze(1) if steps == 1 else current_outputs

    def _vectorized_activation(self, act_type, tensor):
        """Apply activation function to a tensor in a vectorized manner."""
        if act_type == ActivationType.SIGMOID: return torch.sigmoid(tensor)
        if act_type == ActivationType.RELU: return torch.relu(tensor)
        if act_type == ActivationType.TANH: return torch.tanh(tensor)
        if act_type == ActivationType.IDENTITY: return tensor
        if act_type == ActivationType.LEAKY_RELU: return torch.nn.functional.leaky_relu(tensor)
        if act_type == ActivationType.ELU: return torch.nn.functional.elu(tensor)
        return torch.sigmoid(tensor)

    def _apply_activation(self, node, total_input):
        """Apply activation function for a single node."""
        return self._vectorized_activation(node.activation, total_input)
    
    def update_genome_weights(self):
        """
        Synchronize trained weights from PyTorch back to the genome.
        
        This enables hybrid workflows where weights are fine-tuned using
        gradient descent and then returned to the evolutionary population.
        """
        for i, conn in enumerate(self.connection_list):
            conn.weight = self.weight_values[i].item()

