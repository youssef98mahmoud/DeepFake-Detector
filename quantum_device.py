from abc import ABC, abstractmethod

class QuantumDevice(ABC):
    def __init__(self, num_qubits, num_layers, type_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.type_layers = type_layers

    @abstractmethod
    def create_layer(self):
        """Create a quantum layer compatible with PyTorch"""
        pass

    @abstractmethod
    def get_weight_shapes(self):
        """Return weight shapes for the quantum circuit"""
        pass