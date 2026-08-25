from deepfake_detection.devices.quantum_device import QuantumDevice
import torch
import numpy as np

class PyQuestDevice(QuantumDevice):
    """pyQuEST v4 quantum device wrapper

    https://github.com/rrmeister/pyQuEST/tree/feature/quest-v4
    https://pyquest.org/api.html
    """

    def __init__(self, num_qubits, num_layers, type_layers, use_gpu=False):
        super().__init__(num_qubits, num_layers, type_layers)
        self.use_gpu = use_gpu

        import pyquest
        self.pyquest = pyquest

        print(f"pyQuEST device created (GPU: {use_gpu}, OpenMP: {pyquest.openmp})")

    def get_weight_shapes(self):
        if self.type_layers == "basic":
            return (self.num_layers, self.num_qubits)
        elif self.type_layers == "strong":
            return (self.num_layers, self.num_qubits, 3)

    def create_layer(self):
        """Create a pyQuEST-based PyTorch layer"""

        class PyQuestLayer(torch.nn.Module):
            """PyTorch-compatible layer using pyQuEST backend"""

            def __init__(layer_self, device_obj):
                super().__init__()
                layer_self.device_obj = device_obj
                layer_self.num_qubits = device_obj.num_qubits
                layer_self.num_layers = device_obj.num_layers
                layer_self.type_layers = device_obj.type_layers
                layer_self.pyquest = device_obj.pyquest

                # Create quantum register
                layer_self.qureg = layer_self.pyquest.Register(layer_self.num_qubits)

                # Initialize weights
                weight_shape = device_obj.get_weight_shapes()
                layer_self.weights = torch.nn.Parameter(torch.rand(weight_shape) * 2 * np.pi)

            def _create_rx_gate(layer_self, qubit, angle):
                """Create RX rotation gate matrix"""
                c = np.cos(angle / 2)
                s = np.sin(angle / 2)
                matrix = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
                return layer_self.pyquest.operators.MatrixOperator([qubit], matrix)

            def _create_ry_gate(layer_self, qubit, angle):
                """Create RY rotation gate matrix"""
                c = np.cos(angle / 2)
                s = np.sin(angle / 2)
                matrix = np.array([[c, -s], [s, c]], dtype=complex)
                return layer_self.pyquest.operators.MatrixOperator([qubit], matrix)

            def _create_rz_gate(layer_self, qubit, angle):
                """Create RZ rotation gate matrix"""
                matrix = np.array([[np.exp(-1j * angle / 2), 0], 
                                   [0, np.exp(1j * angle / 2)]], dtype=complex)
                return layer_self.pyquest.operators.MatrixOperator([qubit], matrix)

            def _create_cnot_gate(layer_self, control, target):
                """Create CNOT gate matrix"""
                cnot = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]], dtype=complex)
                return layer_self.pyquest.operators.MatrixOperator([control, target], cnot)

            def _create_pauli_z(layer_self, qubit):
                """Create PauliZ observable matrix"""
                pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
                return layer_self.pyquest.operators.MatrixOperator([qubit], pauli_z)

            def forward(layer_self, inputs):
                """Forward pass through quantum circuit"""
                batch_size = inputs.shape[0]
                outputs = []

                for i in range(batch_size):
                    # Initialize to |0> state
                    layer_self.qureg.init_blank_state()

                    # Angle embedding - apply RY rotations for input encoding
                    input_data = inputs[i].detach().cpu().numpy()
                    for j in range(min(len(input_data), layer_self.num_qubits)):
                        ry_gate = layer_self._create_ry_gate(j, float(input_data[j]))
                        layer_self.qureg.apply_operator(ry_gate)

                    # Apply parametrized layers
                    weights_np = layer_self.weights.detach().cpu().numpy()

                    if layer_self.type_layers == "basic":
                        for layer in range(layer_self.num_layers):
                            # Single qubit rotations
                            for q in range(layer_self.num_qubits):
                                ry_gate = layer_self._create_ry_gate(q, float(weights_np[layer, q]))
                                layer_self.qureg.apply_operator(ry_gate)

                            # CNOT ladder for entanglement
                            for q in range(layer_self.num_qubits - 1):
                                cnot_gate = layer_self._create_cnot_gate(q, q + 1)
                                layer_self.qureg.apply_operator(cnot_gate)

                            # Wrap-around CNOT
                            if layer_self.num_qubits > 1:
                                cnot_gate = layer_self._create_cnot_gate(layer_self.num_qubits - 1, 0)
                                layer_self.qureg.apply_operator(cnot_gate)

                    elif layer_self.type_layers == "strong":
                        for layer in range(layer_self.num_layers):
                            # Three rotations per qubit
                            for q in range(layer_self.num_qubits):
                                rx_gate = layer_self._create_rx_gate(q, float(weights_np[layer, q, 0]))
                                layer_self.qureg.apply_operator(rx_gate)

                                ry_gate = layer_self._create_ry_gate(q, float(weights_np[layer, q, 1]))
                                layer_self.qureg.apply_operator(ry_gate)

                                rz_gate = layer_self._create_rz_gate(q, float(weights_np[layer, q, 2]))
                                layer_self.qureg.apply_operator(rz_gate)

                            # All-to-all entanglement
                            for q in range(layer_self.num_qubits):
                                for r in range(q + 1, layer_self.num_qubits):
                                    cnot_gate = layer_self._create_cnot_gate(q, r)
                                    layer_self.qureg.apply_operator(cnot_gate)

                    # Measure PauliZ expectation values
                    expectations = []
                    for q in range(layer_self.num_qubits):
                        pauli_z = layer_self._create_pauli_z(q)
                        expectation = layer_self.qureg.apply_operator(pauli_z)
                        expectations.append(float(expectation))

                    outputs.append(np.array(expectations, dtype=np.float32))

                return torch.tensor(np.array(outputs), dtype=torch.float32, device=inputs.device, requires_grad=True)

            def __del__(layer_self):
                """Cleanup quantum register"""
                try:
                    if hasattr(layer_self, 'qureg') and layer_self.qureg.is_alive:
                        layer_self.qureg.destroy_reg()
                except:
                    pass

        return PyQuestLayer(self)
