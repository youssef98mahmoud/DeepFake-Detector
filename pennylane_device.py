from deepfake_detection.devices.quantum_device import QuantumDevice
import torch

class PennyLaneDevice(QuantumDevice):
    def __init__(self, backend, num_qubits, num_layers, type_layers, diff_type):
        super().__init__(num_qubits, num_layers, type_layers)
        import pennylane as qml
        self.qml = qml
        self.diff_type = diff_type
        self.backend = backend

    def get_weight_shapes(self):
        if self.type_layers == "basic":
            return {"weights": (self.num_layers, self.num_qubits)}
        elif self.type_layers == "strong":
            return {"weights": (self.num_layers, self.num_qubits, 3)}

    def create_layer(self):
        qnode = self.qml.QNode(self.circuit, self.backend, diff_method=self.diff_type)
        return self.qml.qnn.TorchLayer(qnode, self.get_weight_shapes())

    def circuit(self, inputs, weights):
        self.qml.AngleEmbedding(inputs, wires=range(self.num_qubits))

        if self.type_layers == "basic":
            self.qml.BasicEntanglerLayers(weights, wires=range(self.num_qubits))
        elif self.type_layers == "strong":
            self.qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))

        return [self.qml.expval(self.qml.PauliZ(wires=i)) for i in range(self.num_qubits)]

    def create_gated_circuit(self):
        """Create gated quantum circuit for SNN-QNN (returns callable)"""
        def gated_circuit(params):
            """
            params:
                angles: (layer, qubit)
                ent_strengths: (layer, edge) or None if Q==1
                g_layer: (layer,) binary
                g_qubit: (layer, qubit) binary
                g_edge: (layer, edge) binary or None if Q==1
            """
            angles = params['angles']
            ent_strengths = params['ent_strengths']
            g_layer = params['g_layer']
            g_qubit = params['g_qubit']
            g_edge = params['g_edge']

            for l in range(self.num_layers):
                layer_gate = g_layer[l]

                for q in range(self.num_qubits):
                    theta = angles[l, q] * layer_gate * g_qubit[l, q]
                    self.qml.RY(theta, wires=q)

                if self.num_qubits > 1 and ent_strengths is not None and g_edge is not None:
                    for e in range(self.num_qubits):
                        q1 = e
                        q2 = (e + 1) % self.num_qubits
                        phi = ent_strengths[l, e] * layer_gate * g_edge[l, e]
                        self.qml.CRX(phi, wires=[q1, q2])

            return [self.qml.expval(self.qml.PauliZ(q)) for q in range(self.num_qubits)]

        qnode = self.qml.QNode(
            gated_circuit,
            self.backend,
            interface="torch",
            diff_method=self.diff_type
        )

        # Return callable that stacks results
        def circuit_wrapper(params):
            result = qnode(params)
            return torch.stack([
                val if torch.is_tensor(val) else torch.tensor(val, dtype=torch.float32)
                for val in result
            ])

        return circuit_wrapper