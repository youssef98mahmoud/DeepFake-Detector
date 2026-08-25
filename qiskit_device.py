from deepfake_detection.devices.quantum_device import QuantumDevice
import numpy as np
import torch

class QiskitDevice(QuantumDevice):
    """Base class for Qiskit devices with shared circuit building logic"""
    def get_weight_shapes(self):
        if self.type_layers == "basic":
            return (self.num_layers, self.num_qubits)
        elif self.type_layers == "strong":
            return (self.num_layers, self.num_qubits, 3)

    def _build_circuit(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter

        input_params = [Parameter(f'in_{i}') for i in range(self.num_qubits)]
        weight_shape = self.get_weight_shapes()
        num_weight_params = int(np.prod(weight_shape))
        weight_params = [Parameter(f'w_{i}') for i in range(num_weight_params)]

        qc = QuantumCircuit(self.num_qubits)

        for i, param in enumerate(input_params):
            qc.ry(param, i)

        weight_idx = 0
        if self.type_layers == "basic":
            for layer in range(self.num_layers):
                for i in range(self.num_qubits):
                    qc.ry(weight_params[weight_idx], i)
                    weight_idx += 1
                for i in range(self.num_qubits - 1):
                    qc.cx(i, i + 1)
                if self.num_qubits > 1:
                    qc.cx(self.num_qubits - 1, 0)

        elif self.type_layers == "strong":
            for layer in range(self.num_layers):
                for i in range(self.num_qubits):
                    qc.rx(weight_params[weight_idx], i)
                    qc.ry(weight_params[weight_idx + 1], i)
                    qc.rz(weight_params[weight_idx + 2], i)
                    weight_idx += 3
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        qc.cx(i, j)

        return qc, input_params, weight_params

    def _create_observables(self):
        from qiskit.quantum_info import SparsePauliOp
        observables = []
        for i in range(self.num_qubits):
            pauli_str = 'I' * i + 'Z' + 'I' * (self.num_qubits - i - 1)
            observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
        return observables

    def _get_estimator_and_gradient(self):
        try:
            from qiskit.primitives import StatevectorEstimator as Estimator
        except ImportError:
            from qiskit.primitives import Estimator

        from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

        estimator = Estimator()
        gradient = ParamShiftEstimatorGradient(estimator)
        return estimator, gradient

    def create_gated_circuit(self):
        """Create gated quantum circuit for SNN-QNN (returns callable)"""
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        from qiskit_machine_learning.neural_networks import EstimatorQNN

        # Calculate parameter counts
        num_rotation_params = self.num_layers * self.num_qubits
        num_edges = self.num_qubits if self.num_qubits > 1 else 0
        num_entanglement_params = self.num_layers * num_edges if num_edges > 0 else 0

        # Create Qiskit parameters (required for parametrized circuits)
        rotation_params = [Parameter(f'rotation_{i}') for i in range(num_rotation_params)]
        entanglement_params = [Parameter(f'entangle_{i}') for i in range(num_entanglement_params)] if num_entanglement_params > 0 else []

        # Build the quantum circuit structure
        circuit = QuantumCircuit(self.num_qubits)
        rotation_idx = 0
        entanglement_idx = 0

        for l in range(self.num_layers):
            for qubit in range(self.num_qubits):
                circuit.ry(rotation_params[rotation_idx], qubit)
                rotation_idx += 1

            if num_edges > 0:
                for edge in range(self.num_qubits):
                    control_qubit = edge
                    target_qubit = (edge + 1) % self.num_qubits
                    circuit.crx(entanglement_params[entanglement_idx], control_qubit, target_qubit)
                    entanglement_idx += 1

        observables = self._create_observables()
        estimator, gradient = self._get_estimator_and_gradient()

        qnn = EstimatorQNN(
            circuit=circuit,
            input_params=[],
            weight_params=rotation_params + entanglement_params,
            observables=observables,
            estimator=estimator,
            gradient=gradient
        )

        def execute_with_gating(params):
            """
            Apply SNN-controlled gating to parameters before execution.

            The gating happens here (not in circuit) because Qiskit doesn't
            support dynamic circuit modification.
            """
            angles = params['angles']
            ent_strengths = params['ent_strengths']
            g_layer = params['g_layer']
            g_qubit = params['g_qubit']
            g_edge = params['g_edge']

            gated_params = []

            for l in range(self.num_layers):
                for q in range(self.num_qubits):
                    gated_angle = float(angles[l, q] * g_layer[l] * g_qubit[l, q])
                    gated_params.append(gated_angle)

            if num_edges > 0 and ent_strengths is not None and g_edge is not None:
                for l in range(self.num_layers):
                    for e in range(num_edges):
                        gated_entangle = float(ent_strengths[l, e] * g_layer[l] * g_edge[l, e])
                        gated_params.append(gated_entangle)

            # Execute QNN and return tensor (QNN returns numpy array)
            result = qnn.forward([], np.array(gated_params))
            result_tensor = torch.tensor(result, dtype=torch.float32)

            # QNN returns shape (1, num_qubits) - squeeze to (num_qubits,) for consistency with PennyLane
            if result_tensor.dim() > 1:
                result_tensor = result_tensor.squeeze(0)

            return result_tensor

        return execute_with_gating


class QiskitAerDevice(QiskitDevice):
    """Qiskit Aer simulator - local simulation with parameter-shift gradients

    WARNING: Slow due to parameter-shift requiring 2N circuit evaluations.
    Use for validation/testing only.
    """
    def __init__(self, num_qubits, num_layers, type_layers):
        super().__init__(num_qubits, num_layers, type_layers)

    def create_layer(self):
        """Create a Qiskit-based PyTorch layer with parameter-shift gradients"""
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit_machine_learning.neural_networks import EstimatorQNN

        qc, input_params, weight_params = self._build_circuit()
        observables = self._create_observables()
        estimator, gradient = self._get_estimator_and_gradient()

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            observables=observables,
            estimator=estimator,
            gradient=gradient
        )

        num_weight_params = len(weight_params)
        initial_weights = torch.rand(num_weight_params) * 2 * np.pi

        return TorchConnector(qnn, initial_weights=initial_weights)


class QiskitTranspiler(QiskitDevice):
    """Qiskit transpiler - validates circuit compatibility WITHOUT execution"""
    def __init__(self, backend, num_qubits, num_layers, type_layers, optimisation_level=1):
        super().__init__(num_qubits, num_layers, type_layers)
        self.backend = backend
        self.optimisation_level = optimisation_level
        self._layer_count = 0
        self._transpile_and_report()

    def _transpile_and_report(self):
        import logging
        from qiskit import transpile

        # Build circuit once just for the report
        qc, _, _ = self._build_circuit()

        backend_name = self.backend.name if hasattr(self.backend, 'name') else 'target backend'
        logging.info(f"TRANSPILATION REPORT: {backend_name}")

        transpiled_qc = transpile(
            qc,
            backend=self.backend,
            optimization_level=self.optimisation_level
        )

        logging.info(f"  Circuit compatible with {backend_name}")
        logging.info(f"  Original gates:   {dict(qc.count_ops())}")
        logging.info(f"  Transpiled gates: {dict(transpiled_qc.count_ops())}")
        logging.info(f"  Circuit depth:    {transpiled_qc.depth()}")
        logging.info(f"  Optimisation:     level {self.optimisation_level}")

        if hasattr(self.backend, 'num_qubits'):
            logging.info(f"  Backend qubits:   {self.backend.num_qubits}")
            logging.info(f"  Basis gates:      {list(self.backend.operation_names)}")
        elif hasattr(self.backend, 'configuration'):
            config = self.backend.configuration()
            logging.info(f"  Backend qubits:   {config.n_qubits}")
            logging.info(f"  Basis gates:      {config.basis_gates}")

        logging.info(f"Inference runs locally on StatevectorEstimator - no IBM quota used")
        logging.info(f"Ready for hardware: use qiskit.hardware:{backend_name}")

    def create_layer(self):
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit_machine_learning.neural_networks import EstimatorQNN

        self._layer_count += 1

        qc, input_params, weight_params = self._build_circuit()
        observables = self._create_observables()
        estimator, gradient = self._get_estimator_and_gradient()

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            observables=observables,
            estimator=estimator,
            gradient=gradient
        )

        num_weight_params = len(weight_params)
        initial_weights = torch.rand(num_weight_params) * 2 * np.pi

        return TorchConnector(qnn, initial_weights=initial_weights)


class QiskitHardwareDevice(QiskitDevice):
    """Qiskit hardware device - runs on REAL quantum hardware"""

    def __init__(self, backend, num_qubits, num_layers, type_layers, shots=1024, optimisation_level=3):
        super().__init__(num_qubits, num_layers, type_layers)
        self.backend = backend
        self.shots = shots
        self.optimisation_level = optimisation_level
        self.jobs_submitted = 0

    def create_layer(self):
        import logging
        from qiskit import transpile
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_ibm_runtime import EstimatorV2

        qc, input_params, weight_params = self._build_circuit()

        logging.info(f"Transpiling circuit for {self.backend.name}...")
        transpiled_qc = transpile(qc, backend=self.backend, optimization_level=self.optimisation_level)
        logging.info(f"Circuit transpiled (depth: {transpiled_qc.depth()}, gates: {transpiled_qc.count_ops()})\n")

        observables = self._create_observables()
        estimator = EstimatorV2(mode=self.backend)

        qnn = EstimatorQNN(
            circuit=transpiled_qc,
            input_params=input_params,
            weight_params=weight_params,
            observables=observables,
            estimator=estimator
        )

        original_forward = qnn.forward
        def tracked_forward(*args, **kwargs):
            self.jobs_submitted += 1
            if self.jobs_submitted % 10 == 0:
                logging.info(f"Jobs submitted: {self.jobs_submitted}")
            return original_forward(*args, **kwargs)
        qnn.forward = tracked_forward

        num_weight_params = len(weight_params)
        initial_weights = torch.rand(num_weight_params) * 2 * np.pi

        return TorchConnector(qnn, initial_weights=initial_weights)