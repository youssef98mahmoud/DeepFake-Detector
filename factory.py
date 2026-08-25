def create_model(config, backbone_path=None, weights_path=None, device='cpu'):
    '''Model creation factory'''
    model_type = config["modelType"]

    if model_type == "Classic":
        from deepfake_detection.models import classical
        model = classical.ClassicalModel(backbone_path)
    else:
        quantum_device = create_quantum_device(config)

        if model_type == "Hybrid":
            from deepfake_detection.models import hybrid
            model = hybrid.HQNN(quantum_device, backbone_path)
        elif model_type == "Spiking":
            from deepfake_detection.models import snn_hybrid_feedback_gated
            model = snn_hybrid_feedback_gated.SNNQNN(quantum_device, backbone_path=backbone_path)
        elif model_type == "Quantum":
            from deepfake_detection.models import quantum
            model = quantum.QNN(quantum_device)

        else:
            raise ValueError(f"Unknown modelType: {model_type}")

    if weights_path:
        model = _load_weights(model, weights_path, device)

    return model

def _load_weights(model, weights_path, device='cpu'):
    """
    Load weights with automatic PennyLane to Qiskit conversion if needed.
    """
    import torch
    import logging

    saved_state_dict = torch.load(weights_path, map_location=device)
    model_state = model.state_dict()

    model_is_qiskit = any(
        '.weight' in k and 'quan_layers' in k and not k.endswith('.weights')
        for k in model_state.keys()
    )
    saved_is_pennylane = any(
        '.weights' in k and 'quan_layers' in k
        for k in saved_state_dict.keys()
    )

    if model_is_qiskit and saved_is_pennylane:
        logging.info("Converting weights from PennyLane to Qiskit format...")
        converted = {}
        for key, value in saved_state_dict.items():
            if '.weights' in key and 'quan_layers' in key:
                base = key.replace('.weights', '')
                flat_value = value.flatten()
                converted[f"{base}.weight"] = flat_value
                converted[f"{base}._weights"] = flat_value
            else:
                converted[key] = value
        saved_state_dict = converted

    model.load_state_dict(saved_state_dict)
    logging.info(f"Weights loaded from {weights_path}")
    return model

def create_quantum_device(config):
    device_type = config["quantumDevice"]

    # PennyLane devices
    if device_type in ["default.qubit", "lightning.gpu"]:
        import pennylane as qml
        from deepfake_detection.devices import pennylane_device
        backend = qml.device(device_type, wires=config["nQubits"])

        return pennylane_device.PennyLaneDevice(
            backend,
            config["nQubits"],
            config["nLayers"],
            config["typeLayers"],
            config["diffType"]
        )

    # pyQuEST devices
    elif device_type in ["pyquest.cpu", "pyquest.gpu"]:
        from deepfake_detection.devices import pyquest_device
        use_gpu = (device_type == "pyquest.gpu")

        return pyquest_device.PyQuestDevice(
            config["nQubits"],
            config["nLayers"],
            config["typeLayers"],
            use_gpu
        )

    # Qiskit devices
    elif device_type.startswith("qiskit"):
        from deepfake_detection.devices import qiskit_device

        # Qiskit Aer local simulator (no backend needed)
        if device_type == "qiskit.aer":
            return qiskit_device.QiskitAerDevice(
                config["nQubits"],
                config["nLayers"],
                config["typeLayers"]
            )

        # For transpile and hardware, connect to IBM backend
        import os
        from qiskit_ibm_runtime import QiskitRuntimeService
        from dotenv import load_dotenv

        load_dotenv(".ibm_api.env")
        API_KEY = os.getenv("IBM_QUANTUM_API_KEY")

        if not API_KEY:
            raise ValueError("IBM_QUANTUM_API_KEY environment variable not set")

        backend_name = device_type.split(":")[1]

        try:
            service = QiskitRuntimeService(token=API_KEY)
            backend = service.backend(backend_name)
        except Exception as e:
            raise ValueError(
                f"Could not load backend '{backend_name}'. Error: {e}\n"
                f"Make sure you're authenticated with IBM Quantum."
            )

        # Qiskit Transpiler
        if device_type.startswith("qiskit.transpile"):
            optimisation_level = config.get("optimisationLevel", 1)
            return qiskit_device.QiskitTranspiler(
                backend,
                config["nQubits"],
                config["nLayers"],
                config["typeLayers"],
                optimisation_level
            )

        # Qiskit Hardware
        elif device_type.startswith("qiskit.hardware"):
            shots = config.get("shots", 1024)
            optimisation_level = config.get("optimisationLevel", 3)
            return qiskit_device.QiskitHardwareDevice(
                backend,
                config["nQubits"],
                config["nLayers"],
                config["typeLayers"],
                shots,
                optimisation_level
            )
    raise ValueError(
        f"Unknown quantumDevice: {device_type}\n"
        f"Supported: default.qubit, lightning.gpu, pyquest.cpu, pyquest.gpu, "
        f"qiskit.transpile:backend, qiskit.hardware:backend"
    )
