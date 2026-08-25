"""
SNN + Quantum closed-loop controller with sparse parameter gating.

Combines:
  Option A (true feedback loop):
    - Run the quantum circuit for T control steps inside one forward pass.
    - Feed quantum readout q_t back into the SNN/controller state for the next step.

  Option B (sparse parameter gating):
    - SNN produces binary gates that sparsify computation:
        * gate layers (activate/deactivate whole circuit blocks)
        * gate qubits (turn per-qubit rotations on/off)
        * gate edges (switch entanglement topology on/off)

Design notes:
- PennyLane QNodes generally aren't automatically batched, so we evaluate per-sample.
- We avoid tensor-dependent Python branching *inside the QNode* by using gates as
  multiplicative masks on angles/entangling strengths:
    RY(0) = Identity, CRX(0) = Identity.

Training tips:
- You can add a sparsity regularizer via `model.gate_sparsity_loss` (computed per forward call).
- You can also regularize temporal smoothness of gates/angles across steps if desired.
"""
from __future__ import annotations

import torch
import torchvision as torchvis
import numpy as np


class StraightThroughGate(torch.autograd.Function):
    """Straight-through binary gate (STE)

    Forward: hard threshold -> {0,1}
    Backward: gradient of sigmoid(alpha*x) (STE-like, smooth proxy)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(ctx.alpha * x)
        return grad_output * (s * (1 - s) * ctx.alpha), None


def bin_gate(logits: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
    return StraightThroughGate.apply(logits, alpha)


class SurrogateSpike(torch.autograd.Function):
    """Surrogate spike (for LIF)

    Forward: hard threshold
    Backward: piecewise-linear surrogate gradient in a band around 0
    """
    @staticmethod
    def forward(ctx, x, gamma=0.3):
        ctx.save_for_backward(x)
        ctx.gamma = gamma
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        gamma = ctx.gamma
        grad = (x.abs() < 1.0).to(x.dtype) * gamma
        return grad_output * grad, None


def spike_fn(x, gamma=0.3):
    return SurrogateSpike.apply(x, gamma)


class LIFControllerCell(torch.nn.Module):
    """Stateful LIF controller cell (true feedback loop)

    A stateful controller:
      - Takes input u_t and previous membrane v_{t-1}
      - Outputs spike_rate_t and updated membrane v_t

    This enables a true closed-loop across timesteps by feeding back q_t into u_{t+1}.
    """
    def __init__(self, in_features: int, hidden: int, beta: float=0.9, thresh: float=1.0, gamma: float=0.3):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, hidden, dtype=torch.float32)
        self.beta = beta
        self.thresh = thresh
        self.gamma = gamma

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        return torch.zeros(batch_size, self.fc.out_features, device=device, dtype=dtype)

    def step(self, u_t: torch.Tensor, v_prev: torch.Tensor):
        """
        One control step.

        u_t: (B, in_features)
        v_prev: (B, hidden)

        Returns:
          spike_rate_t: (B, hidden)   (here it's binary spikes; we treat it as rate per step)
          v_t: (B, hidden)
        """
        # Ensure float32 while preserving device (CPU or GPU)
        device = u_t.device
        u_t = u_t.to(dtype=torch.float32, device=device)
        v_prev = v_prev.to(dtype=torch.float32, device=device)

        i = self.fc(u_t)
        v = self.beta * v_prev + i
        spk = spike_fn(v - self.thresh, self.gamma)
        v = v - spk * self.thresh
        return spk, v


class GatedQuantumLayer(torch.nn.Module):
    """Wrapper for device-specific gated quantum circuits"""

    def __init__(self, quantum_device):
        super().__init__()
        self.num_qubits = quantum_device.num_qubits
        self.num_layers = quantum_device.num_layers
        # Device returns a callable that accepts params dict
        self.circuit = quantum_device.create_gated_circuit()

    def forward(self, angles, ent_strengths, g_layer, g_qubit, g_edge):
        """Execute gated quantum circuit for each sample in batch"""
        batch_size = angles.shape[0]
        device = angles.device
        outputs = []

        for i in range(batch_size):
            params = {
                'angles': angles[i],
                'ent_strengths': ent_strengths[i] if ent_strengths is not None else None,
                'g_layer': g_layer[i],
                'g_qubit': g_qubit[i],
                'g_edge': g_edge[i] if g_edge is not None else None,
            }
            result = self.circuit(params)
            # Ensure result matches input device and dtype
            if result.device != device:
                result = result.to(device)
            if result.dtype != torch.float32:
                result = result.to(torch.float32)

            outputs.append(result)

        return torch.stack(outputs)


class SNNQNN(torch.nn.Module):
    """
    Closed-loop SNN controller + spike-gated quantum circuit.

    Expected `device` object:
      - device.num_qubits : int
      - device.num_layers : int
      - device.device : PennyLane device instance
    """
    def __init__(
        self,
        device,
        backbone_path=None,
        control_steps: int = 1,          # T: number of closed-loop steps per forward pass
        snn_hidden: int = 256,
        controller_beta: float = 0.9,
        controller_thresh: float = 1.0,
        controller_gamma: float = 0.3,
        gate_alpha: float = 2.0,
        angle_scale: float = np.pi,
        ent_scale: float = np.pi,
        sparsity_lambda: float = 0.0,    # >0 encourages sparse gates (proxy on sigmoid probabilities)
        feedback_scale: float = 0.7,     # scales q_t before feeding back
    ):
        super().__init__()

        # ----- Classical backbone -----
        if backbone_path:
            self.backbone = torchvis.models.resnet18()
            self.backbone.load_state_dict(torch.load(backbone_path))
            self.backbone.eval()
        else:
            self.backbone = torchvis.models.resnet18(
                weights=torchvis.models.ResNet18_Weights.DEFAULT
            )

        self.fc_feat = torch.nn.Linear(1000, 512)
        self.fc_feat2 = torch.nn.Linear(512, 256)

        self.num_qubits = int(device.num_qubits)
        self.num_layers = int(device.num_layers)

        self.control_steps = int(control_steps)
        self.gate_alpha = float(gate_alpha)
        self.angle_scale = float(angle_scale)
        self.ent_scale = float(ent_scale)
        self.sparsity_lambda = float(sparsity_lambda)
        self.feedback_scale = float(feedback_scale)

        # Ring edges for entanglement topology gating
        self.num_edges = self.num_qubits if self.num_qubits > 1 else 0

        # ----- Controller input includes quantum feedback -----
        # u_t = concat(classical_features, q_{t-1})
        controller_in = 256 + self.num_qubits
        self.controller = LIFControllerCell(
            in_features=controller_in,
            hidden=snn_hidden,
            beta=controller_beta,
            thresh=controller_thresh,
            gamma=controller_gamma,
        )

        # ----- Maps from controller spikes -> circuit parameters -----
        self.to_angles = torch.nn.Linear(snn_hidden, self.num_layers * self.num_qubits)
        self.to_gate_layer = torch.nn.Linear(snn_hidden, self.num_layers)
        self.to_gate_qubit = torch.nn.Linear(snn_hidden, self.num_layers * self.num_qubits)

        if self.num_edges > 0:
            self.to_ent = torch.nn.Linear(snn_hidden, self.num_layers * self.num_edges)
            self.to_gate_edge = torch.nn.Linear(snn_hidden, self.num_layers * self.num_edges)
        else:
            self.to_ent = None
            self.to_gate_edge = None

        # ----- Quantum layer -----
        self.quantum_layer = GatedQuantumLayer(device)

        # ----- Classifier head -----
        # We aggregate quantum readouts across control steps -> pooled q -> logits
        self.fc_out = torch.nn.Linear(self.num_qubits, 64, dtype=torch.float32)
        self.fc_out2 = torch.nn.Linear(64, 2, dtype=torch.float32)

        # Expose auxiliary loss term computed each forward call
        self.gate_sparsity_loss = None

    def forward(self, x):
        # ---- Classical features (computed once) ----
        f = self.backbone(x)
        f = torch.nn.functional.relu(self.fc_feat(f))
        f = torch.nn.functional.relu(self.fc_feat2(f))  # (B, 256)

        B = f.shape[0]
        device = f.device
        dtype = f.dtype

        # ---- Initialize feedback and controller state ----
        q_prev = torch.zeros(B, self.num_qubits, device=device, dtype=torch.float32)
        v = self.controller.init_state(B, device)

        # Track quantum readouts over time
        q_steps = []

        # Accumulate sparsity proxy terms over time (optional)
        sparsity_terms = []

        for _ in range(self.control_steps):
            u_t = torch.cat([f, self.feedback_scale * q_prev], dim=1)

            # Controller step: spikes + updated membrane
            spk, v = self.controller.step(u_t, v)  # spk: (B, hidden)

            # ---- Continuous circuit parameters ----
            angles = self.to_angles(spk)
            angles = torch.tanh(angles) * self.angle_scale
            angles = angles.view(B, self.num_layers, self.num_qubits)

            ent = None
            if self.num_edges > 0 and self.to_ent is not None:
                ent = self.to_ent(spk)
                ent = torch.tanh(ent) * self.ent_scale
                ent = ent.view(B, self.num_layers, self.num_edges)

            # ---- Gates (binary) ----
            gate_layer_logits = self.to_gate_layer(spk)  # (B, L)
            gate_qubit_logits = self.to_gate_qubit(spk).view(B, self.num_layers, self.num_qubits)

            g_layer = bin_gate(gate_layer_logits, self.gate_alpha)  # (B, L)
            g_qubit = bin_gate(gate_qubit_logits, self.gate_alpha)  # (B, L, Q)

            g_edge = None
            gate_edge_logits = None
            if self.num_edges > 0 and self.to_gate_edge is not None:
                gate_edge_logits = self.to_gate_edge(spk).view(B, self.num_layers, self.num_edges)
                g_edge = bin_gate(gate_edge_logits, self.gate_alpha)

            # ---- Sparsity proxy (optional) ----
            if self.sparsity_lambda > 0.0:
                p_layer = torch.sigmoid(gate_layer_logits).mean()
                p_qubit = torch.sigmoid(gate_qubit_logits).mean()
                if gate_edge_logits is not None:
                    p_edge = torch.sigmoid(gate_edge_logits).mean()
                    sparsity_terms.append(p_layer + p_qubit + p_edge)
                else:
                    sparsity_terms.append(p_layer + p_qubit)

            # Execute quantum circuit
            # Note loop moved to GatedQuantumLayer due to device-specific execution
            q_t = self.quantum_layer(angles, ent, g_layer, g_qubit, g_edge)
            q_steps.append(q_t)

            # Feedback for next step
            q_prev = q_t

        # ---- Aggregate quantum readouts across steps ----
        q_all = torch.stack(q_steps, dim=0)          # (T, B, Q)
        q_pool = q_all.mean(dim=0).to(torch.float32) # (B, Q)  (simple mean pooling)

        # ---- Set auxiliary sparsity loss ----
        if self.sparsity_lambda > 0.0 and len(sparsity_terms) > 0:
            self.gate_sparsity_loss = self.sparsity_lambda * torch.stack(sparsity_terms).mean()
        else:
            self.gate_sparsity_loss = None

        # ---- Classifier head ----
        h = torch.nn.functional.relu(self.fc_out(q_pool))
        logits = self.fc_out2(h)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs
