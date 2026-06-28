import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode features via Y-rotations
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    
    # Variational layers
    n_layers = weights.shape[0]
    for layer in range(n_layers):
        for wire in range(n_qubits):
            qml.RX(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RZ(weights[layer, wire, 2], wires=wire)
        # Entanglement
        for wire in range(n_qubits):
            qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumClassicalReconstructor(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        # Encoder: 16x16 -> 4 features
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Decoder: 4 features -> 16x16
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        features = self.encoder(x_flat)
        # Map to [-pi/2, pi/2]
        q_inputs = features * (3.14159265 / 2.0)
        q_outputs = self.q_layer(q_inputs)
        recon_flat = self.decoder(q_outputs)
        recon = recon_flat.view(batch_size, 1, 16, 16)
        return recon

class ClassicalOnlyReconstructor(nn.Module):
    """
    Classical comparison network with same architecture but a linear layer replacing the VQC.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )
        
        # Classical analog to the quantum layer
        self.c_layer = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        features = self.encoder(x_flat)
        c_outputs = self.c_layer(features)
        recon_flat = self.decoder(c_outputs)
        recon = recon_flat.view(batch_size, 1, 16, 16)
        return recon

class QuantumResidualAttentionReconstructor(nn.Module):
    """
    Upgraded hybrid reconstruction network addressing the barren plateau problem.
    Incorporates a classical residual connection and an attention gate to control
    the gradient flow and feature integration between classical and quantum layers.
    """
    def __init__(self, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        self.att_gate = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        features = self.encoder(x_flat)
        
        q_inputs = features * (3.14159265 / 2.0)
        q_outputs = self.q_layer(q_inputs)
        
        att_weights = self.att_gate(q_outputs)
        gated_q = q_outputs * att_weights
        
        integrated = gated_q + self.alpha * features
        recon_flat = self.decoder(integrated)
        recon = recon_flat.view(batch_size, 1, 16, 16)
        return recon
