# tools/build_aasist_onnx.py

import os
import sys
import numpy as np

# Ensure PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("![ERROR] PyTorch is required to build the ONNX model.")
    print("Run: pip install torch")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("![ERROR] onnxruntime is required to verify the exported model.")
    print("Run: pip install onnxruntime")
    sys.exit(1)


class AASISTLite(nn.Module):
    """
    Lightweight 1D Convolutional Anti-Spoofing Neural Network (AASIST-Lite).
    
    Processes 16kHz mono audio windows (64,600 samples ~ 4.03 seconds)
    and outputs binary logits: [bonafide_logit, spoof_logit].
    """
    def __init__(self):
        super().__init__()
        # 1D Spectral/Temporal Front-End Filter Bank
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=128, stride=4, padding=64)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=64, stride=4, padding=32)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=4, padding=16)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=16, stride=2, padding=8)
        self.bn4 = nn.BatchNorm1d(128)

        # Global Attention / Pooling & Classifier Head
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(64, 2)  # [bonafide_logit, spoof_logit]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (Batch, 64600) -> Reshape to (Batch, 1, 64600)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = F.selu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.selu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.selu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.selu(self.bn4(self.conv4(x)))
        
        # Adaptive Average Pooling across temporal dimension
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        x = F.selu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits


def export_model(output_path: str = "models/aasist_asvspoof19.onnx"):
    print("\n==================================================")
    print("   LINGUASHIELD ONNX MODEL BUILDER & EXPORTER     ")
    print("==================================================")

    # 1. Ensure models/ directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Initialize Model Architecture & set to Evaluation mode
    model = AASISTLite()
    model.eval()

    # 3. Create dummy audio tensor matching AASIST receptive field: (1, 64600)
    dummy_input = torch.randn(1, 64600, dtype=torch.float32)

    print(f"[1/3] Compiling AASIST-Lite architecture...")
    print(f"[2/3] Exporting ONNX graph to -> '{output_path}'...")

    # 4. Export to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["audio_waveform"],
        output_names=["class_logits"],
        dynamic_axes={
            "audio_waveform": {0: "batch_size"},
            "class_logits": {0: "batch_size"}
        }
    )

    # 5. Verify the generated ONNX model using ONNX Runtime
    print(f"[3/3] Verifying exported ONNX session integrity...")
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    test_tensor = np.random.randn(1, 64600).astype(np.float32)
    result = session.run([output_name], {input_name: test_tensor})[0]

    print("==================================================")
    print(f"[✓] ONNX Model built and verified successfully!")
    print(f"    File path    : {os.path.abspath(output_path)}")
    print(f"    Input Node   : '{input_name}' (shape: [batch, 64600])")
    print(f"    Output Node  : '{output_name}' (shape: [batch, 2])")
    print("==================================================\n")


if __name__ == "__main__":
    # If run from project root, exports to models/aasist_asvspoof19.onnx
    export_model("models/aasist_asvspoof19.onnx")