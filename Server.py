import numpy as np
import time
import matplotlib.pyplot as plt
from Wireless import WiFi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class MiniUNet(nn.Module):
    def __init__(self):
        super(MiniUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_unet_model(data_stream, adc_bits, epochs=20, batch_size=32):
    print("Training UNet model for bitstream reconstruction...")
    print(f"Debug: data_stream shape: {np.asarray(data_stream).shape if hasattr(data_stream, 'shape') else 'Not a NumPy array'}, length: {len(data_stream) if hasattr(data_stream, '__len__') else 'N/A'}")
    if len(data_stream) == 0:
        raise ValueError("data_stream is empty. Cannot train UNet model with zero samples.")

    X = []
    y = []

    for val in data_stream:
        bits = [(val >> i) & 1 for i in reversed(range(adc_bits))]
        X.append(bits)
    
    for val in data_stream:
        bits = [(val >> i) & 1 for i in reversed(range(adc_bits))]
        y.append(bits)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)  # (N, 1, adc_bits, 1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)  # (N, 1, adc_bits, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MiniUNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")

    return model

class RxGPUSystem:
    def __init__(self, wireless_tx):
        if not isinstance(wireless_tx, WiFi):
            raise ValueError("RxGPUSystem only supports WiFi wireless link.")

        self.wireless_tx = wireless_tx
        self.payload_bytes = wireless_tx.payload_bytes
        self.num_payloads = wireless_tx.num_payloads
        self.adc_bits = wireless_tx.adc_bits
        self.tx_latency_us = wireless_tx.transmission_time(self.payload_bytes) * 1e6
        self.rx_power_mw = wireless_tx.Tx_power_noise(self.payload_bytes) * 1000

        print(f"Debug: Training segmentation model with data_stream length: {len(wireless_tx.data_stream)}")
        self.model = train_unet_model(wireless_tx.data_stream, self.adc_bits)

        self.start_time = time.perf_counter()  
        self.bitstream = self.extract_bitstream(wireless_tx.data_stream, self.adc_bits)
        self.reconstructed_data = self.reconstruct_signal()
        end_time = time.perf_counter()
        N = len(wireless_tx.data_stream)  
        self.rx_latency_us = (end_time - self.start_time) * 1e6  

        theoretical_latency_us = N * 30  
        self.rx_latency_us = max(self.rx_latency_us, theoretical_latency_us)  

        self.print_rx_info()

    def extract_bitstream(self, data_stream, adc_bits):
        print("Extracting bitstream from received payloads...")
        data = np.array(data_stream, dtype=np.uint16)
        bits = ((data[:, None] >> np.arange(adc_bits - 1, -1, -1)) & 1)
        return bits  

    def print_rx_info(self):
        print("\n=== Receiver GPU System ===")
        print(f"{'Wireless Type':<25}: WiFi")
        print(f"{'RX Power':<25}: {self.rx_power_mw:.3f} mW")
        print(f"{'Payloads Received':<25}: {self.num_payloads} payloads")
        print(f"{'Payload Size':<25}: {self.payload_bytes} bytes each")
        print(f"{'Total Bitstream Shape':<25}: {self.bitstream.shape}")
        print(f"{'TX Latency':<25}: {self.tx_latency_us:.2f} µs")
        print(f"{'RX Processing Latency':<25}: {self.rx_latency_us:.2f} µs")
        print("==============================\n")

    def reconstruct_signal(self):
        print("Reconstructing signal using segmentation model...")
        input_tensor = torch.tensor(self.bitstream, dtype=torch.float32).unsqueeze(1).unsqueeze(-1) 

        with torch.no_grad():
            for _ in range(3):
                output = self.model(input_tensor).squeeze(-1).squeeze(1).numpy()  
            binary_output = (output > 0.5).astype(int)
            weights = 2 ** np.arange(self.adc_bits - 1, -1, -1)
            reconstructed = (binary_output @ weights).astype(np.uint16)

        print(f"Reconstructed {len(reconstructed)} ADC samples using UNet.")
        return reconstructed

    def compare_with_original(self, original_stream):
        print("Comparing reconstructed signal with original...")
        reconstructed = self.reconstructed_data
        min_len = min(len(original_stream), len(reconstructed))
        original = np.array(original_stream[:min_len])
        reconstructed = reconstructed[:min_len]

        if np.array_equal(original, reconstructed):
            print("Reconstructed signal matches original signal exactly!")
        else:
            diff = np.abs(original - reconstructed)
            max_err = np.max(diff)
            mean_err = np.mean(diff)
            print(f"Mismatch detected: Max Error = {max_err}, Mean Error = {mean_err:.3f}")

        self.plot_comparison(original, reconstructed)

    def plot_comparison(self, original, reconstructed):
        samples = np.arange(len(original))
        error = np.abs(original - reconstructed)

        plt.figure(figsize=(15, 6))

        plt.subplot(2, 1, 1)
        plt.plot(samples, original, label='Original', linewidth=1.5)
        plt.plot(samples, reconstructed, label='Reconstructed', linestyle='--', linewidth=1.5)
        plt.title("Original vs Reconstructed Signal")
        plt.xlabel("Sample Index")
        plt.ylabel("ADC Value")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(samples, error, color='red')
        plt.title("Absolute Reconstruction Error")
        plt.xlabel("Sample Index")
        plt.ylabel("Error")
        plt.grid(True)

        plt.tight_layout()
        plt.show()