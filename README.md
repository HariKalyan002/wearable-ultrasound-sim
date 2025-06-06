# wearable-ultrasound-sim
Wearable ultrasound technology supports real-time, mobile physiological monitoring, making it increasingly valuable for both medical diagnostics and fitness applications. However, maintaining low power consumption is a major challenge, as these devices often exceed acceptable power limits, affecting battery efficiency and thermal management. While data reduction approaches like compression and subsampling can help minimize data volume, existing studies often neglect the unique processing and hardware requirements of wearable ultrasound systems. To bridge this gap, we introduce a Python-based simulation platform tailored to ultrasound hardware, enabling comprehensive analysis of data reduction strategies and system-level trade-offs. This tool facilitates the development of energy-efficient, high-performance wearable ultrasound solutions.
# Code Pipeline
Wearable_ultrasound system pipeline:
Wearable --> Edge device --> Server device
Wearable pipeline:
Transducers --> Channel subsampling --> ADC --> Data reduction Techniques --> MCU --> Wireless TX 
# Code Organization
The overall wearable ultrasound system pipeline is defined in the `Top` file.

* `Wearable.py`: This is the core module responsible for collecting and processing data directly from the tissue.
* `Edge.py`: This component receives data from the wearable module and performs reconstruction using sparse signal processing techniques.
* `Server.py`: This module gathers processed data from both the wearable and edge devices, and further enhances reconstruction using machine learning algorithms.
* `Optimizer.py`: This module implements convex optimization techniques to balance power consumption and latency in wearable ultrasound systems. Its primary objective is to minimize on-device power usage by determining the optimal combination of compression scenario, data reduction method, and compression ratio.
Installing Wearable_ultrasound
The code is implemented in Python and relies on minimal external dependencies, all of which are specified in the `requirements.txt` file.



