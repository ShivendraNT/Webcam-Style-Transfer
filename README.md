# 🎨 Real-Time Neural Style Transfer with Webcam

This project applies **artistic neural style transfer** to live webcam footage using pre-trained **Transformer Networks**.  
The result is a side-by-side video stream: original webcam feed on the left, stylized output on the right.

## 🚀 Features
- Real-time video style transfer (using OpenCV + PyTorch).
- Switch between multiple styles on the fly:
  - 🖼 Mosaic
  - 🍬 Candy
  - 🌧 Rain Princess
  - 🎭 Udnie
- GPU / MPS (Apple Silicon) acceleration supported.
- Modular PyTorch implementation (custom layers, residual blocks, Gram matrix, etc.).

---

## 🛠 Tech Stack
- **Python 3.9+**
- **PyTorch**
- **Torchvision**
- **OpenCV**
- **NumPy**
- **PIL**
- **Matplotlib**


---

## ⚡ Usage
1. Clone this repo:
   git clone https://github.com/<ShivendraNT>/Webcam-Style-Transfer.git
   cd Webcam-Style-Transfer

2. Install requirements:
pip install torch torchvision opencv-python pillow matplotlib

3. Download the pretrained style models into saved_models/:
mosaic.pth
candy.pth
rain_princess.pth
udnie.pth

4. Run the script

python Webcam-Style_Transfer.py

🎮 Controls
1/2/3/4 → Switch between styles
q → Quit program




📖 References
Fast Neural Style Transfer (PyTorch official example)
Johnson et al., Perceptual Losses for Real-Time Style Transfer and Super-Resolution (2016)

📌 Notes
Ensure saved_models/*.pth are downloaded before running.
For best performance, run on GPU (CUDA) or Apple Silicon (MPS).
Works with real-time webcam input.


---