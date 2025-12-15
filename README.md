# Mini-ViT: Vision Transformer on CIFAR-10

A lightweight, reproducible implementation of the Vision Transformer (ViT) architecture, designed to run efficiently on student-grade hardware (e.g., RTX 3050/4050 or CPU). This project trains a custom ViT model on the CIFAR-10 dataset and includes an inference CLI for testing custom images.

## 🚀 Features

*   **From Scratch Implementation**: Complete `PatchEmbedding`, `TransformerBlock`, and `MLP` classes implemented in PyTorch.
*   **Hardware Optimized**: Includes automatic fallback logic to reduce model dimensions (`dim=128` -> `64`) or depth if CUDA Out-Of-Memory (OOM) errors occur.
*   **Reproducible**: Fixed seeds and deterministic reporting.
*   **Interactive Inference**: A CLI tool (`test_mini_vit.py`) to classify local images using the trained model.

## 📂 Project Structure

*   `Mini_ViT_Demo.ipynb`: The main notebook containing the full training pipeline, data loading, model definition, and training loop.
*   `test_mini_vit.py`: A standalone Python script for loading the trained model and running inference on new images.
*   `model_final.pt`: The saved model weights (generated after training).
*   `report.txt`: Automated report containing training metrics and hardware performance stats.
*   `data/`: Directory where CIFAR-10 dataset is downloaded.

## 🛠️ Setup & Installation

Ensure you have Python installed along with the following dependencies:

```bash
pip install torch torchvision pillow numpy
```

*(Note: Running with CUDA support is recommended for faster training, but the code supports CPU-only execution automatically.)*

## 📖 Usage

### 1. Training the Model
Open `Mini_ViT_Demo.ipynb` in Jupyter Notebook or VS Code and run all cells.
*   It will automatically download CIFAR-10.
*   Train for the configured epochs (default: 50).
*   Save the best model to `model_final.pt`.
*   Generate a `report.txt`.

### 2. Running Inference
Once the model is trained, use the CLI script to classify images:

```bash
python test_mini_vit.py
```

*   **Interactive Mode**: The script will ask for an image path.
*   **Drag & Drop**: You can drag an image file into the terminal window to paste its path.
*   Enter `q` to quit.

## 📊 Performance Results

*(Based on a typical run reported in `report.txt`)*

*   **Training Time**: ~43 minutes
*   **Accuracy**: ~78% on Validation Set
*   **Model Config**: Patch Size: 4, Dim: 128, Depth: 4, Heads: 4
*   **Params**: ~0.5M

## ⚠️ Hardware Constraints
This project is specifically tuned for GPUs with ~4GB-6GB VRAM. If you encounter memory errors, the training script is designed to automatically restart with a lighter configuration (e.g., reducing batch size or model depth).
