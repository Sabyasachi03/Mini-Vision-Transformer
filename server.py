from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import sys
import os

# Add current directory to path so we can import test_mini_vit
sys.path.append(os.getcwd())

# Import model architecture and loading logic
try:
    from test_mini_vit import load_model, MiniViT, Config, PatchEmbedding, MLP, TransformerBlock
except ImportError:
    # Check if we are running from a different directory
    print("Could not import test_mini_vit. Ensure you are running from the project root.")
    sys.exit(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally on startup
model = None
config = None

@app.on_event("startup")
async def startup_event():
    global model, config
    print("Loading model...")
    # Ensure model_final.pt exists or let load_model handle it
    model, config = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Transform (Must match training/inference logic)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(config.device)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        predicted_label = classes[predicted_idx.item()]
        
    return {
        "class": predicted_label,
        "confidence": float(confidence.item())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
