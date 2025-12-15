import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# ----------------- Configuration ----------------- #
class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_size = 32
        self.patch_size = 4
        self.num_classes = 10
        # Default Architecture (Must match training!)
        self.dim = 128
        self.depth = 4
        self.heads = 4
        self.mlp_dim = 256

# ----------------- Model Architecture ----------------- #
# (Must match the training script exactly)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels=3):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class MiniViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbedding(config.image_size, config.patch_size, config.dim)
        num_patches = (config.image_size // config.patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.dim, config.heads, config.mlp_dim)
            for _ in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])

# ----------------- Inference Logic ----------------- #

def load_model(model_path='model_final.pt'):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please ensure you have trained the model using the notebook first.")
        sys.exit(1)
        
    config = Config()
    
    # Try to load and handle potential config mismatches (fallback support)
    state_dict = torch.load(model_path, map_location=config.device)
    
    # Simple check for fallback configs based on state dict shape
    # pos_embed shape is [1, 65, dim]
    saved_dim = state_dict['pos_embed'].shape[2]
    if saved_dim != config.dim:
        print(f"Detected different model dimension in checkpoint ({saved_dim} vs {config.dim}). Adjusting config...")
        config.dim = saved_dim
        # Heuristic: if dim is 64, mlp likely 128. If dim 128, mlp 256.
        config.mlp_dim = saved_dim * 2
        
    # Check depth by counting blocks
    saved_blocks = sum(1 for k in state_dict.keys() if 'blocks.' in k and '.norm1.weight' in k)
    if saved_blocks != config.depth:
        print(f"Detected different depth in checkpoint ({saved_blocks} blocks vs {config.depth}). Adjusting config...")
        config.depth = saved_blocks

    model = MiniViT(config).to(config.device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded successfully on {config.device.upper()}")
    print(f"Config: Dim={config.dim}, Depth={config.depth}")
    return model, config

def predict_image(model, config, image_path):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return

    try:
        image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            predicted_label = classes[predicted_idx.item()]
            
        print(f"\n✅ Prediction: {predicted_label.upper()}")
        print(f"📊 Confidence: {confidence.item()*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")

def main():
    print("--- Mini-ViT Inference CLI ---")
    model, config = load_model()
    
    print("\nTip: You can drag and drop images into the terminal to paste the path.")
    while True:
        try:
            image_path = input("\nEnter image path (or 'q' to quit): ").strip()
            # Remove quotes if OS adds them
            image_path = image_path.strip('"').strip("'")
            
            if image_path.lower() in ['q', 'quit', 'exit']:
                break
                
            if image_path == "":
                continue
                
            predict_image(model, config, image_path)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
