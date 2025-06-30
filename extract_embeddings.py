import torch
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import json

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained ResNet50 (remove classification layer)
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # get 2048-dim embeddings
resnet = resnet.to(device).eval()

# Image preprocessor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  #deafult mean
        std=[0.229, 0.224, 0.225]    #default std deviation
    )
])

# Paths
crop_dir = Path("outputs/cropped_players")
meta_dir = Path("outputs/metadata")
embedding_dir = Path("outputs/embeddings")
embedding_dir.mkdir(parents=True, exist_ok=True)

for img_path in crop_dir.glob("*.jpg"):
    # Loading metadata
    json_path = meta_dir / f"{img_path.stem}.jpg.json"
    if not json_path.exists():
        continue

    with open(json_path) as f:
        metadata = json.load(f)

    # Loading image
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Extracting embedding
    with torch.no_grad():
        embedding = resnet(input_tensor).squeeze(0).cpu()

    # Saving embedding
    torch.save({
        "id": metadata["id"],
        "embedding": embedding,
        "crop_path": str(img_path),
        "metadata": metadata
    }, embedding_dir / f"{img_path.stem}.pt")

