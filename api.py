from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import io
from torchvision import transforms
from torch import nn
from torchvision import models

app = FastAPI()

# =========================
# Load Models
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Generator
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, pretrained=True):
        super(Discriminator, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(features))

# Load trained models
generator = Generator().to(device)
discriminator = Discriminator(pretrained=False).to(device)

generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
discriminator.load_state_dict(torch.load("models/discriminator.pth", map_location=device))

generator.eval()
discriminator.eval()

# =========================
# Image Preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict if the image is real or fake
    real_or_fake_prob = discriminator(image_tensor).item()
    prediction = "REAL" if real_or_fake_prob > 0.5 else "FAKE"

    return {"prediction": prediction, "confidence": round(real_or_fake_prob, 2)}

