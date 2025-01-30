from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import torch
import io
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
from torchvision import models
from fastapi.middleware.cors import CORSMiddleware


# Define allowed origins (use "*" to allow all)
origins = [
    "http://localhost:3000",  # Allow frontend in development
    "https://yourfrontend.com",  # Add your production frontend domain
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow only specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app = FastAPI()

# ============================
# 1. Load Pretrained Models
# ============================
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

class Discriminator(nn.Module):
    def __init__(self, pretrained=True):
        super(Discriminator, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(features))

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator(pretrained=True).to(device)

# Load trained weights
generator.load_state_dict(torch.load("modles/generator.pth", map_location=device))
discriminator.load_state_dict(torch.load("modles/discriminator.pth", map_location=device))

generator.eval()
discriminator.eval()

# ============================
# 2. Define Transform
# ============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ============================
# 3. Helper Function to Visualize Artifacts
# ============================
def generate_artifact_image(input_image, generated_image):
    """ Create an artifact image showing differences between input and generated images """
    input_np = input_image.permute(1, 2, 0).cpu().numpy()
    generated_np = generated_image.detach().permute(1, 2, 0).cpu().numpy()
    artifact_np = np.abs(input_np - generated_np)

    # Convert to PIL Image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow((input_np * 0.5 + 0.5))  # De-normalize
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow((generated_np * 0.5 + 0.5))
    axes[1].set_title("Generated Image")
    axes[1].axis("off")

    axes[2].imshow(artifact_np)
    axes[2].set_title("Artifact (Difference)")
    axes[2].axis("off")

    # Save the image as bytes
    img_byte_arr = io.BytesIO()
    plt.savefig(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    plt.close(fig)

    return img_byte_arr

# ============================
# 4. API Endpoint
# ============================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image, and the API will determine if it's real or fake.
    The response includes the classification and an artifact visualization.
    """
    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Discriminator prediction
    real_or_fake_prob = discriminator(image_tensor).item()
    classification = "REAL" if real_or_fake_prob > 0.5 else "FAKE"

    # Generate reconstructed image
    generated_image = generator(image_tensor).squeeze(0)  # Remove batch dimension
    artifact_img_bytes = generate_artifact_image(image_tensor.squeeze(0), generated_image)

    # Return JSON with the image
    return StreamingResponse(artifact_img_bytes, media_type="image/png",
                             headers={"X-Classification": classification, "X-Confidence": str(real_or_fake_prob)})

# ============================
# 5. Root Endpoint (Optional)
# ============================
@app.get("/")
def root():
    return {"message": "Welcome to the Medical Deepfake Detection API!"}
