import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths to model files
DISCRIMINATOR_PATH = os.path.join("modles", "discriminator.pth")
GENERATOR_PATH = os.path.join("modles", "generator.pth")

# Load Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(features))

# Load Generator (Modified to accept an image as input)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Load trained models from the models folder
discriminator = Discriminator().to(device)
generator = Generator().to(device)

discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH, map_location=device))
generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=device))

discriminator.eval()
generator.eval()

# Image preprocessing function
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    processed_image = transform(image).unsqueeze(0).to(device)

    # Run prediction with discriminator
    with torch.no_grad():
        output = discriminator(processed_image).item()
    
    result = "Fake" if output < 0.5 else "Real"

    # Generate an image using the generator (taking input image instead of noise)
    with torch.no_grad():
        generated_image_tensor = generator(processed_image).squeeze(0).cpu()

    # Convert tensors to images
    def tensor_to_pil(tensor):
        tensor = (tensor * 0.5) + 0.5  # Denormalize
        image_pil = transforms.ToPILImage()(tensor)
        return image_pil

    generated_image = tensor_to_pil(generated_image_tensor)

    # Save both images to memory
    img_io_original = io.BytesIO()
    img_io_generated = io.BytesIO()
    
    image.save(img_io_original, "PNG")
    generated_image.save(img_io_generated, "PNG")
    
    img_io_original.seek(0)
    img_io_generated.seek(0)

    response = {
        "result": result,
        "original_image": send_file(img_io_original, mimetype="image/png", as_attachment=False, download_name="original.png"),
        "generated_image": send_file(img_io_generated, mimetype="image/png", as_attachment=False, download_name="generated.png")
    }

    return response

@app.route("/predict", methods=["OPTIONS"])
def options():
    response = jsonify({"message": "CORS preflight request successful"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response, 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
