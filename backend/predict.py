import torch
from torchvision import transforms
from PIL import Image

from model import MedicalNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classes = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]


model = MedicalNet(num_classes=3)

model.load_state_dict(
    torch.load("../models/best_model.pth", map_location=DEVICE)
)

model.to(DEVICE)

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }