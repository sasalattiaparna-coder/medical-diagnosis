import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MedicalNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MedicalNet(num_classes=4)
model.load_state_dict(torch.load('../models/best_model.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


gradients = None
activations = None


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]



def forward_hook(module, input, output):
    global activations
    activations = output


layer = model.model.layer4
layer.register_forward_hook(forward_hook)
layer.register_backward_hook(backward_hook)



def generate_heatmap(image_path):
    image = Image.open(image_path).convert('RGB')

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    output = model(tensor)

    pred_class = output.argmax(dim=1)

    model.zero_grad()

    output[0, pred_class].backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu(), 0)
    heatmap /= torch.max(torch.tensor(heatmap))

    image = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap.numpy(), (image.shape[1], image.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + image

    output_path = 'heatmap.jpg'

    cv2.imwrite(output_path, superimposed_img)

    return output_path