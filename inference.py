import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights



device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = resnet50(weights=None)
    #model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features,2)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model = model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])
])

def predict(image_path,model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _,pred = torch.max(output,1)
        if pred==0:
            return "Defect"
        else:
            return "Normal"

# TEST CODE ONLY
if __name__ == "__main__":

    model = load_model("models/corrosion_model.pth")

    result = predict("input_images/fault1.jpg", model)

    print(result)