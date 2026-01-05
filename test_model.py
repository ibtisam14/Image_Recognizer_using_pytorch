import torch   
from torchvision import models, transforms 
from PIL import Image
import urllib  

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval() 

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" 
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").split("\n") 

transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize( 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225] 
    )
])

image_path = "media/test.jpeg" 
image = Image.open(image_path).convert('RGB') 

img_t = transform(image).unsqueeze(0) 
with torch.no_grad():
    outputs = model(img_t) 
    _, predicted = outputs.max(1) 

predicted_label = imagenet_classes[predicted.item()] 
print(f"Prediction: {predicted_label}") 
