import torch   # PyTorch library for tensor computations and deep learning
from torchvision import models, transforms  # Pre-trained models and image transformations
from PIL import Image # Image processing library
import urllib  # URL handling library

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # Load pre-trained ResNet50 model
model.eval() # Set model to evaluation mode

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" # URL for ImageNet class labels
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").split("\n") # Load and decode class labels

transform = transforms.Compose([ # Image preprocessing transformations
    transforms.Resize(256), # Resize the image to 256 pixels on the shorter side
    transforms.CenterCrop(224), # Crop the center 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(  # Normalize the tensor with mean and std
        mean=[0.485, 0.456, 0.406],  # Mean values for normalization
        std=[0.229, 0.224, 0.225]  # Standard deviation values for normalization
    )
])

image_path = "media/test.jpeg" # Path to the test image
image = Image.open(image_path).convert('RGB') # Open and convert the image to RGB format

img_t = transform(image).unsqueeze(0) # Apply transformations and add batch dimension
with torch.no_grad(): # Disable gradient calculation for inference
    outputs = model(img_t) # Get model predictions
    _, predicted = outputs.max(1)  # Get the index of the highest predicted class

predicted_label = imagenet_classes[predicted.item()] # Map index to class label
print(f"Prediction: {predicted_label}") # Print the predicted label
