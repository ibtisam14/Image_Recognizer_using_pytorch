import torch
from torchvision import models, transforms
from PIL import Image
from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
import json
import urllib

# Load pretrained model once (ResNet50)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Transformation for input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").split("\n")

def classify_image(request):
    prediction = None
    image_url = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded = form.save()
            image_path = uploaded.image.path
            image_url = uploaded.image.url

            # Process the image
            image = Image.open(image_path).convert('RGB')
            img_t = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_t)
                _, predicted = outputs.max(1)
                predicted_label = imagenet_classes[predicted.item()]

            uploaded.predicted_label = predicted_label
            uploaded.save()
            prediction = predicted_label
    else:
        form = ImageUploadForm()

    return render(request, "classifier/upload.html", {
        "form": form,
        "prediction": prediction,
        "image_url": image_url
    })
