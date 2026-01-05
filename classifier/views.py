import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
from .models import UploadedImage
import urllib
import torch.nn as nn

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.eval()  

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet.eval() 

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").split("\n")

ALLOWED_EXT = ['jpg', 'jpeg', 'png', 'webp']


def simple_waste_classifier(label):
    label = label.lower()

    if any(x in label for x in ["bottle", "plastic", "cup", "container"]):
        return "plastic", 0.90

    if any(x in label for x in ["paper", "book", "document", "newspaper"]):
        return "paper", 0.85

    if any(x in label for x in ["can", "metal", "aluminum", "tin"]):
        return "metal", 0.88

    if any(x in label for x in ["glass", "jar", "wine"]):
        return "glass", 0.92

    if any(x in label for x in ["box", "cardboard", "carton"]):
        return "cardboard", 0.87

    return "trash", 0.60


@csrf_exempt
def classify_image(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        form = ImageUploadForm(request.POST, request.FILES)

        if not form.is_valid():
            return JsonResponse({"error": "Invalid form data"}, status=400)

        ext = request.FILES['image'].name.split('.')[-1].lower()
        if ext not in ALLOWED_EXT:
            return JsonResponse(
                {"error": "Allowed file types: jpg, jpeg, png, webp"},
                status=400
            )

        uploaded = form.save()
        image_path = uploaded.image.path
        image_url = uploaded.image.url

        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            return JsonResponse({"error": "Invalid or corrupted image"}, status=400)

        img_tensor = transform(image).unsqueeze(0)  

        with torch.no_grad():
            resnet_out = resnet(img_tensor)

            _, idx1 = resnet_out.max(1)
            resnet_label = imagenet_classes[idx1.item()]
            resnet_conf = torch.nn.functional.softmax(resnet_out, dim=1)[0][idx1].item()

        material_label, material_conf = simple_waste_classifier(resnet_label)

        with torch.no_grad():
            mobile_out = mobilenet(img_tensor)

            _, idx3 = mobile_out.max(1)
            mobile_label = imagenet_classes[idx3.item()]
            mobile_conf = torch.nn.functional.softmax(mobile_out, dim=1)[0][idx3].item()

        uploaded.predicted_label = resnet_label
        uploaded.save()

        return JsonResponse({
            "status": "success",
            "resnet_prediction": {
                "label": resnet_label,
                "confidence": float(resnet_conf)
            },
            "material_prediction": {
                "label": material_label,
                "confidence": float(material_conf)
            },
            "mobilenet_prediction": {
                "label": mobile_label,
                "confidence": float(mobile_conf)
            },
            "image_url": image_url
        })

    except Exception as e:
        return JsonResponse({
            "error": "Unexpected error occurred",
            "details": str(e)
        }, status=500)
