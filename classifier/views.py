import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
from .models import UploadedImage
import urllib
import torch.nn as nn

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ FIRST MODEL (ResNet50 – object detection)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device)
model.eval()

# ✅ Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ✅ ImageNet labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").split("\n")

ALLOWED_EXT = ['jpg', 'jpeg', 'png', 'webp']

# -----------------------------------------------------------
# ✅ SECOND MODEL (Waste / Garbage Classification)
# -----------------------------------------------------------

MATERIAL_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ✅ Load MobileNetV2
waste_model = models.mobilenet_v2(weights=None)
waste_model.classifier[1] = nn.Linear(1280, len(MATERIAL_CLASSES))

# ✅ Load your custom model weights
waste_model.load_state_dict(
    torch.load("classifier/waste_model.pth", map_location=device)
)

waste_model = waste_model.to(device)
waste_model.eval()


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
            return JsonResponse({"error": "Allowed file types: jpg, jpeg, png, webp"}, status=400)

        uploaded = form.save()
        image_path = uploaded.image.path
        image_url = uploaded.image.url

        # ✅ Load image safely
        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            return JsonResponse({"error": "Invalid or corrupted image"}, status=400)

        img_tensor = transform(image).unsqueeze(0).to(device)

        # ✅ ResNet (object name)
        with torch.no_grad():
            outputs = model(img_tensor)

            _, idx1 = outputs.max(1)
            label1 = imagenet_classes[idx1.item()]
            prob1 = torch.nn.functional.softmax(outputs, dim=1)[0][idx1].item()

            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probs, 5)

            top5 = []
            for i in range(5):
                top5.append({
                    "label": imagenet_classes[top5_idx[i].item()],
                    "confidence": float(top5_prob[i].item())
                })

        # ✅ SECOND MODEL (waste classification)
        with torch.no_grad():
            waste_out = waste_model(img_tensor)
            _, w_idx = waste_out.max(1)

            material_label = MATERIAL_CLASSES[w_idx.item()]
            material_conf = torch.nn.functional.softmax(waste_out, dim=1)[0][w_idx].item()

        # ✅ Save to DB
        uploaded.predicted_label = label1
        uploaded.save()

        return JsonResponse({
            "status": "success",
            "object_prediction": {
                "label": label1,
                "confidence": float(prob1)
            },
            "material_prediction": {
                "label": material_label,
                "confidence": float(material_conf)
            },
            "top5_predictions": top5,
            "image_url": image_url,
            "device_used": "GPU" if torch.cuda.is_available() else "CPU"
        })

    except Exception as e:
        return JsonResponse({"error": "Unexpected error", "details": str(e)}, status=500)
