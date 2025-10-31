import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
from .models import UploadedImage
import urllib

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ ResNet model
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


@csrf_exempt
def classify_image(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        form = ImageUploadForm(request.POST, request.FILES)

        if not form.is_valid():
            return JsonResponse({"error": "Invalid form data"}, status=400)

        # ✅ File extension checking
        ext = request.FILES['image'].name.split('.')[-1].lower()
        if ext not in ALLOWED_EXT:
            return JsonResponse({"error": "Allowed file types: jpg, jpeg, png, webp"}, status=400)

        # ✅ Save image
        uploaded = form.save()
        image_path = uploaded.image.path
        image_url = uploaded.image.url

        # ✅ Load image safely
        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            return JsonResponse({"error": "Invalid or corrupted image"}, status=400)

        # ✅ Preprocess
        img_tensor = transform(image).unsqueeze(0).to(device)

        # ✅ Inference
        with torch.no_grad():
            outputs = model(img_tensor)

            # Top 1
            _, idx1 = outputs.max(1)
            label1 = imagenet_classes[idx1.item()]
            prob1 = torch.nn.functional.softmax(outputs, dim=1)[0][idx1].item()

            # Top 5
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probs, 5)

            top5 = []
            for i in range(5):
                top5.append({
                    "label": imagenet_classes[top5_idx[i].item()],
                    "confidence": float(top5_prob[i].item())
                })

        # ✅ Save label in DB
        uploaded.predicted_label = label1
        uploaded.save()

        return JsonResponse({
            "status": "success",
            "top1_prediction": {
                "label": label1,
                "confidence": float(prob1)
            },
            "top5_predictions": top5,
            "image_url": image_url,
            "device_used": "GPU" if torch.cuda.is_available() else "CPU"
        })

    except Exception as e:
        return JsonResponse({"error": "Unexpected error", "details": str(e)}, status=500)
