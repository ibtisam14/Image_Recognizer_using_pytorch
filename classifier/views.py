import torch # PyTorch library
from torchvision import models, transforms # Pre-trained models and image transformations
from PIL import Image, UnidentifiedImageError # Image processing
from django.http import JsonResponse # JSON response handling
from django.views.decorators.csrf import csrf_exempt # CSRF exemption for views
from .forms import ImageUploadForm # Form for image upload
from .models import UploadedImage # Model for storing uploaded images
import urllib # URL handling
import torch.nn as nn # Neural network modules

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # Load pre-trained ResNet50 model
resnet.eval()  # Set model to evaluation mode

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT) # Load pre-trained MobileNetV2 model
mobilenet.eval()  # Set model to evaluation mode

transform = transforms.Compose([ # Image preprocessing transformations
    transforms.Resize(256), # Resize the image to 256 pixels on the shorter side
    transforms.CenterCrop(224), # Crop the center 224x224 pixels
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize( # Normalize the tensor with mean and std
        mean=[0.485, 0.456, 0.406], # Mean values for normalization
        std=[0.229, 0.224, 0.225] # Standard deviation values for normalization
    )
])

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" # URL for ImageNet class labels
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").split("\n") # Load and decode class labels

ALLOWED_EXT = ['jpg', 'jpeg', 'png', 'webp'] # Allowed image file extensions


def simple_waste_classifier(label): # Simple heuristic-based waste classifier
    label = label.lower() # Convert label to lowercase for comparison

    if any(x in label for x in ["bottle", "plastic", "cup", "container"]): # Check for plastic-related keywords
        return "plastic", 0.90 # Return plastic label with confidence

    if any(x in label for x in ["paper", "book", "document", "newspaper"]): # Check for paper-related keywords
        return "paper", 0.85 # Return paper label with confidence

    if any(x in label for x in ["can", "metal", "aluminum", "tin"]): # Check for metal-related keywords
        return "metal", 0.88 # Return metal label with confidence

    if any(x in label for x in ["glass", "jar", "wine"]): # Check for glass-related keywords
        return "glass", 0.92 # Return glass label with confidence

    if any(x in label for x in ["box", "cardboard", "carton"]): # Check for cardboard-related keywords
        return "cardboard", 0.87 # Return cardboard label with confidence

    return "trash", 0.60 # Default to trash with lower confidence


@csrf_exempt # Exempt the view from CSRF verification
def classify_image(request): # View to handle image classification requests
    if request.method != "POST": # Only allow POST requests
        return JsonResponse({"error": "Only POST method allowed"}, status=405) # Method Not Allowed

    try: # Main try block to catch unexpected errors
        form = ImageUploadForm(request.POST, request.FILES) # Initialize the form with POST data and files

        if not form.is_valid(): # Validate the form
            return JsonResponse({"error": "Invalid form data"}, status=400) # Bad Request

        ext = request.FILES['image'].name.split('.')[-1].lower() # Extract file extension
        if ext not in ALLOWED_EXT: # Check if the extension is allowed
            return JsonResponse( # Return error for disallowed file types
                {"error": "Allowed file types: jpg, jpeg, png, webp"}, # Error message
                status=400 # Bad Request
            )

        uploaded = form.save() # Save the uploaded image
        image_path = uploaded.image.path # Get the file path of the uploaded image
        image_url = uploaded.image.url # Get the URL of the uploaded image

        try: # Load and process the image
            image = Image.open(image_path).convert("RGB") # Open the image and convert to RGB
        except UnidentifiedImageError: # Handle corrupted or invalid images
            return JsonResponse({"error": "Invalid or corrupted image"}, status=400) # Bad Request

        img_tensor = transform(image).unsqueeze(0)   # Apply transformations and add batch dimension

        with torch.no_grad(): # Disable gradient calculation for inference
            resnet_out = resnet(img_tensor) # Get predictions from ResNet model

            _, idx1 = resnet_out.max(1) # Get the index of the highest scoring class
            resnet_label = imagenet_classes[idx1.item()] # Map index to class label
            resnet_conf = torch.nn.functional.softmax(resnet_out, dim=1)[0][idx1].item() # Get confidence score

        material_label, material_conf = simple_waste_classifier(resnet_label) # Classify material type using heuristic function

        with torch.no_grad(): # Disable gradient calculation for inference
            mobile_out = mobilenet(img_tensor) # Get predictions from MobileNet model

            _, idx3 = mobile_out.max(1) # Get the index of the highest scoring class
            mobile_label = imagenet_classes[idx3.item()] # Map index to class label
            mobile_conf = torch.nn.functional.softmax(mobile_out, dim=1)[0][idx3].item() # Get confidence score

        uploaded.predicted_label = resnet_label # Store ResNet predicted label
        uploaded.predicted_confidence = resnet_conf # Store ResNet confidence
        uploaded.save() # Save the updated model instance

        return JsonResponse({ # Return the classification results as JSON
            "status": "success", # Indicate successful classification
            "resnet_prediction": { # ResNet prediction details
                "label": resnet_label, # ResNet predicted label
                "confidence": float(resnet_conf) # ResNet confidence score
            },
            "material_prediction": { # Material classification details
                "label": material_label, # Material predicted label
                "confidence": float(material_conf) # Material confidence score
            },
            "mobilenet_prediction": { # MobileNet prediction details
                "label": mobile_label, # MobileNet predicted label
                "confidence": float(mobile_conf)# MobileNet confidence score
            },
            "image_url": image_url # URL of the uploaded image
        })

    except Exception as e: # Catch any unexpected exceptions
        return JsonResponse({ # Return error response
            "error": "Unexpected error occurred", # Error message
            "details": str(e) # Exception details
        }, status=500) # Internal Server Error
