import torch   # import pytourch library
from torchvision import models, transforms # import models and transforms from torchvision
from PIL import Image # import Image from PIL to open and manupulate files 
import urllib  # import urllib to handle URL operations

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # load the pretrained ResNet50 model
model.eval() # set the model to evaluation mode

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" # URL to fetch ImageNet class labels
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").split("\n") # read and decode the class labels

transform = transforms.Compose([ # define the image transformations
    transforms.Resize(256), # resize the image to 256 pixels
    transforms.CenterCrop(224),# center crop the image to 224x224 pixels
    transforms.ToTensor(), # convert the image to a tensor
    transforms.Normalize( # normalize the image with mean and std
        mean=[0.485, 0.456, 0.406], # mean values for normalization
        std=[0.229, 0.224, 0.225] # std values for normalization
    )
])

image_path = "media/test.jpeg"  # path to the test image
image = Image.open(image_path).convert('RGB') # open the image and convert it to RGB

img_t = transform(image).unsqueeze(0) # apply the transformations and add a batch dimension
with torch.no_grad(): # disable gradient calculation
    outputs = model(img_t) # get the model outputs
    _, predicted = outputs.max(1) # get the index of the max log-probability

predicted_label = imagenet_classes[predicted.item()] # get the predicted label from the class labels
print(f"Prediction: {predicted_label}") # print the predicted label
