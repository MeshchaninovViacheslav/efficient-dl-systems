# model.py
import torch
import requests
from PIL import Image
from io import BytesIO
from torchvision import models, transforms
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

class InstanceDetectionModel:
    def __init__(self, score_threshold=0.75):
        # Load a pretrained instance segmentation model (Mask R-CNN)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.category_names = weights.meta["categories"]
        self.model = models.detection.maskrcnn_resnet50_fpn(weights=weights, box_score_thresh=score_threshold)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def predict(self, image_url: str):
        # Download the image
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ValueError(f"Could not download image from {image_url}")
        print("Downloaded image from {image_url}")
        image_data = response.content
        image = Image.open(BytesIO(image_data)).convert("RGB")

        x = self.transform(image).to(self.device)
        with torch.no_grad():
            predictions = self.model([x])[0]
        
        # Extract labels from predictions
        labels = predictions['labels'].tolist()
        
        # Convert label indices to category names
        detected_objects = [self.category_names[label] for label in labels]
        
        return detected_objects


model = InstanceDetectionModel(score_threshold=0.75)
