from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

model = resnet50(weights=ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_plant_species = 16 #num of plant unique plant species in the dataset
model.fc = nn.Linear(model.fc.in_features, num_plant_species)
