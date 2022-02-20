import torch
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([           
transforms.Resize(256),                    
transforms.CenterCrop(224),                
transforms.ToTensor(),                     
transforms.Normalize(                      
mean=[0.485, 0.456, 0.406],                
std=[0.229, 0.224, 0.225]                  
)])


# Model

model = torch.load('last.pt')
device = torch.device('cuda')

# Image
img = Image.open('test.jpg')
imgt = transform(img)
batch_t = torch.unsqueeze(imgt, 0)
# out = model(batch_t)
print(model)
classes = [ 'helmet', 'vest', 'No helmet']

# Inference
# _, index = torch.max(out, 1)
