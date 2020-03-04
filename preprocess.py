import numpy as np
import cv2
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True, progress=True)

model = torch.nn.Sequential(*(list(model.children())[:-1]))

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

def pad(img, desired_size, height, width):
    delta_w = desired_size - width
    delta_h = desired_size - height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im

def normalize(img):
    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    resized = resized.astype('float32')
    resized /= 255.0
    for py in range(0, 224):
        for px in range(0, 224):
            resized[px][py][0] -= 0.485
            resized[px][py][1] -= 0.456
            resized[px][py][2] -= 0.406
            resized[px][py][0] /= 0.229
            resized[px][py][1] /= 0.224
            resized[px][py][2] /= 0.225
    return resized

def preprocess(img):
    height, width, channels = img.shape

    if height > width:
        img = pad(img, height, height, width)
        img = normalize(img)
    else:
        img = pad(img, width, height, width)
        img = normalize(img)

    image = np.reshape(img, [1, 224, 224, 3])
    image = np.transpose(image, [0, 3, 1, 2])
    image = torch.from_numpy(image)

    feature_vector = model(image.to(device))

    feature_vector = feature_vector.cpu().detach().numpy()

    return feature_vector
