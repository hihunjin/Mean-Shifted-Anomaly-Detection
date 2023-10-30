import torch
import torchvision

import segmentation_models_pytorch as smp
from vgg_loss import VGGLoss


model = smp.Unet(
    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(model, train_dataset, optimizer: torch.optim.Optimizer):
    if not model.training:
        model.train()
    perceptual_loss = VGGLoss()
    for epoch in range(5):
        for batch in [torch.rand(1, 3, 224, 224) for _ in range(10)]:
            optimizer.zero_grad()
            output = model(batch)
            loss: torch.Tensor = perceptual_loss(batch, output)
            loss.backward()
            optimizer.step()

train(model, None, optimizer)
