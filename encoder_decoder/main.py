from train import (Decoder, Encoder, ImageDataset)
import torch
import matplotlib.pyplot as plt
from utils import *

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load(out_path / "encoder.pth", map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(out_path / "decoder.pth", map_location=torch.device('cpu')))

encoder.eval()
decoder.eval()

dataset = ImageDataset(2000, 256, textVariant)
image, _ = dataset[0]
with torch.no_grad():
    latent = encoder(image.unsqueeze(0))
    result = decoder(latent)

    plt.subplot(131)
    plt.imshow(image.squeeze().cpu().numpy())
    plt.subplot(132)
    plt.imshow(result.squeeze().cpu().numpy())
    plt.subplot(133)
    diff = image.squeeze() - result.squeeze()
    plt.imshow(diff.cpu().numpy())
    plt.show()
