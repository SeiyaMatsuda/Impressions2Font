import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from .mylib import tile
from sklearn.decomposition import PCA
def visualizer(path, G_model, z, char_num, label, device):
    G_model.eval()
    z_shape = z.shape
    label_shape = label.shape
    char_cl = torch.tensor([0, 1, 2, 7, 4, 17, 14, 13, 18])
    char = torch.eye(char_num)[char_cl].repeat(z_shape[0] * label_shape[0], 1).to(device)
    char_num = len(char_cl)
    z = tile(z, 0, char_num).repeat(label_shape[0], 1).to(device)
    label = tile(label, 0, char_num * z_shape[0]).to(device)
    with torch.no_grad():
        samples = G_model(z, char, label)[0].data.cpu()
        samples = F.interpolate(samples, (128, 128), mode='nearest')
        samples = samples/2 + 0.5
        save_image(samples, path, nrow=char_num)

def learning_curve(dict, path, title ='learning_curve', x_label = 'epoch', y_label = 'loss'):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Traing score と Test score をプロット
    for key, value in dict.items():
        plt.plot(range(len(value)), np.array(value),  label=str(key))
    plt.legend()
    plt.savefig(path)
    plt.clf()
    plt.close()

def imscatter(x, y, data, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    artists = []
    for x0, y0, d in zip(x, y, data):
        im = OffsetImage(d, cmap = plt.cm.gray_r, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    return artists
