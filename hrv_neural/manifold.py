import numpy as np, torch, torch.nn as nn
from umap import UMAP
from scipy.stats import zscore
import matplotlib.pyplot as plt

def demo():
    N=256
    data=torch.randn(N,6)
    # simple autoencoder
    encoder=nn.Linear(6,2)
    with torch.no_grad():
        latent=encoder(data).numpy()
    emb=UMAP(n_components=2).fit_transform(data.numpy())
    radius=np.linalg.norm(latent,axis=1)
    risk_idx=np.where(zscore(radius)>2)[0]
    plt.scatter(emb[:,0],emb[:,1],c=radius,cmap='viridis')
    plt.scatter(emb[risk_idx,0],emb[risk_idx,1],edgecolors='r',facecolors='none')
    plt.title('Risky zones circled'); plt.show()
