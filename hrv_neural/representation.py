import torch, torch.nn as nn
import pytorch_lightning as pl
from umap import UMAP
import matplotlib.pyplot as plt

class VAE(pl.LightningModule):
    def __init__(self,in_dim=6,z_dim=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,32),nn.ReLU(),nn.Linear(32, z_dim*2))
        self.dec = nn.Sequential(nn.Linear(z_dim,32),nn.ReLU(),nn.Linear(32,in_dim))
    def encode(self,x):
        params=self.enc(x)
        mu,logvar = params.chunk(2,dim=-1)
        return mu,logvar
    def reparam(self,mu,logvar):
        return mu + torch.randn_like(mu)*torch.exp(0.5*logvar)
    def forward(self,x):
        mu,logvar=self.encode(x)
        z=self.reparam(mu,logvar)
        return self.dec(z),mu,logvar
    def training_step(self,batch,_):
        x=batch[0]
        x_hat,mu,logvar=self(x)
        recon=nn.functional.mse_loss(x_hat,x)
        kl=-0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
        return recon+kl
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),1e-3)

def demo():
    data=torch.randn(128,6)
    vae=VAE()
    trainer=pl.Trainer(max_epochs=1,logger=False,enable_progress_bar=False)
    trainer.fit(vae,torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data),batch_size=32))
    with torch.no_grad():
        mu,_=vae.encode(data)
    emb=UMAP(n_components=2).fit_transform(mu.detach().cpu().numpy())
    plt.scatter(emb[:,0],emb[:,1])
    plt.title('UMAP of VAE latent'); plt.show()
