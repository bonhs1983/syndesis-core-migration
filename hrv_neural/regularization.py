import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt

def kalman(z,q=1.0,r=30.0):
    xhat,P = z[0],1.0
    out=[xhat]
    for k in range(1,len(z)):
        xpred,Ppred=xhat,P+q
        K=Ppred/(Ppred+r)
        xhat=xpred+K*(z[k]-xpred)
        P=(1-K)*Ppred
        out.append(xhat)
    return np.array(out)

def demo():
    t=np.linspace(0,10,500)
    raw=np.sin(t)+0.3*np.random.randn(len(t))
    filt=kalman(raw)
    plt.plot(raw,alpha=.4,label='raw');plt.plot(filt,label='kalman')
    plt.legend();plt.show()
