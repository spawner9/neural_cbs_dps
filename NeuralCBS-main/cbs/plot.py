import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def plot2Dimage(x_start=0,x_end=128,
                y_start=0,y_end=128,
                dx=1,f=None,
                cmap='RdBu_r',
                title="Wave Field",
                x_label='X',
                y_label='Y',
                figsize=(7,5),
                levels=100):
    x = torch.arange(x_start,x_end)*dx
    y = torch.arange(y_start,y_end)*dx
    X ,Y = torch.meshgrid(x,y,indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    u = f.flatten()
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.tricontourf(X,Y,u,levels=levels,cmap=cmap)
    cbar = plt.colorbar(image)
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3e'))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.invert_yaxis()
    plt.show()