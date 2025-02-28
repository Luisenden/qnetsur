import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  

from cocoex import Suite
plt.style.use("seaborn-v0_8-paper")
font = 16
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'arial',
    'font.size': font,
    'axes.labelsize': font,  
    'xtick.labelsize': font,  
    'ytick.labelsize': font, 
    'legend.fontsize': font,
    'legend.title_fontsize': font,
    'axes.titlesize': font
})
plt.rcParams.update({'lines.markeredgewidth': 0.1})
output_path = './'

suite = Suite("bbob-noisy", "year: 2009", "dimensions: 2") # https://numbbo.github.io/coco-doc/C/#suite-parameters

for i in range(3):
    for j in range(1,10):
        fun = suite.get_problem(f'bbob_noisy_f1{i}{j}_i01_d02')

        X1 = np.linspace(-5, 5, 500)
        X2 = np.linspace(-5, 5, 500)
        data = np.array([[fun([x1, x2]) for x1 in X1] for x2 in X2])
        norm = np.linalg.norm(data)
        data = data/norm
        data = abs(np.log(data))
        X1, X2 = np.meshgrid(X1, X2)

        fig, ax = plt.subplots(1, figsize=(5, 5))
        imag = ax.imshow(data, cmap='Greys')
        ax.set(xticks=np.linspace(0, 500, 5), xticklabels=np.arange(-5, 5.5, 2.5),\
                yticks=np.linspace(0, 500, 5), yticklabels=np.arange(-5, 5.5, 2.5))
        plt.colorbar(imag, shrink=0.7)
        plt.title(f'Ranked heatmap of f1{i}{j}, 2D, inst. 1')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.tight_layout()
        plt.savefig(output_path+f'f1{i}{j}.pdf')
        plt.close()

