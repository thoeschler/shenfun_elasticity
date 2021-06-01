import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.collections import LineCollection
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    for spine in ax.spines:
        if spine[0] == 't' or spine[0] == 'r':
            ax.spines[spine].set_visible(False)

def save_disp_figure(u_hat, multiplier=1.0):    
    V = u_hat.function_space()
    u_sol = u_hat.backward(kind='uniform')
    x_lim = V.spaces[0].bases[0].domain[1]
    # y_lim = V.spaces[0].bases[1].domain[1]
    x_, y_ = V.spaces[0].local_mesh(uniform=True)
    X, Y = np.meshgrid(x_, y_, indexing='ij')
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.set_xlim(0, x_lim)
    ax.set_xlabel('$x$ in $\mathrm{mm}$', fontsize=12)
    ax.set_ylabel('$y$ in $\mathrm{mm}$', fontsize=12)
    plot_grid(X, Y, ax=ax,  color="lightgrey")
    plot_grid(X + multiplier*u_sol[0], Y + multiplier*u_sol[1], ax=ax) 
    plt.savefig('displacement.png', dpi=300)
    

def save_cauchy_stress(T):
    space = T[0][0].function_space()
    x_, y_ = space.local_mesh(uniform=True)
    X, Y = np.meshgrid(x_, y_, indexing='ij')
    for i in range(len(space.bases)):
        for j in range(len(space.bases)):
            # stress component in physical space
            stress = T[i][j].backward(kind='uniform')
            # plot
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.set_xlim(space.bases[0].domain[0], space.bases[0].domain[1])
            ax.set_ylim(space.bases[1].domain[0], space.bases[1].domain[1])
            ax.set_xlabel('$x$ in $\mathrm{mm}$', fontsize=12)
            ax.set_ylabel('$y$ in $\mathrm{mm}$', fontsize=12)
            ax.set_title("$\sigma_{" + str(i + 1) + str(j + 1) + "}$ in $\mathrm{MPa}$")
            lim = max(abs(stress.min()), abs(stress.max()))
            norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
            im = plt.pcolormesh(X, Y, stress, cmap='seismic', shading='gouraud', norm=norm)
            fig.colorbar(im, ax=ax)
            plt.savefig('T' + str(i+1) + str(j+1) + '.png', dpi=300)
            
def save_hyper_stress(T3):
    space = T3[0][0][0].function_space()
    x_, y_ = space.local_mesh(uniform=True)
    X, Y = np.meshgrid(x_, y_, indexing='ij')
    for i in range(len(space.bases)):
        for j in range(len(space.bases)):
            for k in range(len(space.bases)):
                # stress component in physical space
                stress = T3[i][j][k].backward(kind='uniform')

                # plot
                fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
                ax.set_xlim(space.bases[0].domain[0], space.bases[0].domain[1])
                ax.set_ylim(space.bases[1].domain[0], space.bases[1].domain[1])
                ax.set_xlabel('$x$ in $\mathrm{mm}$', fontsize=12)
                ax.set_ylabel('$y$ in $\mathrm{mm}$', fontsize=12)
                ax.set_title("$\sigma_{" + str(i + 1) + str(j + 1) + str(k + 1) + "}$ in $\mathrm{N/mm}$")
                lim = max(abs(stress.min()), abs(stress.max()))
                norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
                im = plt.pcolormesh(X, Y, stress, cmap='seismic', shading='gouraud', norm=norm)
                fig.colorbar(im, ax=ax)
                plt.savefig('T' + str(i+1) + str(j+1) + str(k+1) + '.png', dpi=300)
                
def save_traction_vector_gradient(traction_vector, normal_vector):
    space = traction_vector.function_space().spaces[0]
    x_, y_ = space.local_mesh()
    X, Y = np.meshgrid(x_, y_, indexing='ij')
    for i in range(len(space.bases)):
        # component of traction vector in physical space
        comp = traction_vector[i].backward()
        
        # plot
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        ax.set_xlim(space.bases[0].domain[0], space.bases[0].domain[1])
        ax.set_ylim(space.bases[1].domain[0], space.bases[1].domain[1])
        ax.set_xlabel('$x$ in $\mathrm{mm}$', fontsize=12)
        ax.set_ylabel('$y$ in $\mathrm{mm}$', fontsize=12)
        ax.set_title("$t_{" + str(i + 1) + "}$ in $\mathrm{MPa}$")
        lim = max(abs(comp.min()), abs(comp.max()))
        norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
        im = plt.pcolormesh(X, Y, comp, cmap='seismic', shading='gouraud', norm=norm)
        fig.colorbar(im, ax=ax)
        if normal_vector == (0, 1) or normal_vector == (0, -1):
            plt.savefig('t' + str(i+1) + 'ObenUnten.png', dpi=300)
        elif normal_vector == (1, 0) or normal_vector == (-1, 0):
            plt.savefig('t' + str(i+1) + 'RechtsLinks.png', dpi=300)