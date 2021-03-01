import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.collections import LineCollection
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
import math

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
    Space = T[0][0].function_space()
    x_, y_ = Space.local_mesh()
    X, Y = np.meshgrid(x_, y_, indexing='ij')
    for i in range(len(Space.bases)):
        for j in range(len(Space.bases)):
            x_lim = Space.bases[0].domain[1]
            y_lim = Space.bases[1].domain[1]
            x_, y_ = Space.local_mesh(uniform=True)
            X, Y = np.meshgrid(x_, y_, indexing='ij')
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.set_xlim(0., x_lim)
            ax.set_ylim(0., y_lim)
            ax.set_xlabel('$x$ in $\mathrm{mm}$', fontsize=12)
            ax.set_ylabel('$y$ in $\mathrm{mm}$', fontsize=12)
            ax.set_title("$\sigma_{" + str(i + 1) + str(j + 1) + "}$ in $\mathrm{MPa}$")
            lim = max(abs(T[i][j].min()), abs(T[i][j].max()))
            norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
            im = plt.pcolormesh(X, Y, T[i][j], cmap='seismic', shading='gouraud', norm=norm)
            # if lim >= 1.:
            #     ticks = np.linspace(math.ceil(-lim), math.floor(lim), 6)
            #     if abs(T[i][j].min()) > abs(T[i][j].max()) and math.floor(-lim) < -lim:
            #         ticks = np.insert(ticks, 0, math.floor(-lim*1000.)/1000.)
            #     if abs(T[i][j].min()) <= abs(T[i][j].max()) and math.ceil(lim) > lim:
            #         ticks = np.append(ticks, math.floor(lim*1000.) / 1000.)
            # else:
            #     first_decimal_neq_zero = math.floor( math.log10(lim))
            #     assert first_decimal_neq_zero < 0.
            #     ticks = np.linspace(math.ceil(-lim*10**abs(first_decimal_neq_zero))*10**first_decimal_neq_zero, \
            #                         math.floor(lim*10**abs(first_decimal_neq_zero))*10**first_decimal_neq_zero, 6)
            #     if abs(T[i][j].min()) > abs(T[i][j].max()) and math.floor(-lim*10**abs(first_decimal_neq_zero))*10**first_decimal_neq_zero < -lim:
            #         ticks = np.insert(ticks, 0, -lim)
            #     if abs(T[i][j].min()) <= abs(T[i][j].max()) and math.ceil(lim*10**abs(first_decimal_neq_zero))*10**first_decimal_neq_zero > lim:
            #         ticks = np.append(ticks, lim)
            # print(ticks)
            fig.colorbar(im, ax=ax)
            plt.savefig('T' + str(i+1) + str(j+1) + '.png', dpi=300)
            
def save_hyper_stress(T3):
    Space = T3[0][0][0].function_space()
    x_, y_ = Space.local_mesh()
    X, Y = np.meshgrid(x_, y_, indexing='ij')
    for i in range(len(Space.bases)):
        for j in range(len(Space.bases)):
            for k in range(len(Space.bases)):
                x_lim = Space.bases[0].domain[1]
                y_lim = Space.bases[1].domain[1]
                x_, y_ = Space.local_mesh(uniform=True)
                X, Y = np.meshgrid(x_, y_, indexing='ij')
                fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
                ax.set_xlim(0., x_lim)
                ax.set_ylim(0., y_lim)
                ax.set_xlabel('$x$ in $\mathrm{mm}$', fontsize=12)
                ax.set_ylabel('$y$ in $\mathrm{mm}$', fontsize=12)
                ax.set_title("$\sigma_{" + str(i + 1) + str(j + 1) + str(k + 1) + "}$ in $\mathrm{N/mm}$")
                lim = max(abs(T3[i][j][k].min()), abs(T3[i][j][k].max()))
                norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
                im = plt.pcolormesh(X, Y, T3[i][j][k], cmap='seismic', shading='gouraud', norm=norm)
                fig.colorbar(im, ax=ax)
                plt.savefig('T' + str(i+1) + str(j+1) + str(k+1) + '.png', dpi=300)
                
def save_traction_vector_gradient(traction_vector, normal):
    Space = traction_vector.function_space().spaces[0]
    x_, y_ = Space.local_mesh()
    X, Y = np.meshgrid(x_, y_, indexing='ij')
    for i in range(len(Space.bases)):
        x_lim = Space.bases[0].domain[1]
        y_lim = Space.bases[1].domain[1]
        x_, y_ = Space.local_mesh(uniform=True)
        X, Y = np.meshgrid(x_, y_, indexing='ij')
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        ax.set_xlim(0., x_lim)
        ax.set_ylim(0., y_lim)
        ax.set_xlabel('$x$ in $\mathrm{mm}$', fontsize=12)
        ax.set_ylabel('$y$ in $\mathrm{mm}$', fontsize=12)
        ax.set_title("$t_{" + str(i + 1) + "}$ in $\mathrm{MPa}$")
        lim = max(abs(traction_vector[i].min()), abs(traction_vector[i].max()))
        norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
        im = plt.pcolormesh(X, Y, traction_vector[i], cmap='seismic', shading='gouraud', norm=norm)
        fig.colorbar(im, ax=ax)
        if normal == (0, 1) or normal ==(0, -1):
            plt.savefig('t' + str(i+1) + 'ObenUnten.png', dpi=300)
        elif normal ==(1, 0) or normal ==(-1, 0):
            plt.savefig('t' + str(i+1) + 'RechtsLinks.png', dpi=300)