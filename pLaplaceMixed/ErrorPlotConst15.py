import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from matplotlib.ticker import MaxNLocator
from ConvergenceTriangles import draw_convergence_triangles, style_legend
from DataSym import *


mpl.rcParams[ 'text.usetex' ] = True
#mpl.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.sans-serif'] = 'cm'
#mpl.rcParams[ 'mathtext.fontset' ] = 'stixsans'
mpl.rcParams[ 'savefig.dpi' ] = 300
mpl.rcParams[ 'savefig.format' ] = 'pdf'
#mpl.rcParams[ 'savefig.transparent' ] = True
plt.rcParams["hatch.linewidth"] = 4
plt.style.use('bmh')

with open('./Const15/errors.pkl', 'rb') as datei:
    dictionary = pkl.load(datei) 

fig, ax = plt.subplots(1,2, figsize = ( 14, 6 ) )
  

#ax[0].legend(fontsize=12)

ax1 = ax[ 1 ].twinx() 
ax[ 1 ].set_zorder(2)
ax1.set_zorder(1)
ax[ 1 ].patch.set_visible(False)
ax1.patch.set_visible(True)
ax1.patch.set_alpha(1.0)
ax1.set_facecolor("#eeeeee")
ax1.tick_params(axis='both', labelsize=20) 

ax[ 1 ].loglog( dictionary['th'], dictionary['L2'], label = r'$\|v_h^\tau-\mathrm{I}_\tau^0v\|_{L^\infty(I;L^2(\Sigma))}$', c = 'tab:purple', marker= 'o', lw = 1.25,\
                                        markersize = 10.0, markeredgewidth = 1.0, markerfacecolor = 'tab:purple', markeredgecolor = 'k', )
ax[ 1 ].loglog( dictionary['th'], dictionary['H1'], label = r'$\|\mathbf{f}(\cdot,\nabla v_h^\tau)-\mathbf{f}(\cdot,\nabla \mathrm{I}_\tau^0v)\|_{I\times\Sigma}$', c = 'tab:blue', marker= 's', lw = 1.25,\
                                        markersize = 10.0, markeredgewidth = 1.0, markerfacecolor = 'tab:blue', markeredgecolor = 'k', )
Gamma_plot = np.sqrt( dictionary['Gamma'] )
ax[ 1 ].loglog( dictionary['th'], Gamma_plot, label = r'$\|(\varphi_{\vert \nabla v\vert})^*(\cdot,\vert \Gamma^\tau-\mathrm{I}_\tau^0\Gamma\vert)\|_{1,I\times\Sigma}^{1/2}$', c = 'tab:green', marker= '^', lw = 1.25,\
                                        markersize = 10.0, markeredgewidth = 1.0, markerfacecolor = 'tab:green', markeredgecolor = 'k', )
ax[ 1 ].set_axisbelow( True )
ax[ 1 ].set_facecolor("#eeeeee")
ax[ 1 ].grid( True, which = 'major', ls = '-', color = '0.75' )
ax[ 1 ].grid( True, which = 'minor', ls = ':', color = '0.75' ) 
ax[ 1 ].set_ylabel( r'Errors', fontsize = 20 )
ax[ 1 ].set_xlabel( r'$\tau+h$', fontsize = 20 )

draw_convergence_triangles(ax[ 1 ], dictionary['th'], [
    { 'values' : dictionary['L2'], 'color' : 'tab:purple' },
    { 'values' : dictionary['H1'], 'color' : 'tab:blue' },
    { 'values' : Gamma_plot, 'color' : 'tab:green' },
])

ax1.semilogx( dictionary['th'], dictionary['It'], label = r'Number of Picard-Iterations', c = 'tab:red', marker= '*', lw = 1.25,\
             markersize = 14.0, markeredgewidth = 1.0, markerfacecolor = 'tab:red', markeredgecolor = 'k', zorder = 0.5 )
ax1.set_yticks( range( 1, np.amax(dictionary['It']) +1 ) )
ax1.set_axisbelow( True )
ax1.set_facecolor("#eeeeee") 

ax1.set_ylabel( r'Number of Picard-Iterations', fontsize = 20, color = 'tab:red' )
ax1.spines['right'].set_color('tab:red') 
for axis in ['top','bottom','left']:
    ax1.spines[axis].set_linewidth(1)
    ax1.spines[axis].set_color('k')
ax1.yaxis.label.set_color('tab:red')
ax1.tick_params(axis='y', colors='tab:red')
ax1.set_xlabel( r'$\tau+h$', fontsize = 20 )

style_legend(ax[ 1 ].legend(fontsize=16, loc='lower right'))

ax[ 1 ].tick_params(axis='both', labelsize=20)  


plt.style.use('bmh')
#mesh = IntervalMesh( 32, -1.0, 1.0 )
ps = [  1.5, 1.5 ]
ts = [ -1.0, -0.5, 0.0 ]
ue     = Ue( ps, ts )
alphae = Alphae( ps, ts )
re = Re( ps, ts ) 
X = np.linspace( -1.0, 1.0, 1000 )
U = [ ue( xi ) for xi in X ]
A = [ alphae( xi ) for xi in X ]
R = [ re( xi ) for xi in X ]  
ax[ 0 ].plot(X,A,c='tab:purple',lw=1.5,label=r'$\alpha$')
ax[ 0 ].plot(X,U,c='tab:blue',lw=1.5, label=r'$v$')
ax[ 0 ].set_xlabel(r'$x$', fontsize = 20 )
ax[ 0 ].set_ylabel(r'$v(x),\alpha,\smash{v_{h_i}^{\tau_i}(L,x)}$, $i=1,\ldots,9$', fontsize = 20 )


for n in range( 1, 10 ):
    plt.style.use('bmh')
    # Create initial guess
    mesh = IntervalMesh( 2 ** n, -1.0, 1.0 )
    V    = FunctionSpace( mesh, "P", 1 )
    vL   = Function(V, "./Const15/v" + str( n ) + ".xml")
    X0   = mesh.coordinates()
    Y0   = [ vL( xi ) for xi in X0 ]
    if n < 9:
        ax[ 0 ].plot(X0,Y0,c='tab:blue',ls='--',lw=1.5,alpha=0.45+n*0.05, marker='.', markersize=9.0/n)
    else:
        ax[ 0 ].plot(X0,Y0,c='tab:blue',ls='--',lw=1.5,alpha=0.45+n*0.05, marker='.', markersize=9.0/n,label=r'$\smash{v_{h_i}^{\tau_i}(L)}$, $i=1,\ldots,9$')

ax[ 0 ].legend(fontsize=16.0,loc='best')

ax2 = ax[ 0 ].twinx()  
# Change axis (spine) color to red 
ax2.spines['right'].set_color('tab:red')
# Change tick color 
ax2.tick_params(axis='y', colors='tab:red')
# Change axis labels color 
ax2.yaxis.label.set_color('tab:red')
ax2.set_ylabel(r'$p(x)$', fontsize = 20 ) 
plt.style.use('bmh')
ax2.plot(X,R,c='tab:red',lw=1.5,ls='-')
ax2.set_ylim([1.0,3.0])
#ax2.scatter(X,R,c='tab:red',lw=1.0,s=0.5)

for axis in ['top','bottom','left']:
    ax2.spines[axis].set_linewidth(1)
    ax2.spines[axis].set_color('k')
ax[ 0 ].tick_params(axis='both', labelsize=20)  
ax2.tick_params(axis='both', labelsize=20)  
 
 
fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.94, wspace=0.38)
plt.show()
