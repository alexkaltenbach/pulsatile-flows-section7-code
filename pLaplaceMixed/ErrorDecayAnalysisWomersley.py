from fenics import *
import numpy as np
import os
import pickle

from DataWomersley import Alphae, DUe, Gammae, Ue
from PicardIterationLinear import PicardIterationLinear


set_log_level(30)


def error_decay_analysis(T, radius, nmax=12, l=1, omega=1.0, u0=Constant(0.0), output_dir=None):
    output_dir = f'Womersley{int(radius) if float(radius).is_integer() else radius}' if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)

    dictionary = {
        'th': [],
        'L2': [],
        'H1': [],
        'Gamma': [],
        'It': [],
        'L2_EOC': [],
        'H1_EOC': [],
        'Gamma_EOC': [],
    }

    for n in range(1, nmax):
        print('Step:', n)
        mesh = IntervalMesh(2 ** n, -radius, radius)
        num_steps = 2 ** n
        dt = T / num_steps
        alphae = Alphae(t=0.0, omega=omega, radius=radius, degree=6)

        snap_list, _, _, num_iter = PicardIterationLinear(
            mesh,
            T,
            num_steps,
            l=l,
            alpha=alphae,
            u0=u0,
            snapshots=True,
            info=True,
        )

        File('./' + output_dir + '/v' + str(n) + '.xml') << snap_list[-1][0]
        File('./' + output_dir + '/Gamma' + str(n) + '.xml') << snap_list[-1][1]

        L2_err = []
        H1_err = []
        Gamma_err = []
        T_quad_element = VectorElement(family='Quadrature', cell=mesh.ufl_cell(), degree=6, quad_scheme='default')
        T_quad_space = FunctionSpace(mesh, T_quad_element)

        for i, (vh, _) in enumerate(snap_list):
            ue = Ue(t=i * dt, omega=omega, radius=radius, degree=6)
            L2_err.append(assemble((ue - vh) * (ue - vh) * dx(domain=mesh)))

        for i, (vh, gammah) in enumerate(snap_list[1:], start=1):
            due = DUe(t=i * dt, omega=omega, radius=radius, degree=6)
            due_quad = interpolate(due, T_quad_space)
            gammae = Gammae(t=i * dt, omega=omega, radius=radius, degree=6)
            H1_err.append(dt * assemble(dot(due_quad - grad(vh), due_quad - grad(vh)) * dx(domain=mesh, scheme='default', degree=6)))
            Gamma_err.append(dt * (gammae(0.0) - gammah(0.0)) ** 2.0)

        dictionary['L2'].append(sqrt(np.amax(np.array(L2_err))))
        dictionary['H1'].append(sqrt(np.sum(H1_err)))
        dictionary['Gamma'].append(sqrt(np.sum(Gamma_err)))
        dictionary['It'].append(num_iter)
        dictionary['th'].append(dt + mesh.hmax())
        print('Picard-Iterations:', num_iter)

    print('L2-EOC:')
    for n in range(1, nmax - 1):
        L2_eoc = np.log(dictionary['L2'][n] / dictionary['L2'][n - 1]) / np.log(0.5)
        print(L2_eoc)
        dictionary['L2_EOC'].append(L2_eoc)

    print('H1-EOC:')
    for n in range(1, nmax - 1):
        H1_eoc = np.log(dictionary['H1'][n] / dictionary['H1'][n - 1]) / np.log(0.5)
        print(H1_eoc)
        dictionary['H1_EOC'].append(H1_eoc)

    print('Gamma-EOC:')
    for n in range(1, nmax - 1):
        Gamma_eoc = np.log(dictionary['Gamma'][n] / dictionary['Gamma'][n - 1]) / np.log(0.5)
        print(Gamma_eoc)
        dictionary['Gamma_EOC'].append(Gamma_eoc)

    with open('./' + output_dir + '/errors.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
