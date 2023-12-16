# -*- coding: utf-8 -*-
"""
Fluids Final - Rossby Waves
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import signal


Lx = 1024e3 # km, size of domain
Ly = 1024e3 # km
dx = 8e3 # km, matching grid size to domain
dy = 8e3 # km
Rd = 3 * 100e3 # km, radius of deformation small enough to see effects propagate
n_iter = 1000
skip = 200
K = 10000 # Eddy diffusivity, m2/sec
beta = 0.7 * 10**-9 # Rossby parameter, 1/ms, 
dt = 1500 # Seconds, will tentatively be changed
jason = -np.inf#(dx / (2 * dt))**2
# www.ncl.ucar.edu/Document/Functions/Contributed/beta_dfdy_rossby.shtml


def calc_vel(psi, dx, dy):
    """
    Takes our periodic gradient in x, y to obtain u, v
    
    Need a few extra lines to ensure periodic BCs
    """
    size = psi.shape
    # Create a replica PSI with extra ghost points
    psi_periodic = np.zeros((size[0] + 4, size[1] + 4))
    psi_periodic[2:-2, 2:-2] = psi
    # Left ghost points
    psi_periodic[2:-2, :2] = psi[:, -2:]
    # Right ghost points
    psi_periodic[2:-2, -2:] = psi[:, :2]
    # Top and botton now
    psi_periodic[:2, 2:-2] = psi[-2:, :]
    psi_periodic[-2:, 2:-2] = psi[:2, :]
    
    v, u = np.gradient(psi_periodic, dx, dy)
    # Delete the extra u and v components
    return u[2:-2, 2:-2].T, -v[2:-2, 2:-2].T


def calc_vel_f(psi_f, freqs):
    """
    Calculates the fourier transforms of u, v
    """
    u_f = 1j * freqs[1] * psi_f
    v_f = -1j * freqs[0] * psi_f
    return u_f, v_f


def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Lowpass filter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def initiate_vars(Lx, Ly, dx, dy, dt, Rd):
    """
    Initiate our streamfunction, and calculate our initial u, v, potential
    vorticity, and wavenumbers
    """
    
    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    # Get fourier freqs and stack along appropriate axis
    freqs = np.fft.fftfreq(Nx, dx)
    freqs_x, freqs_y = np.meshgrid(freqs, freqs)
    
    np.random.seed(234)
    #psi = np.random.randn(Ny, Nx)
    # Gaussian start to see wave?
    k1d = signal.windows.gaussian(Nx, std=Nx//5).reshape(Nx, 1)
    l1d = signal.windows.gaussian(Ny, std=Ny//5).reshape(Ny, 1)
    #psi = np.outer(l1d, k1d)
    q = np.random.normal(0, 1e-4, (Nx, Ny))
    q_f = np.fft.fft2(q)
    psi = np.abs(np.fft.ifft2(-q_f /\
                              (freqs_x**2 + freqs_y**2 + (1 / Rd)**2)))
    
    # x = np.vstack([np.arange(0, Nx, dx)] * Nx)
    # y = np.vstack([np.arange(0, Nx, dx)] * Ny)
    
    u, v = calc_vel(psi, dx, dy)
    
    mag = u**2 + v**2
    u *= (abs(mag) > jason).astype(int)
    v *= (abs(mag) > jason).astype(int)
    
    # Calculate potential vorticity
    psi_f = np.fft.fft2(psi)
    q_f = -(freqs_x**2 + freqs_y**2 + (1 / Rd)**2) * psi_f
    
    return psi, u, v, (freqs_x, freqs_y), q_f


def iterate_pot_vort(psi, u, v, q_f, freqs, old_vals=None):
    """
    Does one iteration loop to update potential vorticity
    """
    method = 'fft' # or fft/convolve. Not a decided thing so not a func param
    # Get relevant fourier transforms
    v_f = np.fft.fft2(v)
    u_f = np.fft.fft2(u)
    q = np.real(np.fft.ifft2(q_f))
    
    if old_vals is None:    
        # Forward Euler update equation
        if method == 'fft':
            q_f_new = dt * ((-K * (freqs[0]**2 + freqs[1]**2) * q_f) -\
                       1j * freqs[0] * np.fft.fft2(u * q) - 1j * freqs[1] * np.fft.fft2(v * q) -\
                           v_f * beta) + q_f
        elif method == 'convolve':
            q_f_new = dt * ((-K * (freqs[0]**2 + freqs[1]**2) * q_f -\
                             1j * freqs[0] * signal.fftconvolve(u_f, q_f, 
                                                                mode='same') -\
                                 1j * freqs[1] * signal.fftconvolve(v_f, q_f,
                                                                    mode='same') -\
                                         beta * v_f)) + q_f
    else:
        # Adams-bashworth for other steps
        q_f_old, u_f_old, v_f_old, u_old, v_old = old_vals
        if method == 'fft':
            q_old = np.real(np.fft.ifft2(q_f_old))
            
            q_f_new = 1.5 * dt * ((-K * (freqs[0]**2 + freqs[1]**2) * q_f) -\
                       1j * freqs[0] * np.fft.fft2(u * q) - 1j * freqs[1] * np.fft.fft2(v * q) -\
                           v_f * beta) -\
                      0.5 * dt * ((-K * (freqs[0]**2 + freqs[1]**2) * q_f_old) -\
                                     1j * freqs[0] * np.fft.fft2(u_old * q_old) - 1j *\
                                    freqs[1] * np.fft.fft2(v_old * q_old) -\
                                     v_f_old * beta) + q_f
        elif method == 'convolve':
            q_f_new = 1.5 * dt * ((-K * (freqs[0]**2 + freqs[1]**2) * q_f -\
                                   1j * freqs[0] * signal.fftconvolve(u_f, q_f, 
                                                                mode='same') -\
                                       1j * freqs[1] * signal.fftconvolve(v_f, q_f,
                                                               mode='same') -\
                                           beta * v_f)) -\
                    0.5 * dt * ((-K * (freqs[0]**2 + freqs[1]**2) * q_f_old -\
                                     1j * freqs[0] * signal.fftconvolve(u_f_old, q_f_old, 
                                                                        mode='same') -\
                                         1j * freqs[1] * signal.fftconvolve(v_f_old, q_f_old,
                                                                            mode='same') -\
                                                 beta * v_f_old)) + q_f
        
        
    # Lowpass filter to ensure signal doesn't explode
    q_f_new *= (abs(q_f_new) < 0.005).astype(int)
    #  q_new = np.abs(np.fft.ifft2(q_f_new))

    #psi = np.abs(np.fft.ifft2(-q_f_new /\
    #                          (freqs[0]**2 + freqs[1]**2 + (1 / Rd)**2)))
    psi_f = -q_f_new / (freqs[0]**2 + freqs[1]**2 + (1 / Rd)**2)
    psi = np.real(np.fft.ifft2(psi_f))
    
    #u, v = calc_vel(psi, dx, dy) 
    #u = u.T
    #v = v.T
    u_f_old, v_f_old = u_f, v_f
    u_old, v_old = u, v

    u_f, v_f = calc_vel_f(psi_f, freqs)
    u = np.real(np.fft.ifft2(u_f))
    v = np.real(np.fft.ifft2(v_f))
    
    mag = u**2 + v**2
    u *= (abs(mag) > jason).astype(int)
    v *= (abs(mag) > jason).astype(int)
    
    q_f_old = q_f
    q_f = q_f_new
    
    old_vals = [q_f_old, u_f_old, v_f_old, u_old, v_old]
    
    return psi, u, v, q_f, old_vals


def animate_all(psi, u, v, q_f, freqs, max_t, title='', step=300):
    """
    Animates VECTORS across simulation duration
    """
    plt.style.use('dark_background')
    # Initiate the axes correctly
    fig, ax = plt.subplots(figsize=(6,6))
    mag = u**2 + v**2
    field = ax.quiver(u, v, mag, cmap=plt.cm.jet)
    ax.tick_params(left = False, right = False , labelleft = False, 
                labelbottom = False, bottom = False)
    ax.set_title(title)

    def update(frame, psi, u, v, q_f, freqs, q_f_old):
        psi, u, v, q_f, q_f_old = iterate_pot_vort(psi, u, v, q_f,
                                                   freqs, None)
        
        mag = u**2 + v**2
        field.set_UVC(u, v, mag)
        return field,

    ani = FuncAnimation(fig, update, frames = np.arange(0, n_iter), interval=step,
                        fargs=(psi, u, v, q_f, freqs, None), blit=False)
    return ani


def animate_psi(psi, u, v, q_f, freqs, n_iter, title='', step=300):
    """
    Animates VECTORS across simulation duration
    """
    plt.style.use('dark_background')
    # Initiate the axes correctly
    fig, ax = plt.subplots(figsize=(6,6))
    field = plt.imshow(psi, cmap=plt.cm.jet)
    ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
    ax.set_title(title)

    def update(frame, psi, u, v, q_f, freqs, q_f_old):
        psi, u, v, q_f, q_f_old = iterate_pot_vort(psi, u, v, q_f,
                                                   freqs, q_f_old)
        
        field.set_data(psi)
        return field,

    ani = FuncAnimation(fig, update, frames = np.arange(0, n_iter), interval=step,
                        fargs=(psi, u, v, q_f, freqs, None), blit=False)
    return ani


def plot_speeds(mag_test, scale='log'):
    """
    Plots mean and maximum wind speeds over simulation
     """
    plt.figure(1)
    plt.style.use('default')
    plt.title('Wind Speeds in Simulation', color='white')
    if scale == 'log':
        plt.semilogy(np.arange(0, n_iter), mag_test[:, 1], label='Maximum Speed')
        plt.semilogy(np.arange(0, n_iter), mag_test[:, 0], label='Mean Speed')
        plt.ylabel('log(m/s)', color='white')
    else:
        plt.plot(np.arange(0, n_iter), mag_test[:, 1], label='Maximum Speed')
        plt.plot(np.arange(0, n_iter), mag_test[:, 0], label='Mean Speed')
        plt.ylabel('m/s', color='white')
    plt.xlabel('Iterations', color='white')
    
    plt.legend()
    ax = plt.gca()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.grid()


def plot_pos(pos_test):
    """
    Plots the particle position over time
    """
    plt.figure(2)
    plt.style.use('default')
    plt.plot(pos_test[:, 0], pos_test[:, 1])
    plt.xlabel('X (Periodic)')
    plt.ylabel('Y (Periodic)')
    ax = plt.gca()
    #ax.tick_params(axis='x', colors='black')
    #ax.tick_params(axis='y', colors='black')
    plt.grid()
    plt.title('Pathline of Particle')


def main():
    global ani_psi
    global ani_vel
    psi, u, v, freqs, q_f = initiate_vars(Lx, Ly, dx, dy, dt, Rd)
    old_vals = None
    
    # ani = animate_all(psi, u, v, q_f, freqs, n_iter, 'Meow', step=300)
    # anim = animate_psi(psi, u, v, q_f, freqs, n_iter, 'Meow', step=500)
    
    saved = [None for _ in range(n_iter)]
    saved_u = [None for _ in range(n_iter)]
    saved_v = [None for _ in range(n_iter)]
    # TESTING IF EXPLODES OR STABLE
    mag_test = np.zeros((n_iter, 2))
    pos_test = np.zeros((n_iter, 2))
    pos = np.array([32, 32], dtype=float)
    
    n = 0
    while n < n_iter:
        saved[n] = psi
        saved_u[n] = u[::2, ::4]
        saved_v[n] = v[::2, ::4]
        # Determining pathline of particle
        pos_test[n] = pos
        u_pos = u[int(pos[0]) % 64, int(pos[1]) % 64]
        v_pos = v[int(pos[0]) % 64, int(pos[1]) % 64]
        
        pos += np.array((dt * u_pos, dt * v_pos))
        # TESTING IF EXPLODES OR STABLE
        mag = (u**2 + v**2)**0.5
        mag_test[n] = mag.mean(), mag.max()
        
        
        psi, u, v, q_f, old_vals = iterate_pot_vort(psi, u, v, q_f, 
                                                   freqs, old_vals)
        
        n += 1
        print(f'\rLoading {(100 * n / n_iter):.2f}%', end='\r')
    
    # Debugging PLots
    # plot_speeds(mag_test, 'linear')
    # plot_pos(pos_test)
    
    saved = saved[skip:]
    saved_u = saved_u[skip:]
    saved_v = saved_v[skip:]

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.style.use('dark_background')
    field = plt.imshow(saved[0], cmap=plt.cm.jet)
    ax.set_title('Streamfunction')
    count = plt.text(4, 4, s=f'Iteration {0}', color='Black', size=15)
    def animate(frame, saved):
        field.set_data(saved[frame])
        count.set_text(f'Iteration {frame}')
        return field, count,
    ani_psi = FuncAnimation(fig, animate, frames = np.arange(0, n_iter - skip),
                        fargs=(saved,), blit=False, interval=50)
    
    # ani_psi.save("C:\\Users\\Aakas\\Downloads\\rossby_psi.gif",
    #               dpi=300, writer=PillowWriter(fps=25))
    
    
    fig_v, ax_v = plt.subplots(figsize=(8, 8))
    plt.style.use('dark_background')
    mag = saved_u[0]**2 + saved_v[0]**2
    field_v = ax_v.quiver(saved_u[0], saved_v[0], mag, cmap=plt.cm.jet)
    ax_v.tick_params(left = False, right = False , labelleft = False, 
                labelbottom = False, bottom = False)
    ax_v.set_title('Vectors')
    count_v = plt.text(2, 60, s=f'Iteration {0}', color='White', size=15)
    def animate(frame, saved_u, saved_v):        
        mag = saved_u[frame]**2 + saved_v[frame]**2
        field_v.set_UVC(saved_u[frame], saved_v[frame], 1.5 * mag)
        count_v.set_text(f'Iteration {frame}')
        return field_v, count_v,
    ani_vel = FuncAnimation(fig_v, animate, frames = np.arange(0, n_iter - skip),
                        fargs=(saved_u, saved_v), blit=False, interval=50)
    
    # ani_vel.save("C:\\Users\\Aakas\\Downloads\\rossby_vel.gif",
    #               dpi=300, writer=PillowWriter(fps=25))
    

if __name__ == '__main__':
    main()
