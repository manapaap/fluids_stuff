# -*- coding: utf-8 -*-
"""
Point vortex simuations for ATMOS 505
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from os import chdir


chdir('C:/Users/Aakas/Documents/Grad School/ATMOS 505/vortices')


# I ain't explaining this one
dt = 0.2
# Controls simulation duration
max_t = 20
# Controls size of simulation box and relative coords
size = 1000
# Controls density of arrows
res = 40
# Meow
np.random.seed(6346)


class Vortex:
    def __init__(self, x, y, gamma):
        """
        Initiate our vortex
        """
        self.loc = np.array([x, y], dtype=float)
        self.gamma = gamma
        self.vel = np.zeros(2)

    def get_loc(self):
        """
        Returns the current location of the vortex in a tuple
        """
        return self.loc

    def get_gamma(self):
        """
        Returns gamma of our vortex
        """
        return self.gamma

    def set_vel(self, vel):
        """
        Manually set velocity for debugging and corner
        """
        self.vel = vel

    def speed(self, dist):
        """
        Outputs the speed of vortex circulation at a given point
        """
        with np.errstate(divide='ignore'):
            speed = self.gamma / (2 * np.pi * dist)
        return speed

    def dist_bet(self, other):
        """
        Calculates the distance between this vortex and another one
        """
        loc1 = self.loc
        loc2 = other.get_loc()
        # Euclidian distance
        dist = (loc1 - loc2)**2
        dist = np.sqrt(np.sum(dist))
        return dist

    def perp(self, other):
        """
        Defines a unit vector that defines the perpendicular direction
        of the second vortex pointing away from the first
        
        Takes into account gamma so the vector points in the same direction
        for two opposite gamma's and opposite for different gamma
        """
        vect = other.get_loc() - self.get_loc()
        # Get perpendicular vector: invert entries and flip sign of second
        # (Might have to swap sign of first to ensure gamma consistency)
        vect_new = vect[::-1]
        vect_new[1] *= -1
        vect_new = vect_new / np.linalg.norm(vect_new)
        # Gamma consistency
        return np.sign(-1 * other.get_gamma() * self.gamma) * vect_new

    def update_vel(self, other, mode='meow'):
        """
        Updates the velocity of self with respect to the other vortex
        
        If mode='vect', simply returns the calculated velocity vector
        and doesn't update it
        """
        perp_vect = self.perp(other)
        dist = self.dist_bet(other)
        vel_vect = other.speed(dist) * perp_vect
        if mode == 'vect':
            return vel_vect
        else:
            self.vel = vel_vect

    def update_pos(self, dt):
        """
        Updates position of the vortex depending on the velocity of the
        vortex
        """
        self.loc += self.vel * dt

    def field(self, size, every=res):
        """
        Returns the direction of field around the vortex
        """
        loc_approx = np.rint(self.get_loc())
        x_indices = np.arange(0, size, every)
        y_indices = np.arange(0, size, every)
        x_indices, y_indices = np.meshgrid(x_indices, y_indices)
        # Create our vector: ponting direction - origin of point
        x_dir = x_indices - loc_approx[0]
        y_dir = y_indices - loc_approx[1]
        return x_dir, y_dir

    def vel_field(self, size):
        """
        Inverts the vector field correctly to get the velocity field
        """
        x_dir, y_dir = self.field(size)
        x_dir, y_dir = y_dir, -x_dir
        # Signs from gamma
        x_dir *= np.sign(self.gamma)
        y_dir *= np.sign(self.gamma)
        # Normalize by magnitude
        mag = x_dir**2 + y_dir**2
        with np.errstate(divide='ignore', invalid='ignore'):
            x_dir /= mag
            y_dir /= mag
        # Multiply speed information
        speed = self.speed_field(size)
        x_dir *= speed
        y_dir *= speed

        return x_dir, y_dir

    def speed_field(self, size):
        """
        Creates the scalor field of speeds around the vortex
        """
        x_dir, y_dir = self.field(size)
        # Distance matrix generated using the velocity matrix
        dist_mat = np.sqrt(x_dir**2 + y_dir**2)
        # Create speeds from this distance matrix
        speed_mat = self.speed(dist_mat)
        # Remove the infinite value by the 1/r
        speed_mat[speed_mat == np.inf] = speed_mat[speed_mat != np.inf].max()
        return speed_mat


def plot_velocity(x_field, y_field):
    """
    Plots a quiver field for the velocity field
    """
    # Remove the extreme values that occur near the center of the point vortex
    top_x = np.percentile(x_field, 95)
    top_y = np.percentile(y_field, 95)
    bot_x = np.percentile(x_field, 5)
    bot_y = np.percentile(y_field, 5)
    # Replace high extremes
    x_field[x_field > top_x] = top_x
    y_field[y_field > top_y] = top_y
    # Low extremes
    x_field[x_field < bot_x] = bot_x
    y_field[y_field < bot_y] = bot_y
    # Get the magnitude squared for the magnitude
    mag = x_field**2 + y_field**2
    # Actually plot
    plt.figure(np.random.randint(0, 1000))
    plt.quiver(x_field, y_field, mag, cmap=plt.cm.jet)


def remove_extremes(x_field, y_field, perc=5):
    """
    Removes the extreme magnitudes of the saved vector animation
    in anticipation of animation
    """
    for n, (x, y) in enumerate(zip(x_field, y_field)):
        # Remove the extreme values that occur near the center of the point vortex
        top_x = np.percentile(x, 100-perc)
        top_y = np.percentile(y, 100-perc)
        bot_x = np.percentile(x, perc)
        bot_y = np.percentile(y, perc)
        # Replace high extremes
        x[x > top_x] = top_x
        y[y > top_y] = top_y
        # Low extremes
        x[x < bot_x] = bot_x
        y[y < bot_y] = bot_y

    return x_field, y_field


def animate_all(x_field, y_field, title, scale=3):
    """
    Animates across the whole field and the iteration duration
    """
    plt.style.use('dark_background')
    # Initiate the axes correctly
    fig, ax = plt.subplots(figsize=(6,6))
    mag = x_field[0]**2 + y_field[0]**2
    field = ax.quiver(x_field[0], y_field[0], mag, cmap=plt.cm.jet)
    ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
    ax.set_title(title)

    def update(frame, x_field, y_field):
        mag = x_field[frame]**2 + y_field[frame]**2
        field.set_UVC(x_field[frame], y_field[frame],
                       scale * mag)
        return field,

    ani = FuncAnimation(fig, update, frames = np.arange(0, len(x_field)),
                        fargs=(x_field, y_field), blit=False)
    return ani


def leapfrog(pos_1, pos_2, gamma, perturb=False, val=1, light=False):
    """
    Creates 4 point vortices and lets them interact
    """
    midpoint = size // 2
    if perturb:
        vortices = [Vortex(pos_1 + np.random.randint(-val, val),
                           pos_2+ np.random.randint(-val, val), -gamma),
                Vortex(pos_1+ np.random.randint(-val, val),
                       midpoint + pos_2 + np.random.randint(-val, val), gamma),
                Vortex(midpoint + pos_1 + np.random.randint(-val, val),
                       pos_2 + np.random.randint(-val, val), -gamma),
                Vortex(midpoint + pos_1 + np.random.randint(-val, val),
                       midpoint + pos_2 + np.random.randint(-val, val), -gamma)]
    else:
        vortices = [Vortex(pos_1, pos_2, -gamma),
                Vortex(pos_1, midpoint + pos_2, gamma),
                Vortex(midpoint + pos_1, pos_2, -gamma),
                Vortex(midpoint + pos_1, midpoint + pos_2, gamma)]

    t = 0
    if light:
        x_field = 0
        y_field = 0
    else:
        x_field = [np.zeros([size, size]) for _ in range(int(max_t / dt))]
        y_field = [np.zeros([size, size]) for _ in range(int(max_t / dt))]

    past_pos = [np.zeros([int(max_t / dt), 2]) for _ in vortices]

    for i in range(int(max_t /dt)):
        # Capture the velocity field
        x_f = np.zeros((size // res, size // res))
        y_f = np.zeros((size // res, size // res))
        if light:
            x_field = 0
            y_field = 0
        else:
            for vortex in vortices:
                x_r, y_r = vortex.vel_field(size)
                x_f += x_r
                y_f += y_r
                # Anim snapshot
                x_field[i] = x_f
                y_field[i] = y_f
        # For position too
        for n, vortex in enumerate(vortices):
            past_pos[n][i] = vortex.get_loc()
        # Calculate velocity of each
        for vortex in vortices:
            vel_vect = np.zeros(2)
            for rel_vort in vortices:
                # Skip the calculation if we reference ourselves
                if vortex is rel_vort:
                    continue
                else:
                    vel_vect += vortex.update_vel(rel_vort, mode='vect')
            vortex.set_vel(vel_vect)
        # Update positions
        for vortex in vortices:
            vortex.update_pos(dt)
        t += dt

    # Remove the extremes in anticipation of animation
    if not light:
        x_field, y_field = remove_extremes(x_field, y_field)
    return x_field, y_field, past_pos


def corner(pos, gamma, vel=np.array([0, 0])):
    """
    Bounces a point vortex off a "corner" at the edge of the grid
    
    8 Ghost vortices to simulate this
    """
    real = Vortex(*pos, gamma)
    real.set_vel(vel)

    t = 0
    x_field = [np.zeros([size, size]) for _ in range(int(max_t / dt))]
    y_field = [np.zeros([size, size]) for _ in range(int(max_t / dt))]

    past_pos = np.zeros([int(max_t / dt), 2])

    for i in range(int(max_t /dt)):
        # Create 4 ghost vortices to guide us
        x, y = real.get_loc()
        ghost_1 = Vortex(-x, y, -gamma)
        ghost_2 = Vortex(-x, -y, gamma)
        ghost_3 = Vortex(x, -y, -gamma)
        ghost_4 = Vortex(size + x, -y, gamma)
        ghost_5 = Vortex(size + x, y, -gamma)
        ghost_6 = Vortex(size + x, size + y, gamma)
        ghost_7 = Vortex(x, size + y, -gamma)
        ghost_8 = Vortex(-x, size + y, gamma)
        ghosts = [ghost_1, ghost_2, ghost_3, ghost_4,
                  ghost_5, ghost_6, ghost_7, ghost_8]
        # Calculate the vector field
        x_r, y_r = real.vel_field(size)
        for ghost in ghosts:
            x_g, y_g = ghost.vel_field(size)
            x_r += x_g
            y_r += y_g
        # Save a snapshot for anim'
        x_field[i] = x_r
        y_field[i] = y_r
        past_pos[i] = real.get_loc()
        # Update velocity of real based on ghosts
        vel_vect = np.zeros(2, dtype=float)
        for ghost in ghosts:
            vel_vect += real.update_vel(ghost, mode='vect')
        real.set_vel(vel_vect)
        # Use this to update pos and time
        real.update_pos(dt)
        t += dt

    # Remove the extremes in anticipation of animation
    x_field, y_field = remove_extremes(x_field, y_field)
    return x_field, y_field, past_pos


def dipole(pos_1, pos_2, gamma_1, gamma_2=500000):
    """
    Runs all the code for counter-rotating vortices
    
    Returns two lists of the velocity fields
    """
    vortex_1 = Vortex(*pos_1, gamma_1)
    vortex_2 = Vortex(*pos_2, gamma_2)

    # Let's actually do the looping of these guys in a circle
    t = 0
    x_field = [np.zeros([size, size]) for _ in range(int(max_t / dt))]
    y_field = [np.zeros([size, size]) for _ in range(int(max_t / dt))]

    past_pos_1 = np.zeros([int(max_t / dt), 2])
    past_pos_2 = np.zeros([int(max_t / dt), 2])

    for i in range(int(max_t / dt)):
        x_1, y_1 = vortex_1.vel_field(size)
        x_2, y_2 = vortex_2.vel_field(size)
        # Calculate the updated fields, removing weird nan
        field_x = np.nan_to_num(x_1 + x_2, 0)
        field_y = np.nan_to_num(y_1 + y_2, 0)
        # Save the field values for animation
        x_field[i] = field_x
        y_field[i] = field_y
        # Save the position values for animation
        past_pos_1[i] = vortex_1.get_loc()
        past_pos_2[i] = vortex_2.get_loc()
        # Calculate the velocities for advection
        vortex_1.update_vel(vortex_2)
        vortex_2.update_vel(vortex_1)
        # Actually update positions
        vortex_1.update_pos(dt)
        vortex_2.update_pos(dt)
        # Update time
        t += dt

    # Remove the extremes in anticipation of animation
    x_field, y_field = remove_extremes(x_field, y_field)
    return x_field, y_field, [past_pos_1, past_pos_2]


def plot_dipole(pos_1, pos_2, pos_3, pos_4):
    """
    Plots the trajectories of the dipoles and corner
    """
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    fig.suptitle('Dipole Trajectories')

    axs[0, 0].set_title('Co-Rotating Vortices')
    axs[0, 0].scatter(pos_1[0][:, 0], pos_1[0][:, 1], label='Vortex 1')
    axs[0, 0].scatter(pos_1[1][:, 0], pos_1[1][:, 1], label='Vortex 2')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(0, 1000)
    axs[0, 0].set_xlim(0, 1000)
    axs[0, 0].grid()

    axs[0, 1].set_title('Co-Moving Vortices')
    axs[0, 1].scatter(pos_2[0][:, 0], pos_2[0][:, 1], label='Vortex 1')
    axs[0, 1].scatter(pos_2[1][:, 0], pos_2[1][:, 1], label='Vortex 2')
    axs[0, 1].set_ylim(0, 1000)
    axs[0, 1].grid()

    axs[1, 0].set_title('Imbalanced Vortices')
    axs[1, 0].scatter(pos_3[0][:, 0], pos_3[0][:, 1], label='Small Vortex')
    axs[1, 0].scatter(pos_3[1][:, 0], pos_3[1][:, 1], label='Big Vortex')
    axs[1, 0].set_ylim(0, 1000)
    axs[1, 0].set_xlim(0, 1000)
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].set_title('Bounce off Walls')
    axs[1, 1].scatter(pos_4[:, 0], pos_4[:, 1])
    axs[1, 1].set_ylim(0, 1000)
    axs[1, 1].set_xlim(0, 1000)

    plt.grid()
    plt.tight_layout()


def plot_leapfrog(pos, pos_per):
    """
    Plots the leapfrog configuration of 4 vortices, both perturbed and not
    """
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    fig.suptitle('Quadropole Trajectories')

    for vort in pos:
        axs[0].scatter(vort[:, 0], vort[:, 1], s=0.05)
    axs[0].set_title('Original')
    axs[0].grid()
    axs[0].set_xlim(0, 25000)

    for vort in pos_per:
        axs[1].scatter(vort[:, 0], vort[:, 1], s=0.05)
    axs[1].set_title('Perturbed')
    axs[1].grid()


def main():
    global ani, ani_2, ani_3, ani_4, ani_5, ani_6

    # Situation 1: Same Gamma
    x_field_1, y_field_1, pos_1 = dipole((350, 350), (650, 650), 
                                         400000, 400000)
    ani = animate_all(x_field_1, y_field_1, 'Counter-Rotating Vortices')
    # ani.save('co-rotate.gif', writer='imagemagick', fps=30)
    # Situation 2: Opposite Gamma
    x_field_2, y_field_2, pos_2 = dipole((100, 200), (100, 700), 
                                         -200000, 200000)
    ani_2 = animate_all(x_field_2, y_field_2, 'Co-moving Vortices')
    # ani_2.save('co-move.gif', writer='imagemagick', fps=30)
    # Situation 3: Imbalanced Gamma
    x_field_3, y_field_3, pos_3 = dipole((800, 800), (550, 550), 
                                         50000, -300000)
    ani_3 = animate_all(x_field_3, y_field_3, 'Imbalanced Vortices')
    # ani_3.save('imbalanced.gif', writer='imagemagick', fps=30)

    # Situation 4: Bounce off corner
    x_field_4, y_field_4, pos_4 = corner((200, 400), 400000)
    ani_4 = animate_all(x_field_4, y_field_4, 'Corner Bounce!', scale=750)
    # ani_4.save('box_bounce.gif', writer='imagemagick', fps=30)

    # Situation 5: Leap-Frog!
    x_field_5, y_field_5, pos_5 = leapfrog(250, 250, 200000, light=False)
    ani_5 = animate_all(x_field_5, y_field_5, 'Leapfrog', scale=3)
    # ani_5.save('leapfrog.gif', writer='imagemagick', fps=30)
    # Situation 6: Perturbed Leap-Frog
    x_field_6, y_field_6, pos_6 = leapfrog(250, 250, 200000,
                                           perturb=True, light=False)
    ani_6 = animate_all(x_field_6, y_field_6, 'Perturbed Leapfrog', scale=3)
    # ani_6.save('perturb.gif', writer='imagemagick', fps=30)

    # Plot trajectories
    plot_dipole(pos_1, pos_2, pos_3, pos_4)
    plot_leapfrog(pos_5, pos_6)


if __name__ == '__main__':
    main()
