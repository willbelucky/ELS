import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def TDMAsolver(a, b, c, d):  # Thomas algorithm for solving a tridiagonal matrix
    nf = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]
    xc = bc
    xc[-1] = dc[-1] / bc[-1]
    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]
    return xc


def get_KI_prob(KI, T, rho, mu_x, mu_y, sig_x, sig_y, Nt, path_num):
    dx = (T * mu_x) / Nt
    dy = (T * mu_y) / Nt
    dt = T / Nt

    S1 = np.ones(path_num)
    S2 = np.ones(path_num)

    min_S1 = np.ones(path_num)
    min_S2 = np.ones(path_num)

    for i in range(Nt):
        np.random.seed()
        e1 = np.random.normal(size=path_num)
        time.sleep(0.1)
        np.random.seed()
        e2 = e1 * rho + np.random.normal(size=path_num) * np.sqrt((1 - rho) * rho)
        S1 = S1 + dx + sig_x * e1 * np.sqrt(dt)
        min_S1 = np.amin([S1, min_S1], axis=0)
        S2 = S2 + dy + sig_y * e2 * np.sqrt(dt)
        min_S2 = np.amin([S2, min_S2], axis=0)

    KI_prob = ((min_S1 <= KI) | (min_S2 <= KI)).sum() / path_num

    return KI_prob


def get_els_price(KI, T, F, coupon, pp, K, rho, S1_0, S2_0, mu_x, mu_y, sig_x, sig_y, Nt, Nx, Ny, dt, dx, dy, r):
    w = np.zeros((Nt + 1, Nx + 1, Ny + 1))  # KI doesn't occur
    v = np.zeros((Nt + 1, Nx + 1, Ny + 1))  # KI occurs
    # initial condition
    # NO BARRIER TOUCH (만기 이전에 베리어 친적 없음)
    for i in np.arange(Nx + 1):
        for j in np.arange(Ny + 1):
            w[pp * 0, i, j] = F * min(i * dx / S1_0, j * dy / S2_0)
    w[pp * 0, round(S1_0 * KI / dx):, round(S2_0 * KI / dy):] = F * (1 + coupon[5])
    # BARRIER TOUCH case (만기 이전에 베리어 친적 있음)
    for i in np.arange(Nx + 1):
        for j in np.arange(Ny + 1):
            v[pp * 0, i, j] = F * min(i * dx / S1_0, j * dy / S2_0)
    v[pp * 0, round(S1_0 * K[5] / dx):, round(S2_0 * K[5] / dy):] = F * (1 + coupon[5])
    # OSM stage 1 : w.r.t asset 1, tridiagonal matrix
    alpha = np.zeros(Nx + 1)
    beta = np.zeros(Nx + 1)
    gamma = np.zeros(Nx + 1)
    for i in np.arange(0, Nx + 1, 1):
        alpha[i] = -((sig_x * (i)) ** 2) / 2
        beta[i] = (1 / dt) + (sig_x * (i)) ** 2 + r * (i) + r / 2
        gamma[i] = -((sig_x * (i)) ** 2) / 2 - r * (i)
    Ax = np.zeros([Nx, Nx])
    for i in np.arange(Nx):
        for j in np.arange(Nx):
            if i == j:
                Ax[i, j] = beta[i + 1]
            elif i == j + 1:
                Ax[i, j] = alpha[i + 1]
            elif i == j - 1:
                Ax[i, j] = gamma[i + 1]
    Ax[0, 0] = Ax[0, 0] + 2 * alpha[1]
    Ax[0, 1] = Ax[0, 1] - alpha[1]
    Ax[-1, -1] = Ax[-1, -1] + 2 * gamma[Nx]
    Ax[-1, -2] = Ax[-1, -2] - gamma[Nx]
    # OSM stage 2 : w.r.t asset 2
    alpha2 = np.zeros(Ny + 1)
    beta2 = np.zeros(Ny + 1)
    gamma2 = np.zeros(Ny + 1)
    for j in np.arange(0, Ny + 1, 1):
        alpha2[j] = -0.5 * ((sig_y * (j)) ** 2)
        beta2[j] = (1 / dt) + (sig_y * (j)) ** 2 + r * (j) + r / 2
        gamma2[j] = -0.5 * ((sig_y * (j)) ** 2) - r * (j)
    Ay = np.zeros([Ny, Ny])
    for i in np.arange(Ny):
        for j in np.arange(Ny):
            if i == j:
                Ay[i, j] = beta2[i + 1]
            elif i == j + 1:
                Ay[i, j] = alpha2[i + 1]
            elif i == j - 1:
                Ay[i, j] = gamma2[i + 1]
    Ay[0, 0] = Ay[0, 0] + 2 * alpha2[1]
    Ay[0, 1] = Ay[0, 1] - alpha2[1]
    Ay[-1, -1] = Ay[-1, -1] + 2 * gamma2[Nx]
    Ay[-1, -2] = Ay[-1, -2] - gamma2[Nx]
    # Knock-In event case
    for k in np.arange(Nt):

        if k == pp * 1:
            v[k, round(S1_0 * K[4] / dx):, round(S2_0 * K[4] / dy):] = F * (1 + coupon[4])
        if k == pp * 2:
            v[k, round(S1_0 * K[3] / dx):, round(S2_0 * K[3] / dy):] = F * (1 + coupon[3])
        if k == pp * 3:
            v[k, round(S1_0 * K[2] / dx):, round(S2_0 * K[2] / dy):] = F * (1 + coupon[2])
        if k == pp * 4:
            v[k, round(S1_0 * K[1] / dx):, round(S2_0 * K[1] / dy):] = F * (1 + coupon[1])
        if k == pp * 5:
            v[k, round(S1_0 * K[0] / dx):, round(S2_0 * K[0] / dy):] = F * (1 + coupon[0])

        f = np.zeros((Nx + 1, Ny + 1))
        v_hat = v[k]
        g = np.zeros((Ny + 1, Ny + 1))

        for j in np.arange(1, Ny + 1, 1):
            for i in np.arange(1, Nx + 1, 1):
                f[i, j] = 0.5 * rho * sig_x * sig_y * (i) * dx * (j) * dy * (
                        v[k, i, j] - v[k, i, j - 1] - v[k, i - 1, j] + v[k, i - 1, j - 1]) / (4 * dx * dy) + v[
                              k, i - 1, j - 1] / dt
            v_hat[:Nx, j - 1] = TDMAsolver(Ax.diagonal(-1), Ax.diagonal(0), Ax.diagonal(1), f[1:, j])  # v[k+0.5]
        # Boundary condition for v_hat (or v[k+0.5])
        v_hat[0, :] = 2 * v_hat[1, :] - v_hat[2, :]
        v_hat[Nx, :] = 2 * v_hat[Nx - 1, :] - v_hat[Nx - 2, :]
        v_hat[:, 0] = 2 * v_hat[:, 1] - v_hat[:, 2]
        v_hat[:, Ny] = 2 * v_hat[:, Ny - 1] - v_hat[:, Ny - 2]

        for i in np.arange(1, Nx + 1, 1):
            for j in np.arange(1, Ny + 1, 1):
                g[i, j] = 0.5 * rho * sig_x * sig_y * (i) * dx * (j) * dy * (
                        v_hat[i, j] - v_hat[i, j - 1] - v_hat[i - 1, j] + v_hat[i - 1, j - 1]) / (4 * dx * dy) + v_hat[
                              i - 1, j - 1] / dt
            v[k + 1, i - 1, :Ny] = TDMAsolver(Ay.diagonal(-1), Ay.diagonal(0), Ay.diagonal(1), g[i, 1:])
            # boundary condition for v[k+1]
        v[k + 1, 0, :] = 2 * v[k + 1, 1, :] - v[k + 1, 2, :]
        v[k + 1, Nx, :] = 2 * v[k + 1, Nx - 1, :] - v[k + 1, Nx - 2, :]
        v[k + 1, :, 0] = 2 * v[k + 1, :, 1] - v[k + 1, :, 2]
        v[k + 1, :, Ny] = 2 * v[k + 1, :, Ny - 1] - v[k + 1, :, Ny - 2]
    V0 = v[Nt]
    # No Knock-In event case
    for k in np.arange(Nt):

        if k == pp * 1:
            w[k, round(S1_0 * K[4] / dx):, round(S2_0 * K[4] / dy):] = F * (1 + coupon[4])
        if k == pp * 2:
            w[k, round(S1_0 * K[3] / dx):, round(S2_0 * K[3] / dy):] = F * (1 + coupon[3])
        if k == pp * 3:
            w[k, round(S1_0 * K[2] / dx):, round(S2_0 * K[2] / dy):] = F * (1 + coupon[2])
        if k == pp * 4:
            w[k, round(S1_0 * K[1] / dx):, round(S2_0 * K[1] / dy):] = F * (1 + coupon[1])
        if k == pp * 5:
            w[k, round(S1_0 * K[0] / dx):, round(S2_0 * K[0] / dy):] = F * (1 + coupon[0])

        f = np.zeros((Nx + 1, Ny + 1))
        w_hat = w[k]
        g = np.zeros((Ny + 1, Ny + 1))
        for j in np.arange(1, Ny + 1, 1):
            for i in np.arange(1, Nx + 1, 1):
                f[i, j] = 0.5 * rho * sig_x * sig_y * (i) * dx * (j) * dy * (
                        w[k, i, j] - w[k, i, j - 1] - w[k, i - 1, j] + w[k, i - 1, j - 1]) / (4 * dx * dy) + w[
                              k, i - 1, j - 1] / dt
            w_hat[:Nx, j - 1] = TDMAsolver(Ax.diagonal(-1), Ax.diagonal(0), Ax.diagonal(1), f[1:, j])  # w[k+0.5]

        # Boundary condition for w_hat (or w[k+0.5])
        w_hat[0, :] = 2 * w_hat[1, :] - w_hat[2, :]
        w_hat[Nx, :] = 2 * w_hat[Nx - 1, :] - w_hat[Nx - 2, :]
        w_hat[:, 0] = 2 * w_hat[:, 1] - w_hat[:, 2]
        w_hat[:, Ny] = 2 * w_hat[:, Ny - 1] - w_hat[:, Ny - 2]

        for i in np.arange(1, Nx + 1, 1):
            for j in np.arange(1, Ny + 1, 1):
                g[i, j] = 0.5 * rho * sig_x * sig_y * (i) * dx * (j) * dy * (
                        w_hat[i, j] - w_hat[i, j - 1] - w_hat[i - 1, j] + w_hat[i - 1, j - 1]) / (4 * dx * dy) + w_hat[
                              i - 1, j - 1] / dt
            w[k + 1, i - 1, :Ny] = TDMAsolver(Ay.diagonal(-1), Ay.diagonal(0), Ay.diagonal(1), g[i, 1:])

            # boundary condition for w[k+1]
        w[k + 1, 0, :] = 2 * w[k + 1, 1, :] - w[k + 1, 2, :]
        w[k + 1, Nx, :] = 2 * w[k + 1, Nx - 1, :] - w[k + 1, Nx - 2, :]
        w[k + 1, :, 0] = 2 * w[k + 1, :, 1] - w[k + 1, :, 2]
        w[k + 1, :, Ny] = 2 * w[k + 1, :, Ny - 1] - w[k + 1, :, Ny - 2]

        w[k + 1, :round(S1_0 * KI / dx) + 1, :] = v[k + 1, :round(S1_0 * KI / dx) + 1, :]
        w[k + 1, :, :round(S2_0 * KI / dy) + 1] = v[k + 1, :, :round(S2_0 * KI / dy) + 1]
    W0 = w[Nt]

    # ELS price
    KI_prob = get_KI_prob(KI, T, rho, mu_x, mu_y, sig_x, sig_y, Nt, 1000)
    els_price = W0 * (1 - KI_prob) + V0 * KI_prob  # KI_prob = Prob[touch the KI barrier]

    return els_price


# %% ELS pricing

# parameters
S1_0 = 12976.8604
S2_0 = 2782.57
xmax = S1_0 * 2  # asset 1 (HSCE) maximum value
ymax = S2_0 * 2  # asset 2 (EURO) maximum value

xmin = 0  # asset 1 minimum value
ymin = 0  # asset 2 minimum value

mu_x = -0.03856  # expected return of HSCE
mu_y = 0.0259  # expected return of EURO

sig_x = 0.1749  # volatility of asset 1
sig_y = 0.1560  # volatility of asset 2

rho = 0.1670  # correlation of asset 1 and asset 2
r = 0.0259  # CD금리(91일)

F = 1  # face value
T = 3  # maturity
coupon = [0.0355, 0.0710, 0.1065, 0.1420, 0.1775, 0.2130]  # coupon rate

K = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85]  # early redemption exercise price
KI = 0.60  # Knock-In barrier

pp = 50  # number of time steps in each 6 month

Nt = (T * 2) * pp  # total number of time points
Nx = 100  # number of asset 1 differential points
Ny = 100  # number of asset 2 differential points

Nx0 = round(Nx / 2)  # current price of asset 1 in number of differentials
Ny0 = round(Ny / 2)  # current price of asset 2 in number of differentials

dt = T / Nt  # time step

dx = (xmax - xmin) / Nx  # asset 1 differential
dy = (ymax - ymin) / Ny  # asset 2 differential

els_price = get_els_price(KI, T, F, coupon, pp, K, rho, S1_0, S2_0, mu_x, mu_y, sig_x, sig_y, Nt, Nx, Ny, dt, dx, dy, r)

x = np.zeros(Nx + 1)
y = np.zeros(Ny + 1)
for i in np.arange(Nx + 1):
    x[i] = i * dx
for j in np.arange(Ny + 1):
    y[j] = j * dy

xx, yy = np.meshgrid(x, y)
zz = els_price

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title("ELS price")
ax.set_xlabel("HSCE")
ax.set_ylabel("EuroStoxx50")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% delta
# Asset 1의 가격을 고정하고 asset 2의 가격을 변화시켜가며 delta를 구함
delta_1 = np.zeros(Ny - 2)
for iy in range(1, Ny - 1):
    delta_1[iy - 1] = (zz[Nx0, iy + 1] - zz[Nx0, iy - 1]) / dy

delta_2 = np.zeros(Nx - 2)
for ix in range(1, Nx - 1):
    delta_2[ix - 1] = (zz[ix + 1, Ny0] - zz[ix - 1, Ny0]) / dx

plt.plot(y[0:-3], delta_1)
plt.axvline(x=S2_0 * KI, color='red')
plt.title('Delta: HSCE = {}'.format(S1_0))
plt.ylabel('delta')
plt.xlabel('EuroStoxx50')
plt.show()

plt.plot(x[0:-3], delta_2)
plt.axvline(x=S1_0 * KI, color='red')
plt.title('Delta: EuroStoxx50 = {}'.format(S2_0))
plt.ylabel('delta')
plt.xlabel('HSCE')
plt.show()

# %% gamma
# Asset 1의 가격을 고정하고 asset 2의 가격을 변화시켜가며 gamma를 구함
gamma_1 = np.zeros(Ny - 4)
for iy in range(1, Ny - 3):
    gamma_1[iy - 1] = (delta_1[iy + 1] - delta_1[iy - 1]) / dy

gamma_2 = np.zeros(Nx - 4)
for ix in range(1, Nx - 3):
    gamma_2[ix - 1] = (delta_2[ix + 1] - delta_2[ix - 1]) / dx

plt.plot(y[2:-3], gamma_1)
plt.axvline(x=S2_0 * KI, color='red')
plt.title('Gamma: HSCE = {}'.format(S1_0))
plt.ylabel('gamma')
plt.xlabel('EuroStoxx50')
plt.show()

plt.plot(x[2:-3], gamma_2)
plt.axvline(x=S1_0 * KI, color='red')
plt.title('Gamma: EuroStoxx50 = {}'.format(S2_0))
plt.ylabel('gamma')
plt.xlabel('HSCE')
plt.show()

# %% vega
dsigma = 0.1
sigma_list = [dsigma * x for x in range(1, int(1 / dsigma))]

# Asset 2의 sig_y을 고정하고 sig_x를 변화시켜가며 vega를 구함
option_price_vega_1 = []
for sig_1 in tqdm(sigma_list):
    op_1 = get_els_price(KI, T, F, coupon, pp, K, rho, S1_0, S2_0, mu_x, mu_y, sig_1, sig_y, Nt, Nx, Ny, dt, dx, dy, r)
    option_price_vega_1.append(op_1[Nx0, :])

vega_1 = []
for sig_index in range(1, len(option_price_vega_1) - 1):
    vega = (option_price_vega_1[sig_index + 1] - option_price_vega_1[sig_index - 1]) / (2 * dsigma)
    vega_1.append(vega)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(np.array([sigma_list[1:-1]] * 101).transpose(), np.array([y] * 7), np.array(vega_1),
                       cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
ax.set_title("Vega: HSCE = {}".format(S1_0))
ax.set_xlabel("sigma(HSCE)")
ax.set_ylabel("EuroStoxx50")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Asset 1의 sig_x을 고정하고 sig_y를 변화시켜가며 vega를 구함
option_price_vega_2 = []
for sig_2 in tqdm(sigma_list):
    op_2 = get_els_price(KI, T, F, coupon, pp, K, rho, S1_0, S2_0, mu_x, mu_y, sig_x, sig_2, Nt, Nx, Ny, dt, dx, dy, r)
    option_price_vega_2.append(op_2[Nx0, :])

vega_2 = []
for sig_index in range(1, len(option_price_vega_2) - 1):
    vega = (option_price_vega_2[sig_index + 1] - option_price_vega_2[sig_index - 1]) / (2 * dsigma)
    vega_2.append(vega)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(np.array([sigma_list[1:-1]] * 101).transpose(), np.array([x] * 7), np.array(vega_1),
                       cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
ax.set_title("Vega: EuroStoxx50 = {}".format(S2_0))
ax.set_xlabel("sigma(EURO)")
ax.set_ylabel("HSCE")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
# %% cega
drho = 0.1
rho_list = [drho * x for x in range(-9, 10)]

# HSCE의 가격을 고정하고 rho를 변화시켜가며 cega를 구함
option_price_cega_1 = []
for rho_grave in tqdm(rho_list):
    op_1 = get_els_price(KI, T, F, coupon, pp, K, rho_grave, S1_0, S2_0, mu_x, mu_y, sig_x, sig_y, Nt, Nx, Ny, dt, dx,
                         dy, r)
    option_price_cega_1.append(op_1[Nx0, :])

cega_1 = []
for rho_index in range(1, len(option_price_cega_1) - 1):
    cega = (option_price_cega_1[rho_index + 1] - option_price_cega_1[rho_index - 1]) / (2 * drho)
    cega_1.append(cega)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(np.array([rho_list[1:-1]] * len(y)).transpose(), np.array([y] * (len(rho_list) - 2)),
                       np.array(cega_1), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
ax.set_title("Cega: HSCE = {}".format(S1_0))
ax.set_xlabel("rho")
ax.set_ylabel("EuroStoxx50")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# EURO의 가격을 고정하고 rho를 변화시켜가며 cega를 구함
option_price_cega_2 = []
for rho_grave in tqdm(rho_list):
    op_2 = get_els_price(KI, T, F, coupon, pp, K, rho_grave, S1_0, S2_0, mu_x, mu_y, sig_x, sig_y, Nt, Nx, Ny, dt, dx,
                         dy, r)
    option_price_cega_2.append(op_2[Nx0, :])

cega_2 = []
for rho_index in range(1, len(option_price_cega_2) - 1):
    cega = (option_price_cega_2[rho_index + 1] - option_price_cega_2[rho_index - 1]) / (2 * drho)
    cega_2.append(cega)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(np.array([rho_list[1:-1]] * len(x)).transpose(), np.array([x] * (len(rho_list) - 2)), np.array(
    cega_2), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
ax.set_title("Cega: EuroStoxx50 = {}".format(S2_0))
ax.set_xlabel("rho")
ax.set_ylabel("HSCE")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
