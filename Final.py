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
    mu_x_dt = (T * mu_x) / Nt
    mu_y_dt = (T * mu_y) / Nt
    dt = T / Nt

    S1 = np.ones(path_num)
    S2 = np.ones(path_num)

    min_S1 = np.ones(path_num)
    min_S2 = np.ones(path_num)

    for i in range(Nt):
        np.random.seed()
        e1 = np.random.normal(size=path_num)
        time.sleep(0.01)
        np.random.seed()
        e2 = rho * e1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=path_num)
        S1 = S1 * np.exp(mu_x_dt - (sig_x ** 2 / 2) * dt + sig_x * np.sqrt(dt) * e1)
        min_S1 = np.amin([S1, min_S1], axis=0)
        S2 = S2 * np.exp(mu_y_dt - (sig_y ** 2 / 2) * dt + sig_y * np.sqrt(dt) * e2)
        min_S2 = np.amin([S2, min_S2], axis=0)

    KI_prob = ((min_S1 <= KI) | (min_S2 <= KI)).sum() / path_num

    return KI_prob


def get_els_price(KI, T, F, coupon, pp, K, rho, S1_0, S2_0, mu_x, mu_y, sig_x, sig_y, Nt, Nx, Ny, dt, dx, dy, r):
    w = np.ones((Nt + 1, Nx + 1, Ny + 1))  # KI doesn't occur
    v = np.ones((Nt + 1, Nx + 1, Ny + 1))  # KI occurs
    # initial condition
    # BARRIER NOT TOUCH case (만기 이전에 베리어 친적 없음)
    w = w * F * 0.8
    w[pp * 0, round(S1_0 * KI / dx):, round(S2_0 * KI / dy):] = F * coupon[5]
    # BARRIER TOUCH case (만기 이전에 베리어 친적 있음)
    v = v * F * 0.8
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
    for k in tqdm(np.arange(Nt)):

        if k == pp * 1:
            v[k, round(S1_0 * K[4] / dx):, round(S2_0 * K[4] / dy):] = F * coupon[4]
        if k == pp * 2:
            v[k, round(S1_0 * K[3] / dx):, round(S2_0 * K[3] / dy):] = F * coupon[3]
        if k == pp * 3:
            v[k, round(S1_0 * K[2] / dx):, round(S2_0 * K[2] / dy):] = F * coupon[2]
        if k == pp * 4:
            v[k, round(S1_0 * K[1] / dx):, round(S2_0 * K[1] / dy):] = F * coupon[1]
        if k == pp * 5:
            v[k, round(S1_0 * K[0] / dx):, round(S2_0 * K[0] / dy):] = F * coupon[0]

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

    # No Knock-In event case
    for k in tqdm(np.arange(Nt)):

        if k == pp * 1:
            w[k, round(S1_0 * K[4] / dx):, round(S2_0 * K[4] / dy):] = F * coupon[4]
        if k == pp * 2:
            w[k, round(S1_0 * K[3] / dx):, round(S2_0 * K[3] / dy):] = F * coupon[3]
        if k == pp * 3:
            w[k, round(S1_0 * K[2] / dx):, round(S2_0 * K[2] / dy):] = F * coupon[2]
        if k == pp * 4:
            w[k, round(S1_0 * K[1] / dx):, round(S2_0 * K[1] / dy):] = F * coupon[1]
        if k == pp * 5:
            w[k, round(S1_0 * K[0] / dx):, round(S2_0 * K[0] / dy):] = F * coupon[0]

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

    # ELS price
    KI_prob = get_KI_prob(KI, T, rho, mu_x, mu_y, sig_x, sig_y, Nt, 10000)
    els_price = w * (1 - KI_prob) + v * KI_prob  # KI_prob = Prob[touch the KI barrier]

    return els_price


# %% ELS pricing

# parameters
S1_0 = 256500  # 2015-05-08 SKT end price
S2_0 = 173000  # 2015-05-08 Hyundai Auto end price
xmax = S1_0 * 2  # asset 1 (SKT) maximum value
ymax = S2_0 * 2  # asset 2 (Hyundai Auto) maximum value

xmin = 0  # asset 1 minimum value
ymin = 0  # asset 2 minimum value

mu_x = -0.093195670  # expected return of SKT
mu_y = 0.254659811  # expected return of Hyundai Auto

sig_x = 0.290935889  # volatility of asset 1
sig_y = 0.265054119  # volatility of asset 2

rho = 0.023799362  # correlation of asset 1 and asset 2 for 3 years
r = 0.0213  # CD금리(91일)
S_cost = 0.003  # 기초자산의 거래비용 0.3%

F = 10000  # face value
T = 3  # maturity
coupon = [1 + 0.0475, 1 + 0.0475 * 2, 1 + 0.0475 * 3, 1 + 0.0475 * 4, 1 + 0.0475 * 5, 1 + 0.0475 * 6]  # coupon rate

K = [0.90, 0.90, 0.85, 0.85, 0.80, 0]  # early redemption exercise price
KI = 0.60  # Knock-In barrier

pp = 123  # number of time steps in each 6 month

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
zz = els_price[-1]

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title("ELS price")
ax.set_xlabel("SKT")
ax.set_ylabel("Hyundai Auto")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% delta
delta_1 = np.zeros(shape=(Nt + 1, Nx - 2, Ny - 2))
for it in tqdm(range(0, Nt + 1)):
    for ix in range(1, Nx - 1):
        for iy in range(1, Ny - 1):
            delta_1[it, ix - 1, iy - 1] = (els_price[it, ix + 1, iy] - els_price[it, ix - 1, iy]) / (2 * dx)

delta_2 = np.zeros(shape=(Nt + 1, Nx - 2, Ny - 2))
for it in tqdm(range(0, Nt + 1)):
    for ix in range(1, Nx - 1):
        for iy in range(1, Ny - 1):
            delta_2[it, ix - 1, iy - 1] = (els_price[it, ix, iy + 1] - els_price[it, ix, iy - 1]) / (2 * dy)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(yy[1:-2, 1:-2], xx[1:-2, 1:-2], delta_2[-1], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title("Delta: dP / dSKT")
ax.set_xlabel("Hyundai Auto")
ax.set_ylabel("SKT")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(xx[1:-2, 1:-2], yy[1:-2, 1:-2], delta_1[-1], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title("Delta: dP / dHyundai Auto")
ax.set_xlabel("SKT")
ax.set_ylabel("Hyundai Auto")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
# %% gamma
gamma_1 = np.zeros(shape=(Nt + 1, Nx - 4, Ny - 4))
for it in tqdm(range(0, Nt + 1)):
    for ix in range(1, Nx - 3):
        for iy in range(1, Ny - 3):
            gamma_1[it, ix - 1, iy - 1] = (delta_1[it, ix + 1, iy] - delta_1[it, ix - 1, iy]) / dx

gamma_2 = np.zeros(shape=(Nt + 1, Nx - 4, Ny - 4))
for it in tqdm(range(0, Nt + 1)):
    for ix in range(1, Nx - 3):
        for iy in range(1, Ny - 3):
            gamma_2[it, ix - 1, iy - 1] = (delta_2[it, ix, iy + 1] - delta_2[it, ix, iy - 1]) / dy


# %% Delta hedging


def hedge_simulation(path_num, hedging_method: int, period=None, a=None, b=None):
    if hedging_method not in [1, 2, 3, 4]:
        raise ValueError("hedging_method should be 1 or 2 or 3 or 4.")

    mu_x_dt = (T * mu_x) / Nt
    mu_y_dt = (T * mu_y) / Nt
    dt = T / Nt

    S1 = np.ones((Nt + 1, path_num))
    S2 = np.ones((Nt + 1, path_num))

    min_S1 = np.ones(path_num)
    min_S2 = np.ones(path_num)

    B1 = np.ones((Nt + 1, path_num))
    B2 = np.ones((Nt + 1, path_num))

    delta_hedge_1 = np.zeros((Nt + 1, path_num))
    delta_hedge_2 = np.zeros((Nt + 1, path_num))

    S1_hedged = np.ones((Nt + 1, path_num))
    S2_hedged = np.ones((Nt + 1, path_num))

    hedge_cost = 0

    for t in range(Nt):

        S_costs_1 = np.zeros(path_num)
        S_costs_2 = np.zeros(path_num)

        new_delta_1 = np.zeros(path_num)
        new_delta_2 = np.zeros(path_num)
        for i in range(path_num):
            new_delta_1[i] = delta_1[
                t,
                np.min([np.round(S1[t, i] * (Nx0 - 1)).astype(int), 97]),
                np.min([np.round(S2[t, i] * (Ny0 - 1)).astype(int), 97])
            ]
            new_delta_2[i] = delta_2[
                t,
                np.min([np.round(S1[t, i] * (Nx0 - 1)).astype(int), 97]),
                np.min([np.round(S2[t, i] * (Ny0 - 1)).astype(int), 97])
            ]

        if hedging_method == 1:  # Time-based strategy
            if t % round(period) == 0:
                # t+1의 델타를 구한다.
                delta_hedge_1[t + 1] = new_delta_1
                delta_hedge_2[t + 1] = new_delta_2
                S_costs_1[:] = S_cost
                S_costs_2[:] = S_cost
            else:
                delta_hedge_1[t + 1] = delta_hedge_1[t]
                delta_hedge_2[t + 1] = delta_hedge_2[t]
        elif hedging_method == 2:  # Move-based strategy
            hedge_condition_1 = (np.abs((S1[t] - S1_hedged[t]) / S1_hedged[t]) > a)
            hedge_condition_2 = (np.abs((S2[t] - S2_hedged[t]) / S2_hedged[t]) > a)
            delta_hedge_1[t + 1, hedge_condition_1] = new_delta_1[hedge_condition_1]
            delta_hedge_2[t + 1, hedge_condition_2] = new_delta_2[hedge_condition_2]
            S1_hedged[t + 1] = S1_hedged[t]
            S2_hedged[t + 1] = S2_hedged[t]
            S1_hedged[t + 1, hedge_condition_1] = S1[t, hedge_condition_1]
            S2_hedged[t + 1, hedge_condition_2] = S2[t, hedge_condition_2]
            S_costs_1[hedge_condition_1] = S_cost
            S_costs_2[hedge_condition_2] = S_cost
        elif hedging_method == 3:  # Delta move-based strategy
            hedge_condition_1 = (np.abs(new_delta_1 - delta_hedge_1[t]) > b)
            hedge_condition_2 = (np.abs(new_delta_2 - delta_hedge_2[t]) > b)
            delta_hedge_1[t + 1] = delta_hedge_1[t]
            delta_hedge_2[t + 1] = delta_hedge_2[t]
            delta_hedge_1[t + 1, hedge_condition_1] = new_delta_1[hedge_condition_1]
            delta_hedge_2[t + 1, hedge_condition_2] = new_delta_2[hedge_condition_2]
            S_costs_1[hedge_condition_1] = S_cost
            S_costs_2[hedge_condition_2] = S_cost
        else:  # No strategy
            pass

        # Delta hedging한 만큼 현금을 사용한다.
        B1[t] = B1[t] - delta_hedge_1[t + 1]
        B2[t] = B2[t] - delta_hedge_2[t + 1]

        np.random.seed()
        e1 = np.random.normal(size=path_num)
        time.sleep(0.01)
        np.random.seed()
        e2 = rho * e1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=path_num)
        S1[t + 1] = S1[t] * np.exp(mu_x_dt - (sig_x ** 2 / 2) * dt + sig_x * np.sqrt(dt) * e1)
        min_S1 = np.amin([S1[t + 1], min_S1], axis=0)
        S2[t + 1] = S2[t] * np.exp(mu_y_dt - (sig_y ** 2 / 2) * dt + sig_y * np.sqrt(dt) * e2)
        min_S2 = np.amin([S2[t + 1], min_S2], axis=0)

        # V에 따라 이자를 받거나, 이자를 낸다.
        B1[t + 1] = B1[t] * (1 + r) ** dt
        B2[t + 1] = B2[t] * (1 + r) ** dt

        # log(S_t+1) - log(S_t)에 따라 delta_hedging 한만큼 손해를 보거나 이득을 본다.
        B1[t + 1] = B1[t + 1] + (S1[t + 1] / S1[t]) * delta_hedge_1[t + 1] * (1 - S_costs_1)
        B2[t + 1] = B2[t + 1] + (S2[t + 1] / S2[t]) * delta_hedge_2[t + 1] * (1 - S_costs_2)

        hedge_cost += (delta_hedge_1[t + 1] * S_costs_1 + delta_hedge_2[t + 1] * S_costs_2) / 2

    payoff = np.zeros(path_num)
    ending_balance = np.zeros(path_num)

    # Redemption 1
    for t in range(1, T * 2):
        payoff[(payoff == 0) & (S1[t * pp] >= K[t - 1]) & (S2[t * pp] >= K[t - 1])] = coupon[t - 1]
        balance_condition = (ending_balance == 0) & (S1[t * pp] >= K[t - 1]) & (S2[t * pp] >= K[t - 1])
        ending_balance[balance_condition] = (B1[t * pp, balance_condition] + B2[t * pp, balance_condition]) / 2

    payoff[(payoff == 0) & (min_S1 >= KI) & (min_S2 >= KI)] = coupon[5]
    payoff[payoff == 0] = 0.8
    ending_balance[ending_balance == 0] = (B1[-1, ending_balance == 0] + B2[-1, ending_balance == 0]) / 2

    margin = (ending_balance - payoff) * F
    hedge_cost = hedge_cost * F

    return np.mean(margin), np.mean(hedge_cost), np.std(margin)


# %% Time-based strategy
days = range(1, 124, 2)
time_based_margins = []
time_based_hedge_costs = []
for day in tqdm(days):
    margin, hedge_cost, _ = hedge_simulation(path_num=1000, hedging_method=1, period=day)
    time_based_margins.append(margin)
    time_based_hedge_costs.append(hedge_cost)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(days, time_based_margins, 'b-', label='Margin')
ax1.set_xlabel('Period')
ax1.set_ylabel('Margin')
ax2.plot(days, time_based_hedge_costs, 'r:')
ax1.plot(np.nan, 'r:', label='Hedge cost')
ax2.set_ylabel('Hedge cost')
ax1.legend(loc='best')
plt.title('Time-based strategy')
plt.legend()
plt.show()

# %% Move-based strategy
As = [x / 100.0 for x in range(1, 21)]
asset_move_based_margins = []
asset_move_based_hedge_costs = []
for a in tqdm(As):
    margin, hedge_cost, _ = hedge_simulation(path_num=1000, hedging_method=2, a=a)
    asset_move_based_margins.append(margin)
    asset_move_based_hedge_costs.append(hedge_cost)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(As, asset_move_based_margins, 'b-', label='Margin')
ax1.set_xlabel('a')
ax1.set_ylabel('Margin')
ax2.plot(As, asset_move_based_hedge_costs, 'r:')
ax1.plot(np.nan, 'r:', label='Hedge cost')
ax2.set_ylabel('Hedge cost')
ax1.legend(loc='best')
plt.title('Move-based strategy')
plt.legend()
plt.show()

# %% Delta move-based strategy
Bs = [x / 100.0 for x in range(1, 21)]
delta_move_based_margins = []
delta_move_based_hedge_costs = []
for b in tqdm(Bs):
    margin, hedge_cost, _ = hedge_simulation(path_num=1000, hedging_method=3, b=b)
    delta_move_based_margins.append(margin)
    delta_move_based_hedge_costs.append(hedge_cost)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(Bs, delta_move_based_margins, 'b-', label='Margin')
ax1.set_xlabel('b')
ax1.set_ylabel('Margin')
ax2.plot(Bs, delta_move_based_hedge_costs, 'r:')
ax1.plot(np.nan, 'r:', label='Hedge cost')
ax2.set_ylabel('Hedge cost')
ax1.legend(loc='best')
plt.title('Delta move-based strategy')
plt.legend()
plt.show()

# %% No strategy
path_nums = range(1, 11)
std_nos = []
std_times = []
for path_num in tqdm(path_nums):
    _, _, std_no = hedge_simulation(int(np.e ** path_num), 4)
    _, _, std_time = hedge_simulation(int(np.e ** path_num), 3, b=0.01)
    std_nos.append(std_no)
    std_times.append(std_time)

fig, ax1 = plt.subplots()
ax1.plot(path_nums[1:], std_nos[1:], 'b-', label='No strategy')
ax1.plot(path_nums[1:], std_times[1:], 'r:', label='Time-based strategy')
ax1.set_xlabel('log(# of path)')
ax1.set_ylabel('Stdev')
ax1.legend(loc='best')
plt.title('No strategy vs Time-based strategy')
plt.legend()
plt.show()
