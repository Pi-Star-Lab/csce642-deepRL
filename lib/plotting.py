# Plotting code by by Denny Britz
# repository: https://github.com/dennybritz/reinforcement-learning

import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

matplotlib.style.use("ggplot")
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0], num=num_tiles
    )
    y = np.linspace(
        env.observation_space.low[1], env.observation_space.high[1], num=num_tiles
    )
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(
        lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y])
    )

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        cmap=matplotlib.cm.coolwarm,
        vmin=30.0,
        vmax=160.0,
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Value")
    ax.set_title('Mountain "Cost To Go" Function')
    fig.colorbar(surf)
    plt.show(block=True)


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(
        lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y])
    )
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=matplotlib.cm.coolwarm,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_xlabel("Player Sum")
        ax.set_ylabel("Dealer Showing")
        ax.set_zlabel("Value")
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show(block=True)

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def plot_episode_stats(stats, smoothing_window=20, final=False):
    assert stats.episode_lengths[0] >= 0, "Can't print DP statistics"

    # Plot the episode reward over time
    fig = plt.figure(1)
    rewards = pd.Series(stats.episode_rewards)
    rewards_smoothed = rewards.rolling(
        smoothing_window, min_periods=smoothing_window
    ).mean()
    plt.clf()
    if final:
        plt.title("Result")
    else:
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.plot(rewards, label="Raw", c="b", alpha=0.3)
    if len(rewards_smoothed) >= smoothing_window:
        plt.plot(
            rewards_smoothed,
            label=f"Smooth (win={smoothing_window})",
            c="k",
            alpha=0.7,
        )
    plt.legend()
    if final:
        # plt.pause(5)
        plt.show(block=True)
    else:
        plt.pause(0.001)
