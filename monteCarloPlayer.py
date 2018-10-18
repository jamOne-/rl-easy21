import easy21
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('--N0', type=int, default=100)
parser.add_argument('--episodes', type=int, default=100000)
parser.add_argument('--print', action='store_true')
parser.add_argument('--progress', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--plot-show', action='store_true')
parser.add_argument('--plot-file', type=str, default='montecarlo.png')
args = parser.parse_args()


N_0 = args.N0
EPISODES = args.episodes
Nsa = defaultdict(int)
Qsa = defaultdict(int)


def step_size(sa):
  return 1 / Nsa[sa]


def epsilon(state):
  return N_0 / (N_0 + get_Ns(state))


def get_Ns(state):
  return Nsa[state, 0] + Nsa[state, 1]


def get_action(state):
  e = epsilon(state)

  if random.random() < e:
    return random.randint(0, 1)
  else:
    q0, q1 = Qsa[state, 0], Qsa[state, 1]

    if q0 > q1:
      return 0
    elif q0 == q1:
      return random.randint(0, 1)
    else:
      return 1


def play_from_state(state):
  is_terminal = False
  history = []

  while not is_terminal:
    action = get_action(state)
    history.append((state, action))
    reward, state, is_terminal = easy21.step(state, action)
  
  return reward, history


def calculate_V():
  for episode_i in range(EPISODES):
    if args.progress and episode_i % 10000 == 0:
      print('{} / {}'.format(episode_i, EPISODES), end='\r', flush=True)

    state = easy21.init_state()
    reward, history = play_from_state(state)

    for sa in history:
      Nsa[sa] += 1
      Qsa[sa] += step_size(sa) * (reward - Qsa[sa])

  if args.print: print_greedy()
  if args.plot or args.plot_show: draw_plot()


def get_plot_data():
  X, Y, Z = [], [], []

  for dealer in range(1, 11):
    for player in range(1, easy21.MAX_VALUE + 1):
      state = dealer, player

      X.append(dealer)
      Y.append(player)
      Z.append(max(Qsa[state, 0], Qsa[state, 1]))

  return X, Y, Z


def print_greedy():
  for dealer in range(1, 11):
    for player in range(1, easy21.MAX_VALUE + 1):
      state = dealer, player
      best_action = get_action(state)
      print('{}, {} ({}):\n\tstick: {} ({})\n\thit: {} ({})'.format(dealer, player, get_Ns(state), Qsa[state, 0], Nsa[state, 0], Qsa[state, 1], Nsa[state, 1]))


def draw_plot():
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  X, Y, Z = get_plot_data()
  surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm)

  ax.set_xlabel('dealer hand')
  ax.set_ylabel('player sum')
  ax.set_zlabel('value of state')
  plt.title('V*')
  plt.xticks(np.arange(1, 11))
  
  if args.plot: plt.savefig(args.plot_file)
  if args.plot_show: plt.show()


calculate_V()
