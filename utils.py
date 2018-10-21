import easy21
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def dump_Q(Q, file_path):
  with open(file_path, 'wb') as file:
    pickle.dump(Q, file)


def load_Qdump(file_path):
  with open(file_path, 'rb') as file:
    return pickle.load(file)


def get_plot_data(Q):
  X, Y, Z = [], [], []

  for dealer in range(1, 11):
    for player in range(1, easy21.MAX_VALUE + 1):
      state = dealer, player

      X.append(dealer)
      Y.append(player)
      Z.append(max(Q[state, 0], Q[state, 1]))

  return X, Y, Z


def draw_V(Q, title='V*', file_path=None, show=False):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  X, Y, Z = get_plot_data(Q)
  surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm)

  ax.set_xlabel('dealer hand')
  ax.set_ylabel('player sum')
  ax.set_zlabel('value of state')
  plt.title(title)
  plt.xticks(np.arange(1, 11))

  if file_path != None: plt.savefig(file_path)
  if show: plt.show()


def get_Ns(N, state):
  return N[state, 0] + N[state, 1]


def print_Q(Q, N):
  for dealer in range(1, 11):
    for player in range(1, easy21.MAX_VALUE + 1):
      state = dealer, player
      print('{}, {} ({}):\n\tstick: {} ({})\n\thit: {} ({})'.format(dealer, player, get_Ns(N, state), Q[state, 0], N[state, 0], Q[state, 1], N[state, 1]))
