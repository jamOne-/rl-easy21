import utils
import easy21
import random
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--N0', type=int, default=100)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--progress', action='store_true')
parser.add_argument('--mse-plot', action='store_true')
parser.add_argument('--mse-episodes-plot', action='store_true')
parser.add_argument('--v-plot', action='store_true')
args = parser.parse_args()

N_0 = args.N0
EPISODES = args.episodes
DISCOUNT = 1
monteCarloQ = utils.load_Qdump('monteCarloQ.pickle')
lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def step_size(N, sa):
  return 1 / N[sa]


def epsilon(N, state):
  return N_0 / (N_0 + get_Ns(N, state))


def get_Ns(N, state):
  return N[state, 0] + N[state, 1]


def get_action(Q, N, state):
  e = epsilon(N, state)

  if random.random() < e:
    return random.randint(0, 1)
  else:
    q0, q1 = Q[state, 0], Q[state, 1]

    if q0 > q1:
      return 0
    elif q0 == q1:
      return random.randint(0, 1)
    else:
      return 1


def sarsa(lamb, all_errors=False):
  Q = defaultdict(int)
  N = defaultdict(int)
  errors = []

  for episode_i in range(EPISODES):
    if args.progress and episode_i % (0.1 * EPISODES) == 0:
      print('{:.2f}%...'.format((episode_i + EPISODES * lamb * 10 ) * 100 / (EPISODES * len(lambdas))), end='\r', flush=True)

    E = defaultdict(int)
    state = easy21.init_state()
    action = get_action(Q, N, state)
    N[state, action] += 1
    is_terminal = False

    while not is_terminal:
      reward, new_state, is_terminal = easy21.step(state, action)
      new_action = get_action(Q, N, new_state)

      E[state, action] += 1
      N[new_state, new_action] += 1
      d = reward + DISCOUNT * Q[new_state, new_action] - Q[state, action]

      for (state, action), e in E.items():
        Q[state, action] += step_size(N, (state, action)) * d * e
        E[state, action] *= DISCOUNT * lamb
      
      state, action = new_state, new_action

    if all_errors: errors.append(calculate_error(Q))

  if all_errors:
    return errors, Q
  else:
    return calculate_error(Q), Q


def calculate_error(Q):
  error = 0

  for sa, value in monteCarloQ.items():
    error += (value - Q[sa]) ** 2

  return error / len(monteCarloQ)


def mse_plot():
  mse_errors = []
  for lamb in lambdas:
    mse_errors.append(sarsa(lamb)[0])

  plt.plot(lambdas, mse_errors)
  plt.xlabel('λ factor')
  plt.ylabel('Mean Square Error')
  plt.title('MSE of Sarsa(λ) value function ({} episodes)'.format(EPISODES))
  plt.grid(True)
  plt.savefig('plots/sarsa_mse_{}.png'.format(EPISODES))
  plt.show()


def mse_episodes_plot():
  for lamb in lambdas:
    errors, _ = sarsa(lamb, all_errors=True)
    plt.plot(list(range(EPISODES)), errors)
  
  plt.legend(['λ={:0.1f}'.format(lamb) for lamb in lambdas], ncol=4)
  plt.xlabel('episode')
  plt.ylabel('Mean Square Error')
  plt.title('MSE of Sarsa(λ) against episode number')
  plt.grid(True)
  plt.savefig('plots/sarsa_mse_episodes_{}.png'.format(EPISODES))
  plt.show()


def v_plot():
  for lamb in [0, 1]:
    _, Q = sarsa(lamb)
    utils.draw_V(
      Q,
      title='V* by Sarsa(λ={}) ({} episodes)'.format(lamb, EPISODES),
      file_path='plots/sarsa_v_{}_{}'.format(lamb, EPISODES),
      show=True
    )


if args.mse_plot: mse_plot()
if args.mse_episodes_plot: mse_episodes_plot()
if args.v_plot: v_plot()
