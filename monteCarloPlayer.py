import utils
import easy21
import random
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--N0', type=int, default=100)
parser.add_argument('--episodes', type=int, default=100000)
parser.add_argument('--print', action='store_true')
parser.add_argument('--progress', action='store_true')
parser.add_argument('--dump', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--plot-show', action='store_true')
parser.add_argument('--plot-file', type=str, default='plots/montecarlo.png')
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
    if args.progress and episode_i % (0.01 * EPISODES) == 0:
      print('{:.2f}%...'.format(episode_i * 100 / EPISODES, EPISODES), end='\r', flush=True)

    state = easy21.init_state()
    reward, history = play_from_state(state)

    for sa in history:
      Nsa[sa] += 1
      Qsa[sa] += step_size(sa) * (reward - Qsa[sa])


calculate_V()
if args.dump: utils.dump_Q(Qsa, 'monteCarloQ.pickle')
if args.print: utils.print_Q(Qsa, Nsa)
if args.plot or args.plot_show: utils.draw_V(
  Qsa,
  file_path=args.plot and args.plot_file or None,
  show=args.plot_show
)
