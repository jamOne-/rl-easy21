import random

MAX_VALUE = 21
BLACK_RATIO = 2/3
DEALER_STICK_AT = 17


def draw_card(color=None, black_ratio=BLACK_RATIO):
  color = color or (1 if random.random() < black_ratio else -1)
  value = random.randint(1, 10)
  return color, value


def init_state():
  return draw_card(1)[1], draw_card(1)[1]


def step(state, action):
  dealer, player = state
  reward = 0
  terminal = False

  if action == 'h' or action == 'hit':
    color, value = draw_card()
    player += color * value
    player_busted = is_busted(player)

    reward = -1 if player_busted else 0
    terminal = player_busted

  elif action == 's' or action == 'stick':
    while dealer > 0 and dealer < DEALER_STICK_AT:
      color, value = draw_card()
      dealer += color * value

    reward = calculate_reward(dealer, player)
    terminal = True

  return reward, (dealer, player), terminal


def calculate_reward(dealer, player):
  if is_busted(dealer) or player > dealer:
      return 1
  elif player == dealer:
    return 0
  else:
    return -1


def is_busted(player):
  return player < 1 or player > MAX_VALUE