import colorama
import easy21

colorama.init(convert=True)


while True:
  state = easy21.init_state()
  print('state:', state)

  is_terminal = False
  while not is_terminal:
    print('action ({}it/{}tick): '.format(colorama.Fore.LIGHTGREEN_EX + 'h' + colorama.Style.RESET_ALL, colorama.Fore.LIGHTRED_EX + 's' + colorama.Style.RESET_ALL), end='')
    action = input()
    reward, state, is_terminal = easy21.step(state, action)
    print('state:', state)

  print('reward:', reward)
  print('\n')
