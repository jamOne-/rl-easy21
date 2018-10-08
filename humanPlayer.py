import easy21

while True:
  state = easy21.init_state()
  print('state:', state)

  reward, action = 0, 'hit'
  while reward == 0 and action == 'hit':
    action = input('action: ')
    reward, state = easy21.step(state, action)
    print('state:', state)

  print('reward:', reward)
  print('\n')
