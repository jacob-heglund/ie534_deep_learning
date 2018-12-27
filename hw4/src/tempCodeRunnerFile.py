    state = optimizer.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000