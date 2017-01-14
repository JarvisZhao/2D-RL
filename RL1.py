import numpy as np
import pandas as pd 
import time

map_size = 5
N_states = map_size * map_size
Actions = ['left','right','up','down']
epsilon = 1
alpha = 0.1
LAMBDA = 0.9
MAX_episode = 20
Fresh_rate = 0.1

def build_q_table(n_states,actions):
	table = pd.DataFrame(np.zeros((n_states, len(actions))),
		columns = actions)
	return table

def choose_actions(state, q_table):
	state_actions = q_table.iloc[state, :]
	if (np.random.uniform()>epsilon) or (state_actions.all()==0):
		action_name = np.random.choice(Actions)
	else:
		action_name = state_actions.argmax()
	return action_name
def get_env_feedback(S,A):
	if A == 'right':
		if S == N_states -2:
			S_ = 'Terminal'
			R = 1
		elif S%map_size == map_size-1:
			S_ = S
			R = 0
		else:
			S_ = S +1
			R = 0
	elif A == 'left':
		R = 0
		if S%map_size == 0:
			S_ = S
		else:
			S_ = S-1
	elif A == 'up':
		R = 0
		if int(S/map_size) == 0:
			S_ =S
		else:
			S_ = S-map_size
	else:
		if S == N_states - map_size-1:
			S_ = 'Terminal'
			R=1
		elif int(S/map_size) ==map_size-1:
			S_ = S
			R = 0
		else:
			S_ = S+map_size
			R = 0
	return S_, R




def update_env(S, episode, step_counter):
	env_list = []
	for k in range(map_size-1):
		temp = ['-']*(map_size)
		env_list.append(temp)
	env_list.append(['-']*(map_size-1) + ['T'])
	if S == 'Terminal':
		intersection = 'Episode %s: total_step = %s' %(episode+1, step_counter)
		print('\r{}'.format(intersection), end="")
		time.sleep(2)
		print('\r                       ',end='')
	else:
		env_list[int(S/map_size)][int(S%map_size)] = 'o'
		for i in range(map_size):
			intersection = ''.join(env_list[i])
			print('\r{}'.format(intersection))
		time.sleep(Fresh_rate)

def rl():
	q_table = build_q_table(N_states,Actions)
	for episode in range(MAX_episode):
		step_counter = 0
		S = 0
		is_terminated = False
		update_env(S, episode, step_counter)
		while not is_terminated:
			
			A = choose_actions(S, q_table)
			print(S)
			print(A)
			S_,R =get_env_feedback(S,A)
			q_predict = q_table.ix[S,A]
			print(S_)
			if S_ != 'Terminal':
				q_target = R + LAMBDA * q_table.iloc[S_, : ].max()
			else:
				q_target = R
				is_terminated = True

			q_table.ix[S, A] += alpha * (q_target - q_predict)
			
			S = S_
			update_env(S, episode,step_counter+1)
			step_counter+=1
	return q_table


if __name__ == '__main__':
	q_table = rl()
	print(q_table)
	



