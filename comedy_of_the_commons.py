import numpy as np

class COTCEnv(object):
	def __init__(self, num_agents, ep_len):
		self.num_agents = num_agents
		self.points = np.zeros(self.num_agents)
		self.t = 1
		self.ep_len = ep_len
	def step(self, action_n):
		act_n = [np.random.choice(self.num_agents, p=dist) for dist in action_n]
		info_n = [None] * len(action_n)
		self.points += [(act_n.count(i) == 1) * i for i in act_n] # isn't this slick
		if self.t % self.ep_len == 0: # Episode is over
			rew_n = [0] * len(action_n)
			done_n = [True] * len(action_n)
		else:
			rew_n = [np.where(np.argsort(self.points) == i)[0][0] + 1 for i in range(len(self.points))] # slicker than an electric eel
			done_n = [False] * len(action_n)
		self.t += 1
		return [np.copy(self.points) for _ in range(self.num_agents)], rew_n, done_n, info_n
	def reset(self):
		self.points = np.zeros(self.num_agents)
		return [np.copy(self.points) for _ in range(self.num_agents)]