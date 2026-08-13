import numpy as np
import random
import time

from PlayerAgent import PlayerAgent, RandomAgent, RolloutMCTSAgent, agent_game


class ELOManager:
	@staticmethod
	def expected_result(agent_1_elo, agent_2_elo):
	    return 1 / (1 + 10**((agent_2_elo-agent_1_elo) / 400))


	@staticmethod
	def delta_elo(expected_result, actual_result, k=32):
	    return k*(actual_result - expected_result)


	@staticmethod
	def play_rated_game(agent_1: PlayerAgent, agent_2: PlayerAgent):
		exp = ELOManager.expected_result(agent_1.elo, agent_2.elo)

		result = agent_game(agent_1, agent_2)
		if result == 1:
			actual = 1
		elif result == 0:
			actual = 0.5
		elif result == -1:
			actual = 0

		delta = ELOManager.delta_elo(exp, actual)

		agent_1.elo += delta
		agent_2.elo -= delta



	def __init__(self, agent_list):
		self.agent_list = agent_list


	def calibrate(self, iterations):
		print('\nInitial ELOs:')
		for agent in self.agent_list:
			print(f'{agent.name}: {agent.elo}')

		agent_elo_history = np.empty((len(self.agent_list), -(-iterations // 10)))  # ceil(iterations/10)

		print('Calibrating ELOs...')
		start_time = time.time()
		progress_every = max(1, iterations // 20)
		for i in range(iterations):
			if (i%10)==0:
				agent_elo_history[:,int(i/10)] = [agent.elo for agent in self.agent_list]

			agents = np.random.choice(self.agent_list, 2, replace=False)
			ELOManager.play_rated_game(agents[0], agents[1])

			if (i+1) % progress_every == 0 or (i+1) == iterations:
				elapsed = time.time() - start_time
				rate = elapsed / (i+1)
				eta = rate * (iterations - (i+1))
				print(f'  {i+1}/{iterations} games played, {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining')


		print('\nFinal ELOs:')
		for agent in self.agent_list:
			print(f'{agent.name}: {agent.elo}')

		return agent_elo_history


if __name__ == '__main__':
	iterations = 20

	elo = ELOManager([
		RandomAgent(), 
		RolloutMCTSAgent(iterations=16), 
		RolloutMCTSAgent(iterations=64), 
		RolloutMCTSAgent(iterations=256), 
		RolloutMCTSAgent(iterations=1024), 
		RolloutMCTSAgent(iterations=4096), 
	])

	history = elo.calibrate(iterations)

	print(history)