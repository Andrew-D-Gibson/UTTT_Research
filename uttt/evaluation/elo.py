class ELOManager:
	@staticmethod
	def expected_result(agent_1_elo, agent_2_elo):
	    return 1 / (1 + 10**((agent_2_elo-agent_1_elo) / 400))


	@staticmethod
	def delta_elo(expected_result, actual_result, k=32):
	    return k*(actual_result - expected_result)