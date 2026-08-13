# Library imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Class imports
from GenerateRawMCTSExamples import generate_pool
from PretrainManager import PretrainManager
from NetworkArchitectureTester import NetworkArchitectureTester

if __name__ == '__main__':
	# Fire-and-forget entry point: builds/resumes the raw-MCTS pool first
	# (generate_pool() is a no-op if it already meets target, so this is safe to
	# run unattended or restart after an interruption), then runs the
	# training-only loop against it. A single `python Pretrain.py` overnight
	# does both steps with nothing else to trigger by hand.
	if not os.path.exists('Network.keras'):
		model = NetworkArchitectureTester.convNet()
		model.save('Network.keras')

	generate_pool()

	pretrainer = PretrainManager()
	pretrainer.pretrain()
