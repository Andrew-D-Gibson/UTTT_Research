# Library imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Class imports
from TrainingManager import TrainingManager
from NetworkArchitectureTester import NetworkArchitectureTester

if __name__ == '__main__':
	if not os.path.exists('Network.keras'):
		model = NetworkArchitectureTester.convNet()
		model.save('Network.keras')

	trainer = TrainingManager()
	trainer.train()