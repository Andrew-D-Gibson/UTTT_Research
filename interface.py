# Entry point for the pygame GUI (human vs network, or spectator). Runs from the repo root.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from uttt.interface import main

if __name__ == '__main__':
    main()
