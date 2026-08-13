# Library imports
import os
import csv
import time
import pickle
import numpy as np
import tensorflow as tf

# Config setup
from Config import config

POOL_PATH = 'PretrainExamples/pool.pkl'


class PretrainManager:
    # Counterpart to TrainingManager for the raw-MCTS bootstrap phase: no
    # self-play (examples already sit in POOL_PATH, produced once by
    # GenerateRawMCTSExamples.py) and no candidate/champion gating (there's no
    # opponent to gate against here - progress is judged externally, by watching
    # pretrain_log.csv and running NetworkTournament.py). Every round trains
    # self.network in place and is kept unconditionally.
    def __init__(self):
        self.network = tf.keras.models.load_model('Network.keras')

        if not os.path.exists(POOL_PATH):
            raise FileNotFoundError(
                f'{POOL_PATH} not found - run GenerateRawMCTSExamples.py first to build the raw-MCTS example pool.')

        with open(POOL_PATH, 'rb') as file:
            self.pool = pickle.load(file)

        print(f'Loaded pretrain pool: {len(self.pool)} examples from {POOL_PATH}')

        self.log_path = 'pretrain_log.csv'
        self.log_fields = [
            'round', 'num_examples', 'epochs_run', 'epochs_configured', 'stopped_early',
            'train_loss', 'val_loss',
            'train_policy_loss', 'val_policy_loss',
            'train_value_loss', 'val_value_loss',
            'train_policy_acc', 'val_policy_acc',
        ]

        self.rounds_run = 0

    def log_metrics(self, row):
        file_exists = os.path.exists(self.log_path)
        with open(self.log_path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.log_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def train_on_examples(self, examples, round_num):
        board_train = np.empty((len(examples),9,9,4))
        search_probs_train = np.empty((len(examples),81))
        eval_train = np.empty((len(examples),1))

        for i, example in enumerate(examples):
            board_train[i,:,:,:] = example.board_array
            search_probs_train[i,:] = example.search_probs
            eval_train[i] = example.reward

        loss_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['training_patience'],
            restore_best_weights=True,
        )

        history = self.network.fit(
            board_train,
            [search_probs_train, eval_train],
            batch_size=config['training']['minibatch_size'],
            epochs=config['training']['training_epochs'],
            validation_split=0.1,
            shuffle=True,
            callbacks=[loss_callback],
            verbose=2)  # one line/epoch, no progress bar - keeps a long fit() from looking hung

        h = history.history
        train_loss, val_loss = h['loss'][-1], h['val_loss'][-1]
        train_policy_loss, val_policy_loss = h['policy_output_loss'][-1], h['val_policy_output_loss'][-1]
        train_value_loss, val_value_loss = h['value_output_loss'][-1], h['val_value_output_loss'][-1]
        train_policy_acc, val_policy_acc = h['policy_output_accuracy'][-1], h['val_policy_output_accuracy'][-1]

        epochs_run = len(h['loss'])
        epochs_configured = config['training']['training_epochs']
        stopped_early = epochs_run < epochs_configured
        epoch_note = f'stopped early at {epochs_run}/{epochs_configured}' if stopped_early else f'ran all {epochs_run}'

        print(f' Loss   train/val: {train_loss:.4f} / {val_loss:.4f}')
        print(f' Policy train/val: acc {train_policy_acc:.3f}/{val_policy_acc:.3f}, loss {train_policy_loss:.4f}/{val_policy_loss:.4f}')
        print(f' Value  train/val loss: {train_value_loss:.4f} / {val_value_loss:.4f}')
        print(f' Epochs: {epoch_note}')

        self.log_metrics({
            'round': round_num,
            'num_examples': len(examples),
            'epochs_run': epochs_run,
            'epochs_configured': epochs_configured,
            'stopped_early': stopped_early,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_policy_loss': train_policy_loss, 'val_policy_loss': val_policy_loss,
            'train_value_loss': train_value_loss, 'val_value_loss': val_value_loss,
            'train_policy_acc': train_policy_acc, 'val_policy_acc': val_policy_acc,
        })

    def print_settings(self):
        print('=== Pretrain run settings ===')
        print(f'  pool: {len(self.pool)} examples at {POOL_PATH}')
        print(f"  rounds: {config['training']['num_of_episodes']}, "
              f"sample_size={min(config['training']['training_sample_size'], len(self.pool))}, "
              f"minibatch={config['training']['minibatch_size']}, "
              f"epochs<={config['training']['training_epochs']} (patience={config['training']['training_patience']})")

    def pretrain(self):
        self.print_settings()

        num_rounds = config['training']['num_of_episodes']
        sample_size = np.min([config['training']['training_sample_size'], len(self.pool)])

        for round_num in range(num_rounds):
            round_start_time = time.time()
            print(f'\n\n===== Pretrain round {round_num} =====')

            training_sample = np.random.choice(self.pool, sample_size, replace=False)
            print(f'Training on {sample_size} examples (sampled from {len(self.pool)} in pool)')

            self.train_on_examples(training_sample, round_num)

            # No gating: every round is kept unconditionally, so Networks/ fills in
            # densely here (Pretrain_0, Pretrain_1, ...) unlike self-play's sparse
            # Episode_N. NetworkTournament.py picks these up alongside Episode_N
            # checkpoints once you switch over to self-play.
            self.network.save(f'Networks/Pretrain_{round_num}.keras')
            self.network.save('Network.keras')
            self.rounds_run += 1

            print(f'===== Pretrain round {round_num} done in {time.time() - round_start_time:.1f}s =====')
