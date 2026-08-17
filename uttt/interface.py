"""Pygame GUI for Ultimate Tic-Tac-Toe: play against a loaded network, or watch
two networks play each other in real time.

Usage:
    python interface.py human --network data/Network.keras --human-color x
    python interface.py spectate --network1 data/Network.keras --network2 data/Network.keras

Controls are shown in the in-app legend and vary slightly by mode (see draw_legend()).
"""

import argparse
import queue
import threading
import time

import numpy as np
import pygame
import tensorflow as tf

from uttt.config import config
from uttt.paths import NETWORK_PATH
from uttt.search.mcts import MCTS
from uttt.board.uttt_board import UTTTBoard


# --- Board geometry -----------------------------------------------------
CELL = 56
GAP = 8                      # extra spacing between the three 3x3 sub-board groups
MARGIN = 26
BOARD_PIXELS = 9 * CELL + 2 * GAP
STATUS_H = 190
WINDOW_W = BOARD_PIXELS + 2 * MARGIN
WINDOW_H = BOARD_PIXELS + 2 * MARGIN + STATUS_H

DEPTH_STEP = 25
DEPTH_MIN = 10
DEPTH_MAX = 3000
SPECTATE_MOVE_DELAY = 0.35   # seconds to pause on a finished move before the next agent thinks

BG = (24, 24, 32)
CELL_EMPTY = (240, 240, 245)
CELL_ELIGIBLE = (221, 247, 229)
CELL_DECIDED = (205, 205, 212)
GRID_THIN = (175, 175, 185)
GRID_THICK = (40, 40, 55)
ELIGIBLE_OUTLINE = (250, 190, 40)
X_COLOR = (206, 55, 55)
O_COLOR = (50, 100, 210)
DRAW_COLOR = (140, 140, 150)
TEXT_COLOR = (235, 235, 240)
DIM_TEXT = (150, 150, 160)
PANEL_BG = (18, 18, 24)


def px_x(col):
    return MARGIN + col * CELL + GAP * (col // 3)


def px_y(row):
    return MARGIN + row * CELL + GAP * (row // 3)


def move_to_rowcol(move):
    subboard, local = divmod(move, 9)
    srow, scol = divmod(subboard, 3)
    lrow, lcol = divmod(local, 3)
    return srow * 3 + lrow, scol * 3 + lcol


def cell_rect(row, col):
    return pygame.Rect(px_x(col), px_y(row), CELL, CELL)


def subboard_rect(subboard):
    srow, scol = divmod(subboard, 3)
    return pygame.Rect(px_x(scol * 3), px_y(srow * 3), 3 * CELL, 3 * CELL)


def cell_owner(board, move):
    bit = 1 << move
    if board.x & bit:
        return 'X'
    if board.o & bit:
        return 'O'
    return None


def subboard_owner(board, subboard):
    bit = 1 << subboard
    if board.subboard_x & bit:
        return 'X'
    if board.subboard_o & bit:
        return 'O'
    if board.subboard_draws & bit:
        return 'D'
    return None


class InterfaceAgent:
    """Wraps an MCTS tree with runtime-adjustable search depth and move-selection
    style, so a single agent instance can be tuned live from the UI instead of
    being locked to one PlayerAgent subclass (see uttt/player/agent.py's
    NetworkMCTSAgent/ProbabilisticNetworkMCTSAgent, which hardcode one or the other).
    """

    def __init__(self, network, name):
        self.network = network
        self.name = name
        self.mcts = MCTS(network=network)

    def get_move(self, depth, deterministic):
        self.mcts.search(iterations=depth)
        if deterministic:
            child_choice = np.argmax(self.mcts.pi)
        else:
            child_choice = np.random.choice(len(self.mcts.pi), p=self.mcts.pi)
        return self.mcts.children[child_choice].move

    def make_move(self, move):
        self.mcts = self.mcts.make_move(move)

    def reset(self):
        self.mcts.reset()

    @property
    def board(self):
        return self.mcts.board


def load_network(path):
    print(f'Loading network: {path}')
    return tf.keras.models.load_model(path)


def draw_board(screen, board, fonts, highlight_eligible):
    for subboard in range(9):
        rect = subboard_rect(subboard)
        owner = subboard_owner(board, subboard)

        for local in range(9):
            move = subboard * 9 + local
            row, col = move_to_rowcol(move)
            r = cell_rect(row, col)

            if owner is not None:
                color = CELL_DECIDED
            elif highlight_eligible and subboard in board.eligible_subboards:
                color = CELL_ELIGIBLE
            else:
                color = CELL_EMPTY
            pygame.draw.rect(screen, color, r)
            pygame.draw.rect(screen, GRID_THIN, r, 1)

            mark = cell_owner(board, move)
            if mark == 'X':
                label = fonts['cell'].render('X', True, X_COLOR)
            elif mark == 'O':
                label = fonts['cell'].render('O', True, O_COLOR)
            else:
                label = None
            if label is not None:
                screen.blit(label, label.get_rect(center=r.center))

        pygame.draw.rect(screen, GRID_THICK, rect, 3)

        if owner is not None:
            overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            if owner == 'X':
                overlay.fill((*X_COLOR, 60))
                glyph = fonts['big'].render('X', True, (*X_COLOR, 200))
            elif owner == 'O':
                overlay.fill((*O_COLOR, 60))
                glyph = fonts['big'].render('O', True, (*O_COLOR, 200))
            else:
                overlay.fill((*DRAW_COLOR, 70))
                glyph = fonts['big'].render('=', True, (*DRAW_COLOR, 200))
            overlay.blit(glyph, glyph.get_rect(center=(rect.width // 2, rect.height // 2)))
            screen.blit(overlay, rect.topleft)

        if highlight_eligible and owner is None and subboard in board.eligible_subboards and not board.is_game_over():
            pygame.draw.rect(screen, ELIGIBLE_OUTLINE, rect, 4)

    outer = pygame.Rect(px_x(0), px_y(0), BOARD_PIXELS, BOARD_PIXELS)
    pygame.draw.rect(screen, GRID_THICK, outer, 4)


def draw_text_lines(screen, fonts, lines, x, y):
    for text, color in lines:
        surf = fonts['status'].render(text, True, color)
        screen.blit(surf, (x, y))
        y += surf.get_height() + 4
    return y


def draw_legend(screen, fonts, x, y, mode):
    common = ['[UP/DOWN] search depth   [D] toggle deterministic/probabilistic   [R] restart   [Esc] quit']
    if mode == 'spectate':
        common.insert(0, '[Space] pause/resume   [Right] step (while paused)')
    for line in common:
        surf = fonts['legend'].render(line, True, DIM_TEXT)
        screen.blit(surf, (x, y))
        y += surf.get_height() + 2


def game_over_text(board):
    if board.value == 1:
        return 'X wins!', X_COLOR
    if board.value == -1:
        return 'O wins!', O_COLOR
    return "It's a draw!", DRAW_COLOR


class ThinkingWorker:
    """Runs one agent.get_move() call on a background thread so the pygame event
    loop / renderer never blocks on a multi-hundred-iteration MCTS search."""

    def __init__(self):
        self.result_queue = queue.Queue()
        self.busy = False
        self.started_at = None

    def start(self, agent, depth, deterministic):
        self.busy = True
        self.started_at = time.time()

        def run():
            move = agent.get_move(depth, deterministic)
            self.result_queue.put(move)

        threading.Thread(target=run, daemon=True).start()

    def poll(self):
        try:
            move = self.result_queue.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return move

    def thinking_label(self):
        dots = '.' * (int((time.time() - self.started_at) * 2) % 4)
        return f'Thinking{dots}'


def run_human_mode(network_path, human_color, depth):
    network = load_network(network_path)
    ai = InterfaceAgent(network, name=f'AI ({network_path})')
    human_is_x = human_color == 'x'

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption('Ultimate Tic-Tac-Toe — Human vs Network')
    fonts = make_fonts()
    clock = pygame.time.Clock()

    deterministic = True
    worker = ThinkingWorker()
    running = True

    def maybe_start_ai_turn():
        board = ai.board
        if not board.is_game_over() and (board.x_move != human_is_x):
            worker.start(ai, depth, deterministic)

    maybe_start_ai_turn()

    while running:
        board = ai.board
        human_turn = (not worker.busy) and (not board.is_game_over()) and (board.x_move == human_is_x)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    depth = min(DEPTH_MAX, depth + DEPTH_STEP)
                elif event.key == pygame.K_DOWN:
                    depth = max(DEPTH_MIN, depth - DEPTH_STEP)
                elif event.key == pygame.K_d:
                    deterministic = not deterministic
                elif event.key == pygame.K_r and not worker.busy:
                    ai.reset()
                    maybe_start_ai_turn()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and human_turn:
                mx, my = event.pos
                for move in board.find_moves():
                    row, col = move_to_rowcol(move)
                    if cell_rect(row, col).collidepoint(mx, my):
                        ai.make_move(move)
                        maybe_start_ai_turn()
                        break

        if worker.busy:
            move = worker.poll()
            if move is not None:
                ai.make_move(move)

        screen.fill(BG)
        draw_board(screen, board, fonts, highlight_eligible=True)

        status_y = px_y(9) + 14
        if board.is_game_over():
            text, color = game_over_text(board)
            lines = [(text, color), ('[R] to play again', DIM_TEXT)]
        else:
            turn_label = 'Your move' if human_turn else f'{ai.name} thinking' if worker.busy else "Opponent's move"
            mover = 'X' if board.x_move else 'O'
            lines = [
                (f"{mover} to move — {turn_label}{'' if not worker.busy else ' ' + worker.thinking_label()}",
                 TEXT_COLOR),
                (f'You are playing {"X" if human_is_x else "O"}', DIM_TEXT),
            ]
        lines.append((f'Search depth: {depth}    Mode: {"deterministic" if deterministic else "probabilistic"}',
                      DIM_TEXT))
        draw_text_lines(screen, fonts, lines, MARGIN, status_y)
        draw_legend(screen, fonts, MARGIN, WINDOW_H - 30, 'human')

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def run_spectate_mode(network1_path, network2_path, depth):
    network1 = load_network(network1_path)
    network2 = load_network(network2_path) if network2_path != network1_path else network1
    x_agent = InterfaceAgent(network1, name=network1_path)
    o_agent = InterfaceAgent(network2, name=network2_path)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption('Ultimate Tic-Tac-Toe — Network vs Network')
    fonts = make_fonts()
    clock = pygame.time.Clock()

    deterministic = True
    paused = False
    worker = ThinkingWorker()
    next_dispatch_time = 0.0
    running = True

    def current_agent():
        return x_agent if x_agent.board.x_move else o_agent

    def dispatch_if_ready():
        nonlocal next_dispatch_time
        board = x_agent.board
        if (not worker.busy) and (not paused) and (not board.is_game_over()) and time.time() >= next_dispatch_time:
            worker.start(current_agent(), depth, deterministic)

    def step_once():
        nonlocal next_dispatch_time
        if not worker.busy and not x_agent.board.is_game_over():
            worker.start(current_agent(), depth, deterministic)
            next_dispatch_time = 0.0

    while running:
        board = x_agent.board

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    depth = min(DEPTH_MAX, depth + DEPTH_STEP)
                elif event.key == pygame.K_DOWN:
                    depth = max(DEPTH_MIN, depth - DEPTH_STEP)
                elif event.key == pygame.K_d:
                    deterministic = not deterministic
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    step_once()
                elif event.key == pygame.K_r and not worker.busy:
                    x_agent.reset()
                    o_agent.reset()
                    next_dispatch_time = 0.0

        if worker.busy:
            move = worker.poll()
            if move is not None:
                x_agent.make_move(move)
                o_agent.make_move(move)
                next_dispatch_time = time.time() + SPECTATE_MOVE_DELAY
        else:
            dispatch_if_ready()

        screen.fill(BG)
        draw_board(screen, board, fonts, highlight_eligible=True)

        status_y = px_y(9) + 14
        if board.is_game_over():
            text, color = game_over_text(board)
            lines = [(text, color), ('[R] to play again', DIM_TEXT)]
        else:
            mover = 'X' if board.x_move else 'O'
            mover_name = x_agent.name if board.x_move else o_agent.name
            state = worker.thinking_label() if worker.busy else ('Paused' if paused else 'Waiting...')
            lines = [
                (f'{mover} to move — {mover_name}', TEXT_COLOR),
                (state, DIM_TEXT),
            ]
        lines.append((f'X = {x_agent.name}   O = {o_agent.name}', DIM_TEXT))
        lines.append((f'Search depth: {depth}    Mode: {"deterministic" if deterministic else "probabilistic"}',
                      DIM_TEXT))
        draw_text_lines(screen, fonts, lines, MARGIN, status_y)
        draw_legend(screen, fonts, MARGIN, WINDOW_H - 30, 'spectate')

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def make_fonts():
    return {
        'cell': pygame.font.SysFont('arial', 28, bold=True),
        'big': pygame.font.SysFont('arial', 90, bold=True),
        'status': pygame.font.SysFont('arial', 20),
        'legend': pygame.font.SysFont('arial', 14),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Ultimate Tic-Tac-Toe GUI')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    human = subparsers.add_parser('human', help='Play against a loaded network')
    human.add_argument('--network', default=NETWORK_PATH)
    human.add_argument('--human-color', choices=['x', 'o', 'random'], default='x')
    human.add_argument('--depth', type=int, default=100)

    spectate = subparsers.add_parser('spectate', help='Watch two networks play each other')
    spectate.add_argument('--network1', default=NETWORK_PATH)
    spectate.add_argument('--network2', default=NETWORK_PATH)
    spectate.add_argument('--depth', type=int, default=100)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'human':
        human_color = args.human_color
        if human_color == 'random':
            human_color = np.random.choice(['x', 'o'])
        run_human_mode(args.network, human_color, args.depth)
    else:
        run_spectate_mode(args.network1, args.network2, args.depth)


if __name__ == '__main__':
    main()
