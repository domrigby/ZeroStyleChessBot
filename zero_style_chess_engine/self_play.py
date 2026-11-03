"""Self-play agent process: plays games with an MCTS and sends the positions to the trainer as GameRecords."""

import chess
import time
from multiprocessing import Queue, Process, synchronize
from typing import List, Optional

from neural_nets.data.fen_chess_puzzles.curriculum import LichessCurriculum
from zero_style_chess_engine.config import SearchConfig
from zero_style_chess_engine.mcts import MCTS, START_STATE, Node
from zero_style_chess_engine.messages import GameRecord, GameStats, PositionRecord
import os

from util.parallel_error_log import error_logger
from util.chess_functions import save_chess_board_as_im


class SelfPlayWorker(Process):
    save_images_of_checkmates = False

    def __init__(
        self,
        save_dir: str,
        agent_id: int,
        stop_event: synchronize.Event,
        eval_request_queue: Optional[Queue] = None,
        game_record_queue: Optional[Queue] = None,
        eval_result_queue: Optional[Queue] = None,
        stats_queue: Optional[Queue] = None,
        start_state: str = START_STATE,
        sims: int = 1000,
        maximum_lag: int = 8,
        curriculum: Optional[LichessCurriculum] = None,
        search_config: Optional[SearchConfig] = None,
    ) -> None:

        super().__init__()

        self.stop_event = stop_event

        # Key of this agent's result queue in the evaluator
        self.agent_id = agent_id

        self.save_dir = save_dir
        self.log_file_name = os.path.join(save_dir, f"{self.__class__.__name__}_{self.agent_id}.txt")

        # Search budget per move
        self.sims = sims

        # Game length and temperature schedule (the search uses the rest of the config)
        self.search_config = search_config if search_config is not None else SearchConfig()

        # Self-play adds Dirichlet noise to the priors for exploration
        self.mcts = MCTS(
            eval_request_queue,
            eval_result_queue,
            agent_id=agent_id,
            stop_event=stop_event,
            search_config=self.search_config,
            add_noise=True,
            maximum_lag=maximum_lag,
        )

        # Local count for the number of memories we've saved
        self.saved_memory_local = 0

        self.game_record_queue = game_record_queue
        self.stats_queue = stats_queue

        self.start_state = str(start_state)

        # Puzzle curriculum for the first games; None plays every game from start_state
        self.curriculum = curriculum

    def _record_from(self, end_node: Node, stop_when_suboptimal: bool) -> None:
        """Send the positions leading to end_node to the trainer, valued by the game's outcome.

        :param stop_when_suboptimal: for end positions the search found off the played line. Walk back only while
            each move was its parent's best (highest Q), so a line found by exploration alone isn't kept, and stop
            before the root. Then drop to an even number of positions so wins and losses stay balanced. Otherwise
            (a played game) keep every position back to the root.
        """
        assert end_node.parent_node is not None and self.game_record_queue is not None

        # Get game winner
        if end_node.game_won:
            winner = end_node.parent_node.player
        else:
            winner = None

        game_states: List[PositionRecord] = []

        # Traverse up the tree, starting from the last position a move was played from
        position: Optional[Node] = end_node.parent_node

        while position is not None:
            if stop_when_suboptimal and (
                position.parent_node is None or position.parent_node.Qs.argmax() != position.move_idx
            ):
                break

            state = position.state
            moves = position.moves
            visit_counts = position.Ns

            if winner is not None:
                if position.player == winner:
                    value = 1
                else:
                    value = -1
            else:
                value = 0

            game_states.append(PositionRecord(fen=state, moves=moves, visit_counts=visit_counts, value=value))
            self.saved_memory_local += 1
            position = position.parent_node

        # Make sure its even to not get bias to wins
        if stop_when_suboptimal and len(game_states) % 2 != 0:
            game_states = game_states[:-1]

        if len(game_states) == 0:
            return

        self.game_record_queue.put(GameRecord(agent_id=self.agent_id, positions=game_states))

    def run(self) -> None:
        self.train()

        # Don't block exit flushing queues whose readers have already stopped
        for queue in (self.mcts.eval_request_queue, self.game_record_queue, self.stats_queue):
            if queue is not None:
                queue.cancel_join_thread()

    @error_logger
    def train(self, n_games: int = 1000000) -> None:

        game_count = 0

        print(f"Starting training run on agent {self.agent_id}")

        while not self.stop_event.is_set():
            # Start a new game
            move_count = 0
            main_board = chess.Board()

            if self.curriculum is not None:
                # This has some progressively more difficult chess puzzles
                start_state, max_length = self.curriculum.get_start_state(game_count)
                move_count = int(start_state.split()[-1])
                max_length = self.search_config.max_game_length
            else:
                start_state = self.start_state
                max_length = self.search_config.max_game_length

            # Reset main board to start state
            self.mcts.reset(start_state)
            main_board.set_fen(start_state)

            node = self.mcts.root
            player_check_mated: Optional[str] = None

            while not main_board.is_game_over() and not self.stop_event.is_set():
                start_time = time.time()
                end_nodes = self.mcts.search(number_of_expansions=self.sims)
                end_time = time.time()

                # A search cut short by the stop event may not have visited the root's moves: don't pick one
                if self.stop_event.is_set():
                    break

                for end_node in end_nodes:
                    # These are states which we didn't visit with the root node but were end games.
                    # In order to maintain high quality data we only save the move whilst the game was being played optimally
                    # I.e. if it was only found due to randomness then we don't count it as it a move a real player would never make
                    self._record_from(end_node, stop_when_suboptimal=True)

                print(f"\n\nTime with parallel search {end_time - start_time:.3f}s.")

                # TODO: sort out now visiting
                if move_count > self.search_config.tau_switch_move or (
                    self.curriculum is not None and game_count < self.search_config.curriculum_low_tau_games
                ):
                    tau = self.search_config.tau_late
                else:
                    tau = self.search_config.tau_early

                node, move, move_idx = self.mcts.select_child(tau=tau)
                chess_move = chess.Move.from_uci(move)
                main_board.set_fen(self.mcts.root.state)
                main_board.push(chess_move)

                print(f"Agent {self.agent_id} Game {game_count} Move: {move_count}")

                if main_board.is_checkmate():
                    player_check_mated = main_board.fen().split()[1]

                    if self.save_images_of_checkmates:
                        save_chess_board_as_im(
                            main_board, f"{self.save_dir}/agent_{self.agent_id}_game_{game_count}.svg", move=chess_move
                        )

                elif main_board.is_stalemate():
                    print("Stalemate!")

                elif main_board.is_insufficient_material():
                    print("Insufficient material to checkmate!")

                elif move_count >= max_length:
                    print("Maximum move length")
                    break
                else:
                    if main_board.is_check():
                        print("King is in check!")

                    self.mcts.advance_root(node, move_idx)

                    # Increment move count
                    move_count += 1

                # Save the played game once it's over. A game-ending move doesn't advance the root, so check the board
                # and the node just played, not the root. The engine also ends games python-chess doesn't (50 moves)
                if main_board.is_game_over() or node.branch_complete:
                    self._record_from(node, stop_when_suboptimal=False)
                    break

            if self.stop_event.is_set():
                break

            white_win: Optional[bool]
            if player_check_mated == "w":
                white_win = False
            elif player_check_mated == "b":
                white_win = True
            else:
                white_win = None

            if self.stats_queue is not None:
                self.send_update_to_main_thread(white_win, move_count)

            game_count += 1

    def send_update_to_main_thread(self, white_win: Optional[bool], game_length: int) -> None:
        assert self.stats_queue is not None
        self.stats_queue.put_nowait(GameStats(white_win=white_win, game_length=game_length))
