"""
This module contains class implementations for three environments:
- Rubik's Cube: Cube3:
- 2D Rubik's Cube: Cube2D:
- Permutation Group: PermutationGroup:

Note that we prioritize readability and reproducibility over speed optimization in this repository.

The Andrews-Curtis environment is implemented elsewhere




This module implements puzzle environments in two styles:

- Stateless: actions return new states
- Stateful: actions modify internal state

Cube3 and Cube2 are mixed between stateless and stateful.

The stateless style enables parallel processing, while stateful is simpler for sequential use.




"""

import random
import numpy as np
from torch_AC import *
from AC_env import *
class Cube3:
    """
    A class for 3x3x3 Rubik's Cube
    """
    def __init__(self):
        self.DTYPE = np.int64

        # Define initial and goal state
        self.reset()
        self.goal = np.arange(0, 9 * 6, dtype=self.DTYPE) // 9
        self.name = "Cube3"
        # Define moves
        ## faces and turns
        faces = ["U", "D", "L", "R", "B", "F"]
        ## [90 degrees clockwise, 90 degrees counter-clockwise]
        degrees = ["", "'"]
        degrees_inference = degrees[::-1]
        self.moves = [f"{f}{n}" for f in faces for n in degrees]
        self.moves_inference = [f"{f}{n}" for f in faces for n in degrees_inference]
        self.moves_ix_torch = torch.arange(len(self.moves), device = 'cuda' if torch.cuda.is_available() else 'cpu')
        self.inverse_moves_ix_torch = torch.tensor([1,0,3,2,5,4,7,6,9,8,11,10], device = 'cuda' if torch.cuda.is_available() else 'cpu')
        # Opposite faces
        self.pairing = {
            "R": "L",
            "L": "R",
            "F": "B",
            "B": "F",
            "U": "D",
            "D": "U",
        }
        # Prohibit obviously redundant moves.
        self.moves_available_after = {
            m: [v for v in self.moves if v[0] != m[0]] + [m] 
            for m in self.moves
        } # self-cancelling moves on the same face

        # [OPTIMIZATION] slicing by move string (e.g., R', U, F) => indices (e.g., 2, 6, 1)
        self.moves_ix = [self.moves.index(m) for m in self.moves]
        self.moves_ix_available_after = {
            self.moves.index(m): [self.moves.index(m) for m in available_moves]
            for m, available_moves in self.moves_available_after.items()
        }
        self.moves_ix_inference = [self.moves.index(m) for m in self.moves_inference]
        
        self.pairing_ix = {
            0: 1,
            1: 0,
            2: 3,
            3: 2,
            4: 5,
            5: 4,
        } # Points to the opposite face index

        # Vectorize the sticker group replacement operations
        self.__vectorize_moves()
        self.num_moves=len(self.moves)
        self.input_dim= 9*6*6
        self.state_dim = 9*6
        self.num_classes = 6# classes in state representation
    def reset(self):
        """Resets the cube state to the solved state."""
        self.state = np.arange(0, 9 * 6, dtype=self.DTYPE) // 9
        

    def is_solved(self):
        """Checks if the cube is in the solved state."""
        return np.all(self.state == self.goal)

    def finger(self, move):
        """Applies a single move on the cube state using move string."""
        self.state[self.sticker_target[move]] = self.state[self.sticker_source[move]]

    def finger_ix(self, ix):
        """The same `finger` method **but using indices of moves for faster execution"""
        self.state[self.sticker_target_ix[ix]] = self.state[self.sticker_source_ix[ix]]

    #def finger_ix_vec(self, ix):
    #    """The same `finger` method **but using indices of moves for faster execution"""
    #    self.state[:,self.sticker_target_ix[ix]] = self.state[:,self.sticker_source_ix[ix]]

    def apply_scramble(self, scramble):
        """Applies a sequence of moves (scramble) to the cube state."""
        if isinstance(scramble, str):
            scramble = scramble.split()
        for m in scramble:
            if m[-1]=='2':
                for _ in range(2):
                    self.finger(m[0])
            else:
                    self.finger(m)

    def scrambler(self, scramble_length):
        """
        Generates a random scramble of given length and returns the cube state and scramble moves as a generator.
        Please note that index-based implementations (faster) follow commented lexical logics.
        """
        while True:
            # Reset the cube state, scramble, and return cube state and scramble moves
            self.reset()
            scramble = []

            for i in range(scramble_length):
                if i:
                    last_move = scramble[-1]
                    if i > 1:   # [3rd~ moves]
                        while True:
                            # move = random.choice(self.moves_available_after[last_move])
                            move = random.choice(self.moves_ix_available_after[last_move])

                            if scramble[-2] == last_move == move:
                                # Three subsequent moves on the same face, which could be one
                                continue
                            # elif (
                            #     scramble[-2][0] == move[0] and len(scramble[-2] + move) == 3
                            #     and last_move[0] == self.pairing[move[0]]
                            # ):
                            elif (
                                scramble[-2]//2 == move//2 and scramble[-2]%2 != move%2
                                and last_move//2 == self.pairing_ix[move//2]
                            ):
                                # Two mutually canceling moves sandwiching an opposite face move
                                continue
                            else:
                                break
                    else:       # [2nd move]
                        # move = random.choice(self.moves_available_after[last_move])
                        move = random.choice(self.moves_ix_available_after[last_move])
                else:           # [1st move]
                    # move = random.choice(self.moves)
                    move = random.choice(self.moves_ix)

                # self.finger(move)
                self.finger_ix(move)
                scramble.append(move)

                yield self.state, move


    def __vectorize_moves(self):
        """
        Vectorizes the sticker group replacement operations for faster computation.
        This method defines ```self.sticker_target``` and ```self.sticker_source``` to manage sticker colors (target is replaced by source).
        They define indices of target and source stickers so that the moves can be vectorised.

        Colors:

                0 0 0
                0 0 0
                0 0 0
        2 2 2   5 5 5   3 3 3   4 4 4
        2 2 2   5 5 5   3 3 3   4 4 4
        2 2 2   5 5 5   3 3 3   4 4 4
                1 1 1
                1 1 1
                1 1 1

        Order of stickers on each face:

             2   5   8
             1   4   7
            [0]  3   6

        Indices of state (each starting with 9*(n-1)):

                         2   5   8
                         1   4   7
                        [0]  3   6
             20  23 26  47  50  53  29  32 35  38  41 44
             19  22 25  46  49  52  28  31 34  37  40 43
            [18] 21 24 [45] 48  51 [27] 30 33 [36] 39 42
                        11   14 17
                        10   13 16
                        [9]  12 15
        """
        self.sticker_target, self.sticker_source = dict(), dict()

        self.sticker_replacement = {
            # Sticker A is replaced by another sticker at index B -> A:B
            'U':{0: 6, 1: 3, 2: 0, 3: 7, 5: 1, 6: 8, 7: 5, 8: 2, 20: 47, 23: 50, 26: 53, 29: 38, 32: 41, 35: 44, 38: 20, 41: 23, 44: 26, 47: 29, 50: 32, 53: 35},
            'D':{9: 15, 10: 12, 11: 9, 12: 16, 14: 10, 15: 17, 16: 14, 17: 11, 18: 36, 21: 39, 24: 42, 27: 45, 30: 48, 33: 51, 36: 27, 39: 30, 42: 33, 45: 18, 48: 21, 51: 24},
            'L':{0: 44, 1: 43, 2: 42, 9: 45, 10: 46, 11: 47, 18: 24, 19: 21, 20: 18, 21: 25, 23: 19, 24: 26, 25: 23, 26: 20, 42: 11, 43: 10, 44: 9, 45: 0, 46: 1, 47: 2},
            'R':{6: 51, 7: 52, 8: 53, 15: 38, 16: 37, 17: 36, 27: 33, 28: 30, 29: 27, 30: 34, 32: 28, 33: 35, 34: 32, 35: 29, 36: 8, 37: 7, 38: 6, 51: 15, 52: 16, 53: 17},
            'B':{2: 35, 5: 34, 8: 33, 9: 20, 12: 19, 15: 18, 18: 2, 19: 5, 20: 8, 33: 9, 34: 12, 35: 15, 36: 42, 37: 39, 38: 36, 39: 43, 41: 37, 42: 44, 43: 41, 44: 38},
            'F':{0: 24, 3: 25, 6: 26, 11: 27, 14: 28, 17: 29, 24: 17, 25: 14, 26: 11, 27: 6, 28: 3, 29: 0, 45: 51, 46: 48, 47: 45, 48: 52, 50: 46, 51: 53, 52: 50, 53: 47}
        }
        for m in self.moves:
            if len(m) == 1:
                assert m in self.sticker_replacement
            else:
                if "'" in m:
                    self.sticker_replacement[m] = {
                        v: k for k, v in self.sticker_replacement[m[0]].items()
                    }
                elif "2" in m:
                    self.sticker_replacement[m] = {
                        k: self.sticker_replacement[m[0]][v]
                        for k, v in self.sticker_replacement[m[0]].items()
                    }
                else:
                    raise

            self.sticker_target[m] = list(self.sticker_replacement[m].keys())
            self.sticker_source[m] = list(self.sticker_replacement[m].values())

            for i, idx in enumerate(self.sticker_target[m]):
                assert self.sticker_replacement[m][idx] == self.sticker_source[m][i]

        # For index slicing
        self.sticker_target_ix = np.array([np.array(self.sticker_target[m]) for m in self.moves])
        self.sticker_source_ix = np.array([np.array(self.sticker_source[m]) for m in self.moves])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sticker_target_ix_torch = torch.tensor(self.sticker_target_ix, dtype=torch.long, device=device)
        self.sticker_source_ix_torch = torch.tensor(self.sticker_source_ix, dtype=torch.long, device=device)



import torch
import os
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import math
# Assume Cube3 and TrainConfig are defined elsewhere
class ScrambleGenerator(IterableDataset):
    def __init__(
            self,
            max_depth=1,#TrainConfig.max_depth,
            total_samples=1,#TrainConfig.num_steps * TrainConfig.batch_size_per_depth
            seed=None,
            env=Cube3()
        ):
        super(ScrambleGenerator, self).__init__()
        self.max_depth = max_depth
        self.total_samples = total_samples
        self.seed = seed
        self.env = env
        self.generator = self.env.scrambler(self.max_depth)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            iter_start = 0
            iter_end = self.total_samples
            worker_seed = self.seed
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.total_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.total_samples)
            worker_seed = self.seed + worker_id if self.seed is not None else None

        #if self.generator is None:
        #    self.env = Cube3()
        #    self.generator = self.env.scrambler(self.max_depth)

        if worker_seed is not None:
            random.seed(worker_seed)
            np.random.seed(worker_seed)

        for _ in range(iter_start, iter_end):
            X = np.zeros((self.max_depth, self.env.state_dim), dtype=int)
            y = np.zeros((self.max_depth,), dtype=int)
            for j in range(self.max_depth):
                state, last_move = next(self.generator)
                X[j, :] = state
                y[j] = last_move
            yield X, y


class PermutationGroup:
    def __init__(self, n):
        self.n = n
        self.name = "PermutationGroup"+str(n)
        self.moves = []
        for i in range(n - 1):
            self.moves.append(f'({i},{i+1})')
            self.moves.append(f'({i+1},{i})')
        self.identity = list(range(n))

        self.sticker_replacement = {}
        for i in range(n - 1):
            move = f'({i},{i+1})'
            replacement = {j: j for j in range(n)}
            replacement[i] = i + 1
            replacement[i + 1] = i
            self.sticker_replacement[move] = replacement
            
            # Add inverse permutation
            inverse_move = f'({i+1},{i})'
            inverse_replacement = {j: j for j in range(n)}
            inverse_replacement[i + 1] = i
            inverse_replacement[i] = i + 1
            self.sticker_replacement[inverse_move] = inverse_replacement

        self.sticker_target = {}
        self.sticker_source = {}

        for m in self.moves:
            # The source/target mechanism is used for efficient move application
            # 'source' represents the original positions of stickers
            # 'target' represents where those stickers should move to
            # For each move (including inverse permutations):
            # We create a mapping of target positions to source positions
            # For (i,i+1) or (i+1,i):
            # The sticker at position i becomes the source for the target at i+1 (or vice versa)
            # The sticker at position i+1 becomes the source for the target at i (or vice versa)
            # All other stickers remain in place (source = target)
     
            self.sticker_target[m] = list(self.sticker_replacement[m].keys())
            self.sticker_source[m] = list(self.sticker_replacement[m].values())

        # For index slicing
        self.sticker_target_ix = np.array([np.array(self.sticker_target[m]) for m in self.moves])
        self.sticker_source_ix = np.array([np.array(self.sticker_source[m]) for m in self.moves])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sticker_target_ix_torch = torch.tensor(self.sticker_target_ix, dtype=torch.long, device=device)
        self.sticker_source_ix_torch = torch.tensor(self.sticker_source_ix, dtype=torch.long, device=device)
    
        # print("target")
        # print(self.sticker_target_ix)
        # print("source")
        # print(self.sticker_source_ix)
        self.num_moves = len(self.moves)
        self.input_dim = self.n * self.n
        self.state_dim = self.n
        self.num_classes = self.n
        self.goal = self.identity

    def apply_move(self, state, move):
        return [self.sticker_replacement[move].get(x, x) for x in state]

    def scramble(self, num_moves):
        state = self.identity.copy()
        moves = []
        for _ in range(num_moves):
            move = np.random.choice(self.moves)
            state = self.apply_move(state, move)
            moves.append(move)
        return state, moves

    def is_solved(self, state):
        return np.all(state == self.identity)

    def scrambler(self, max_depth):
        while True:
            state = self.identity.copy()
            for depth in range(max_depth):
                move = np.random.choice(self.moves)
                state = self.apply_move(state, move)
                yield state, self.moves.index(move)

    def apply_move_fast(self, state, move_index):
        target = self.sticker_target_ix[move_index]
        source = self.sticker_source_ix[move_index]
        new_state = np.array(state).copy()
        state = np.array(state)
        new_state[target] = state[source]
        return new_state

    def apply_move_vec(self, state, move_index):
        target = self.sticker_target_ix[move_index]
        source = self.sticker_source_ix[move_index]
        new_state = np.array(state).copy()
        state = np.array(state)
        new_state[:, target] = state[:, source]
        return new_state

    def scramble_fast(self, num_moves):
        state = np.array(self.identity)
        moves = []
        for _ in range(num_moves):
            move_index = np.random.randint(len(self.moves))
            state = self.apply_move_fast(state, move_index)
            moves.append(self.moves[move_index])
        return state.tolist(), moves

    def scrambler_fast(self, max_depth):
        while True:
            state = np.array(self.identity)
            for depth in range(max_depth):
                move_index = np.random.randint(len(self.moves))
                state = self.apply_move_fast(state, move_index)
                yield state, move_index
class Cube2():
    def __init__(self):
        # move indices
        self.moveInds = { \
          "U": 0, "U'": 1, "U2": 2, "R": 3, "R'": 4, "R2": 5, "F": 6, "F'": 7, "F2": 8, \
          "D": 9, "D'": 10, "D2": 11, "L": 12, "L'": 13, "L2": 14, "B": 15, "B'": 16, "B2": 17, \
          "x": 18, "x'": 19, "x2": 20, "y": 21, "y'": 22, "y2": 23, "z": 24, "z'": 25, "z2": 26 \
        }


        # Move indices for basic moves (U, U', R, R', F, F', D, D', L, L', B, B')
        self.good_move_indices = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]
        self.moveIndsFixed = dict(zip([key for key,value in self.moveInds.items() if value in self.good_move_indices], list(range(len(self.good_move_indices)))))
        self.moves = [key  for key,value in self.moveInds.items() if value in self.good_move_indices]
        # move definitions
        self.moveDefs = np.array([ \
          [  2,  0,  3,  1, 20, 21,  6,  7,  4,  5, 10, 11, 12, 13, 14, 15,  8,  9, 18, 19, 16, 17, 22, 23], \
          [  1,  3,  0,  2,  8,  9,  6,  7, 16, 17, 10, 11, 12, 13, 14, 15, 20, 21, 18, 19,  4,  5, 22, 23], \
          [  3,  2,  1,  0, 16, 17,  6,  7, 20, 21, 10, 11, 12, 13, 14, 15,  4,  5, 18, 19,  8,  9, 22, 23], \
          [  0,  9,  2, 11,  6,  4,  7,  5,  8, 13, 10, 15, 12, 22, 14, 20, 16, 17, 18, 19,  3, 21,  1, 23], \
          [  0, 22,  2, 20,  5,  7,  4,  6,  8,  1, 10,  3, 12,  9, 14, 11, 16, 17, 18, 19, 15, 21, 13, 23], \
          [  0, 13,  2, 15,  7,  6,  5,  4,  8, 22, 10, 20, 12,  1, 14,  3, 16, 17, 18, 19, 11, 21,  9, 23], \
          [  0,  1, 19, 17,  2,  5,  3,  7, 10,  8, 11,  9,  6,  4, 14, 15, 16, 12, 18, 13, 20, 21, 22, 23], \
          [  0,  1,  4,  6, 13,  5, 12,  7,  9, 11,  8, 10, 17, 19, 14, 15, 16,  3, 18,  2, 20, 21, 22, 23], \
          [  0,  1, 13, 12, 19,  5, 17,  7, 11, 10,  9,  8,  3,  2, 14, 15, 16,  6, 18,  4, 20, 21, 22, 23], \
          [  0,  1,  2,  3,  4,  5, 10, 11,  8,  9, 18, 19, 14, 12, 15, 13, 16, 17, 22, 23, 20, 21,  6,  7], \
          [  0,  1,  2,  3,  4,  5, 22, 23,  8,  9,  6,  7, 13, 15, 12, 14, 16, 17, 10, 11, 20, 21, 18, 19], \
          [  0,  1,  2,  3,  4,  5, 18, 19,  8,  9, 22, 23, 15, 14, 13, 12, 16, 17,  6,  7, 20, 21, 10, 11], \
          [ 23,  1, 21,  3,  4,  5,  6,  7,  0,  9,  2, 11,  8, 13, 10, 15, 18, 16, 19, 17, 20, 14, 22, 12], \
          [  8,  1, 10,  3,  4,  5,  6,  7, 12,  9, 14, 11, 23, 13, 21, 15, 17, 19, 16, 18, 20,  2, 22,  0], \
          [ 12,  1, 14,  3,  4,  5,  6,  7, 23,  9, 21, 11,  0, 13,  2, 15, 19, 18, 17, 16, 20, 10, 22,  8], \
          [  5,  7,  2,  3,  4, 15,  6, 14,  8,  9, 10, 11, 12, 13, 16, 18,  1, 17,  0, 19, 22, 20, 23, 21], \
          [ 18, 16,  2,  3,  4,  0,  6,  1,  8,  9, 10, 11, 12, 13,  7,  5, 14, 17, 15, 19, 21, 23, 20, 22], \
          [ 15, 14,  2,  3,  4, 18,  6, 16,  8,  9, 10, 11, 12, 13,  1,  0,  7, 17,  5, 19, 23, 22, 21, 20], \
          [  8,  9, 10, 11,  6,  4,  7,  5, 12, 13, 14, 15, 23, 22, 21, 20, 17, 19, 16, 18,  3,  2,  1,  0], \
          [ 23, 22, 21, 20,  5,  7,  4,  6,  0,  1,  2,  3,  8,  9, 10, 11, 18, 16, 19, 17, 15, 14, 13, 12], \
          [ 12, 13, 14, 15,  7,  6,  5,  4, 23, 22, 21, 20,  0,  1,  2,  3, 19, 18, 17, 16, 11, 10,  9,  8], \
          [  2,  0,  3,  1, 20, 21, 22, 23,  4,  5,  6,  7, 13, 15, 12, 14,  8,  9, 10, 11, 16, 17, 18, 19], \
          [  1,  3,  0,  2,  8,  9, 10, 11, 16, 17, 18, 19, 14, 12, 15, 13, 20, 21, 22, 23,  4,  5,  6,  7], \
          [  3,  2,  1,  0, 16, 17, 18, 19, 20, 21, 22, 23, 15, 14, 13, 12,  4,  5,  6,  7,  8,  9, 10, 11], \
          [ 18, 16, 19, 17,  2,  0,  3,  1, 10,  8, 11,  9,  6,  4,  7,  5, 14, 12, 15, 13, 21, 23, 20, 22], \
          [  5,  7,  4,  6, 13, 15, 12, 14,  9, 11,  8, 10, 17, 19, 16, 18,  1,  3,  0,  2, 22, 20, 23, 21], \
          [ 15, 14, 13, 12, 19, 18, 17, 16, 11, 10,  9,  8,  3,  2,  1,  0,  7,  6,  5,  4, 23, 22, 21, 20]  \
        ])

        self.name = "Cube2"
        # piece definitions
        self.pieceDefs = np.array([ \
          [  0, 21, 16], \
          [  2, 17,  8], \
          [  3,  9,  4], \
          [  1,  5, 20], \
          [ 12, 10, 19], \
          [ 13,  6, 11], \
          [ 15, 22,  7], \
        ])

        # OP representation from (hashed) piece stickers
        self.pieceInds = np.zeros([58, 2], dtype=int)
        self.pieceInds[50] = [0, 0]; self.pieceInds[54] = [0, 1]; self.pieceInds[13] = [0, 2]
        self.pieceInds[28] = [1, 0]; self.pieceInds[42] = [1, 1]; self.pieceInds[ 8] = [1, 2]
        self.pieceInds[14] = [2, 0]; self.pieceInds[21] = [2, 1]; self.pieceInds[ 4] = [2, 2]
        self.pieceInds[52] = [3, 0]; self.pieceInds[15] = [3, 1]; self.pieceInds[11] = [3, 2]
        self.pieceInds[47] = [4, 0]; self.pieceInds[30] = [4, 1]; self.pieceInds[40] = [4, 2]
        self.pieceInds[25] = [5, 0]; self.pieceInds[18] = [5, 1]; self.pieceInds[35] = [5, 2]
        self.pieceInds[23] = [6, 0]; self.pieceInds[57] = [6, 1]; self.pieceInds[37] = [6, 2]

        # piece stickers from OP representation
        self.pieceCols = np.zeros([7, 3, 3], dtype=int)
        self.pieceCols[0, 0, :] = [0, 5, 4]; self.pieceCols[0, 1, :] = [4, 0, 5]; self.pieceCols[0, 2, :] = [5, 4, 0]
        self.pieceCols[1, 0, :] = [0, 4, 2]; self.pieceCols[1, 1, :] = [2, 0, 4]; self.pieceCols[1, 2, :] = [4, 2, 0]
        self.pieceCols[2, 0, :] = [0, 2, 1]; self.pieceCols[2, 1, :] = [1, 0, 2]; self.pieceCols[2, 2, :] = [2, 1, 0]
        self.pieceCols[3, 0, :] = [0, 1, 5]; self.pieceCols[3, 1, :] = [5, 0, 1]; self.pieceCols[3, 2, :] = [1, 5, 0]
        self.pieceCols[4, 0, :] = [3, 2, 4]; self.pieceCols[4, 1, :] = [4, 3, 2]; self.pieceCols[4, 2, :] = [2, 4, 3]
        self.pieceCols[5, 0, :] = [3, 1, 2]; self.pieceCols[5, 1, :] = [2, 3, 1]; self.pieceCols[5, 2, :] = [1, 2, 3]
        self.pieceCols[6, 0, :] = [3, 5, 1]; self.pieceCols[6, 1, :] = [1, 3, 5]; self.pieceCols[6, 2, :] = [5, 1, 3]

        # useful arrays for hashing
        self.hashOP = np.array([1, 2, 10])
        self.pow3 = np.array([1, 3, 9, 27, 81, 243])
        self.pow7 = np.array([1, 7, 49, 343, 2401, 16807])
        self.fact6 = np.array([720, 120, 24, 6, 2, 1])

        self.sticker_source_ix = self.moveDefs[self.good_move_indices]
        self.sticker_target_ix = np.array([np.arange(24) for _ in self.good_move_indices])
        self.num_moves = len(self.good_move_indices)
        self.input_dim = 4*6*6
        self.state_dim = 4*6
        self.num_classes=6
        self.goal = self.initState()

    def initState(self):
        return np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])

    def doMove(self, s, move):
        #return s[self.sticker_source_ix[move]]
        s[self.sticker_target_ix[move]] = s[self.sticker_source_ix[move]]
        return s

    def doAlgStr(self, s, moveslist):
        moves = moveslist#alg.split(" ")
        for m in moves:
            if m in self.moveInds:
                s = self.doMove(s, self.moveIndsFixed[m])
        return s

    def isSolved(self, s):
        for i in range(6):
            if not (s[4 * i:4 * i + 4] == s[4 * i]).all():
                return False
        return True

    def normFC(self, s):
        normCols = np.zeros(6, dtype=int)
        normCols[s[18] - 3] = 1
        normCols[s[23] - 3] = 2
        normCols[s[14]] = 3
        normCols[s[18]] = 4
        normCols[s[23]] = 5
        return normCols[s]

    def getOP(self, s):
        return self.pieceInds[np.dot(s[self.pieceDefs], self.hashOP)]

    def getStickers(self, sOP):
        s = np.zeros(24, dtype=int)
        s[[14, 18, 23]] = [3, 4, 5]
        for i in range(7):
            s[self.pieceDefs[i]] = self.pieceCols[sOP[i, 0], sOP[i, 1], :]
        return s

    def indexO(self, sOP):
        return np.dot(sOP[:-1, 1], self.pow3)

    def indexP(self, sOP):
        return np.dot(sOP[:-1, 0], self.pow7)

    def indexP2(self, sOP):
        return np.dot([sOP[i, 0] - np.count_nonzero(sOP[:i, 0] < sOP[i, 0]) for i in range(6)], self.fact6)

    def indexOP(self, sOP):
        return self.indexO(sOP) * 5040 + self.indexP2(sOP)

    def printCube(self, s):
        print("      ┌──┬──┐")
        print("      │ {}│ {}│".format(s[0], s[1]))
        print("      ├──┼──┤")
        print("      │ {}│ {}│".format(s[2], s[3]))
        print("┌──┬──┼──┼──┼──┬──┬──┬──┐")
        print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(s[16], s[17], s[8], s[9], s[4], s[5], s[20], s[21]))
        print("├──┼──┼──┼──┼──┼──┼──┼──┤")
        print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(s[18], s[19], s[10], s[11], s[6], s[7], s[22], s[23]))
        print("└──┴──┼──┼──┼──┴──┴──┴──┘")
        print("      │ {}│ {}│".format(s[12], s[13]))
        print("      ├──┼──┤")
        print("      │ {}│ {}│".format(s[14], s[15]))
        print("      └──┴──┘")
    
    def reset(self):
        self.state = self.initState()
    
    def apply_scramble(self, scramble):
        self.state = self.doAlgStr(self.state, scramble)
    
    def finger_ix(self, move):
        self.state[self.sticker_target_ix[move]] = self.state[self.sticker_source_ix[move]]
        #
        #return self.doMove(self.state, move)
    def finger(self,move_name):
        self.state = self.doMove(self.state, self.moveIndsFixed[move_name])

    def scrambler(self, max_depth):
        while True:
            self.state = self.initState()
            for depth in range(max_depth):
                move_index = np.random.randint(self.num_moves)
                self.finger_ix(move_index)
                yield self.state, move_index

    

    def is_solved(self):
        """Checks if the cube is in the solved state."""
        return np.all(self.state == self.goal)




if __name__ == "__main__":
    # Create a cube instance
    cube = Cube2()
    cube.reset()
    # Print initial solved state
    print("Initial solved state:")
    cube.printCube(cube.state)
    
    # Apply some moves and print result
    scramble = "x y R U' R' U' F2 U' R U R' U F2"
    scramble = "R"
    print(f"\nAfter applying {scramble}:")
    cube.apply_scramble(scramble)
    cube.printCube(cube.state)
    # Test scrambler generator
    print("\nTesting scrambler generator:")
    max_depth = 3
    generator = cube.scrambler(max_depth)
    for i in range(5):  # Generate 5 samples
        state, move_index = next(generator)
        print(f"Sample {i}:")
        cube.printCube(state)
        print(f"Move applied: {cube.moveDefs[move_index]}\n")
    
    # Normalize and print result
    # print("\nAfter normalizing:")
    # cube.state = cube.normFC(cube.state)
    # cube.printCube(cube.state)


# class SL2Fp:
#     def __init__(self, p):
#         self.p = p
#         self.moves = np.array([[[1,1],[0,1]],[[1,-1],[0,1]],[[1,0],[1,1]],[[1,0],[-1,1]]])
#         # for i in range(n - 1):
#         #     self.moves.append(f'({i},{i+1})')
#         #     self.moves.append(f'({i+1},{i})')
#         self.identity = np.eye(2)

#         self.num_moves = len(self.moves)
#         self.input_dim = 2**2
#         self.state_dim = 4*p
#         self.num_classes = self.n
#         self.goal = self.identity

#     def apply_move(self, state, move):
#         return [self.sticker_replacement[move].get(x, x) for x in state]

#     def scramble(self, num_moves):
#         state = self.identity.copy()
#         moves = []
#         for _ in range(num_moves):
#             move = np.random.choice(self.moves)
#             state = self.apply_move(state, move)
#             moves.append(move)
#         return state, moves

#     def is_solved(self, state):
#         return np.all(state == self.identity)

#     def scrambler(self, max_depth):
#         while True:
#             state = self.identity.copy()
#             for depth in range(max_depth):
#                 move = np.random.choice(self.moves)
#                 state = self.apply_move(state, move)
#                 yield state, self.moves.index(move)

#     def apply_move_fast(self, state, move_index):
#         target = self.sticker_target_ix[move_index]
#         source = self.sticker_source_ix[move_index]
#         new_state = np.array(state).copy()
#         state = np.array(state)
#         new_state[target] = state[source]
#         return new_state

#     def apply_move_vec(self, state, move_index):
#         target = self.sticker_target_ix[move_index]
#         source = self.sticker_source_ix[move_index]
#         new_state = np.array(state).copy()
#         state = np.array(state)
#         new_state[:, target] = state[:, source]
#         return new_state

#     def scramble_fast(self, num_moves):
#         state = np.array(self.identity)
#         moves = []
#         for _ in range(num_moves):
#             move_index = np.random.randint(len(self.moves))
#             state = self.apply_move_fast(state, move_index)
#             moves.append(self.moves[move_index])
#         return state.tolist(), moves

#     def scrambler_fast(self, max_depth):
#         while True:
#             state = np.array(self.identity)
#             for depth in range(max_depth):
#                 move_index = np.random.randint(len(self.moves))
#                 state = self.apply_move_fast(state, move_index)
#                 yield state, move_index

if __name__ == "__main__":
    #Example usage:
    perm_group = PermutationGroup(10)  # S^5 permutation group
    scrambled_state, moves = perm_group.scramble(3)
    print(f"Scrambled state: {scrambled_state}")
    print(f"Moves applied: {moves}")
    
    # Test the faster versions of scrambling and move application
    
    import time
    
    # Create an instance of the PermutationGroup
    perm_group = PermutationGroup(5)  # S^5 permutation group
    
    # Test apply_move_fast
    print("Testing apply_move_fast:")
    state = perm_group.identity.copy()
    move_index = 0  # First move in the list
    new_state = perm_group.apply_move_fast(state, move_index)
    print(f"Original state: {state}")
    print(f"After applying move {perm_group.moves[move_index]}: {new_state}")
    
    # Test scramble_fast
    print("\nTesting scramble_fast:")
    num_moves = 5
    scrambled_state, moves = perm_group.scramble_fast(num_moves)
    print(f"Scrambled state: {scrambled_state}")
    print(f"Moves applied: {moves}")
    
    # Test scrambler_fast
    print("\nTesting scrambler_fast:")
    max_depth = 3
    generator = perm_group.scrambler_fast(max_depth)
    for i in range(5):  # Generate 5 samples
        state, move_index = next(generator)
        print(f"Sample {i}: State (one-hot) = {state}, Move = {perm_group.moves[move_index]}")
    
    # Benchmark comparison
    print("\nBenchmarking:")
    
    def benchmark_original(n_samples):
        start_time = time.time()
        generator = perm_group.scrambler(max_depth)
        for _ in range(n_samples):
            next(generator)
        return time.time() - start_time
    
    def benchmark_fast(n_samples):
        start_time = time.time()
        generator = perm_group.scrambler_fast(max_depth)
        for _ in range(n_samples):
            next(generator)
        return time.time() - start_time
    
    n_samples = 10000
    original_time = benchmark_original(n_samples)
    fast_time = benchmark_fast(n_samples)
    
    print(f"Time taken for {n_samples} samples:")
    print(f"Original version: {original_time:.4f} seconds")
    print(f"Fast version: {fast_time:.4f} seconds")
    print(f"Speedup: {original_time / fast_time:.2f}x")




#TORCH

@torch.jit.script
def apply_move_vec(state: torch.Tensor, move_index: int, sticker_source_ix: torch.Tensor, sticker_target_ix: torch.Tensor) -> torch.Tensor:
    target = sticker_target_ix[move_index]
    source = sticker_source_ix[move_index]
    new_state = state.clone()
    new_state[:, target] = state[:, source]
    return new_state

# @torch.jit.script
# def BAD_apply_list_of_moves_to_states(states: torch.Tensor, moves: torch.Tensor, sticker_source_ix: torch.Tensor, sticker_target_ix: torch.Tensor) -> torch.Tensor:
#     # Get all source and target indices for the moves at once
#     target = sticker_target_ix[moves]  # Shape: [num_moves, target_indices_per_move]
#     source = sticker_source_ix[moves]  # Shape: [num_moves, source_indices_per_move]
    
#     # Clone states before modifying to avoid warning about expanded tensors
#     states_clone = states.clone()
#     states_clone[:, target] = states[:, source]
#     return states_clone

@torch.jit.script
def apply_list_of_moves_to_states(states: torch.Tensor, moves: torch.Tensor, sticker_source_ix: torch.Tensor, sticker_target_ix: torch.Tensor) -> torch.Tensor:
    # Get all source and target indices for the moves at once
    target = sticker_target_ix[moves]  # Shape: [batch_size, target_indices_per_move]
    source = sticker_source_ix[moves]  # Shape: [batch_size, source_indices_per_move]
    
    # Clone states before modifying to avoid warning about expanded tensors
    states_clone = states.clone()
    
    # Create batch indices that match the expanded target/source shapes
    batch_idx = torch.arange(states.shape[0], device=states.device)[:, None].expand(-1, target.shape[1])
    
    # Use advanced indexing with properly shaped batch indices
    states_clone[batch_idx.flatten(), target.flatten()] = states[batch_idx.flatten(), source.flatten()]
    return states_clone

@torch.jit.script
def apply_all_moves_to_all_states(x,STICKER_SOURCE_IX,STICKER_TARGET_IX):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Ensure x is on the correct device
    #x = x.to(device)
    
    batch_size, num_features = x.shape
    num_moves = STICKER_SOURCE_IX.shape[0]
    
    # Create a tensor of all possible moves
    #all_moves = torch.arange(num_moves, device=device,dtype=int)
    
    # Initialize the result tensor
    result = torch.zeros((batch_size, num_moves, num_features), dtype=torch.long,device=x.device)# device=device,dtype=int)
    
    # Apply each move to all states in the batch
    for move in range(num_moves):
        # Use apply_move for each move across the entire batch
        result[:, move, :] = apply_move_vec(x, move,STICKER_SOURCE_IX,STICKER_TARGET_IX)   
    return result # which has structure (batch, move, state)

@torch.jit.script
def apply_all_moves_to_all_states_no_reshape(x: torch.Tensor,out_tensor: torch.Tensor,STICKER_SOURCE_IX: torch.Tensor,STICKER_TARGET_IX: torch.Tensor):
    batch_size, num_features = x.shape
    num_moves = STICKER_SOURCE_IX.shape[0]
    #result = torch.zeros((batch_size* num_moves, num_features), dtype=torch.long, device=x.device)
    out_tensor[:batch_size*num_moves,:] = x.tile(num_moves,1)
    for move in range(num_moves):
        out_tensor[move*batch_size:(move+1)*batch_size, STICKER_TARGET_IX[move]] = x[:, STICKER_SOURCE_IX[move]]
    return out_tensor



def generate_trajectory_torch_batch_cube(batch_size,scramble_length, env,device,start_state, return_apply_all,fwd_diffusion_function, compute_all_states=True, apply_time = False, start_states_in_ball = 0):
    """
    Generates a single batch of scrambled states and their corresponding moves.

    Args:
        batch_size (int): Number of trajectories to generate in parallel
        scramble_length (int): Length of each scramble sequence
        env: The cube environment
        device (str): Device to place tensors on ('cpu' or 'cuda')
        start_state (torch.Tensor): Initial state to start scrambling from
        return_apply_all (bool): Whether to return all possible next states
        fwd_diffusion_function (callable): Function that computes move probabilities for each state (including masking bad moves)

    Returns:
        tuple:
            - states (torch.Tensor): States for each step of each trajectory (batch_size, scramble_length+1, state_size)
            - moves (torch.Tensor): Moves taken at each step (batch_size, scramble_length)
            - all_next_states (torch.Tensor or None): If return_apply_all is True, contains all possible next states
    """
    # Initialize tensors to store batch
    states = torch.zeros((batch_size, scramble_length+1, env.state.size), device=device, dtype=start_state.dtype)
    moves = torch.zeros((batch_size, scramble_length), dtype=torch.long, device=device)
    if return_apply_all:
        all_next_states = torch.zeros((batch_size, scramble_length+1, env.num_moves, env.state.size), device=device, dtype=start_state.dtype)
        
    else:
        all_next_states = None
    if start_states_in_ball == 0:
        states[:,0] = start_state.clone()
        current_states = start_state.clone().expand(batch_size, -1)
    else:
        # For each state, randomly choose how many moves to apply (0 to start_states_in_ball)
        num_moves_per_state = torch.randint(0, start_states_in_ball + 1, (batch_size,), device=device)
        current_states = start_state.clone().expand(batch_size, -1)
        
        # For states that need moves, generate and apply them
        max_moves = num_moves_per_state.max()
        if max_moves > 0:
            moves_to_apply = torch.randint(0, env.num_moves, (batch_size, max_moves), device=device)
            for i in range(max_moves):
                # Only apply moves to states that need this many moves
                mask = i < num_moves_per_state
                if mask.any():
                    # Clone before indexing to avoid warning on expanded tensor
                    current_states = current_states.clone()
                    current_states[mask] = apply_list_of_moves_to_states(
                        current_states[mask], 
                        moves_to_apply[mask,i], 
                        env.sticker_source_ix_torch, 
                        env.sticker_target_ix_torch
                    )
        states[:,0] = current_states.clone()

    # Generate scrambles for all items in batch simultaneously
    #if return_apply_all:
    #    all_next_states[:,0,:,:] = apply_all_moves_to_all_states(current_states,env.sticker_source_ix_torch,env.sticker_target_ix_torch)
    
    for s in range(scramble_length):
        # Compute weights for all states in batch using vectorized computation
        #weights = compute_weights_from_state_vec(current_states, env, weight_contraction, total_relator_weight, double_weight=double_weight)
        if apply_time:
            time_input = torch.tensor([(s)/scramble_length], device=device).expand(batch_size)
        if return_apply_all and not compute_all_states:
            all_next_states[:,s,:,:] = apply_all_moves_to_all_states(current_states,env.sticker_source_ix_torch,env.sticker_target_ix_torch)
        if compute_all_states:
            if apply_time:
                weights_BM, all_next_states_BMS = fwd_diffusion_function(current_states,time_input)# all next states of current_states, which is [:,s]. so should go in [s]
            else:
                weights_BM, all_next_states_BMS = fwd_diffusion_function(current_states)
            batch_moves_B = torch.multinomial(weights_BM, num_samples=1).squeeze(-1)
            current_states = all_next_states_BMS[torch.arange(batch_size),batch_moves_B,:]
        else:
            if apply_time:
                weights_BM = fwd_diffusion_function(current_states,time_input)
            else:
                weights_BM = fwd_diffusion_function(current_states)
            #all_next_states_BMS = apply_all_moves_to_all_states(current_states.reshape(-1, env.state_dim),STICKER_SOURCE_IX,STICKER_TARGET_IX).view(-1, env.num_moves, env.state_dim)
            batch_moves_B = torch.multinomial(weights_BM, num_samples=1).squeeze(-1)
            current_states = apply_list_of_moves_to_states(current_states, batch_moves_B, env.sticker_source_ix_torch, env.sticker_target_ix_torch)
        if return_apply_all and compute_all_states:
            all_next_states[:,s,:,:] = all_next_states_BMS
        # Apply moves to all states at once
        #next_states_BS = finger_ix_fast_vec_torch_list_of_moves(current_states, batch_moves_B)
        # Store results
        
        states[:,s+1] = current_states
        moves[:,s] = batch_moves_B

        
    if return_apply_all:# at the end
        all_next_states[:,scramble_length,:,:] = apply_all_moves_to_all_states(current_states,env.sticker_source_ix_torch,env.sticker_target_ix_torch)
        all_next_states = all_next_states
    else:
        all_next_states = None
    return states.detach(), moves.detach(), all_next_states.detach()


# def apply_all_moves_to_all_states_base(x: torch.Tensor,
#                                      STICKER_SOURCE_IX: torch.Tensor,
#                                      STICKER_TARGET_IX: torch.Tensor):
#     batch_size, num_features = x.shape
#     num_moves = STICKER_SOURCE_IX.shape[0]
#     result = torch.zeros((batch_size, num_moves, num_features), dtype=torch.long, device=x.device)
#     for move in range(num_moves):
#         result[:, move, :] = apply_move_vec(x, move, STICKER_SOURCE_IX, STICKER_TARGET_IX)
#     return result

# # Create traced version with static STICKER arrays
# apply_all_moves_to_all_states = torch.jit.trace(
#     apply_all_moves_to_all_states_base,
#     (
#         torch.zeros((1, 24), dtype=torch.long),  # Example input shape
#         STICKER_SOURCE_IX,
#         STICKER_TARGET_IX
#     )
# )
#     #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # Ensure x is on the correct device
#     #x = x.to(device)