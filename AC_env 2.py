"""
This module contains class implementations for three environments:
- Rubik's Cube: Cube3
- 15 Puzzle: Puzzle15
- Lights Out: LightsOut7

Please note that we prioritize readability and reproducibility over speed optimization in this repository.
"""

import copy
import random
import numpy as np
import torch_AC

class AC_presentation:
    """
    A class for 3x3x3 Rubik's Cube
    """
    def __init__(self,max_relator_length=25):
        self.name = "AC_presentation_" + str(max_relator_length)
        self.DTYPE = int

        # Define initial and goal state
        self.n_gen = 2
        self.max_relator_length = max_relator_length
        self.zero_state = self.initState()
        self.solved_state = self.initState()
        self.goal = self.initState()

        # Define 
        self.moves = range(12)

        self.inverse_moves = np.array([2,3,0,1,8,9,10,11,4,5,6,7])

        # Vectorize the sticker group replacement operations

        self.num_moves=len(self.moves)
        self.input_dim= self.max_relator_length*self.n_gen*(self.n_gen*2+1)
        self.state_dim = self.max_relator_length*self.n_gen
        self.num_classes = 1+2*self.n_gen# classes in state representation
        self.reset()

    def initState(self):
        zero_state =np.zeros(self.max_relator_length*2,dtype=self.DTYPE)
        zero_state[0]=1
        zero_state[self.max_relator_length]=2
        return zero_state

    def reset(self):
        """Resets the cube state to the solved state."""
        self.state = self.initState()
        return self.initState()

    def is_solved(self):
        """Checks if the cube is in the solved state."""
        return np.all(self.state == self.goal)

    def split_into_each_generator(self):
        return self.state[:self.max_relator_length], self.state[self.max_relator_length:]

    # def finger(self, move):
    #     """Applies a single move on the cube state using move string."""
    #     self.state[self.sticker_target[move]] = self.state[self.sticker_source[move]]

    # def finger_ix(self, ix):
    #     """The same `finger` method **but using indices of moves for faster execution"""
    #     self.state[self.sticker_target_ix[ix]] = self.state[self.sticker_source_ix[ix]]

    #def finger_ix_vec(self, ix):
    #    """The same `finger` method **but using indices of moves for faster execution"""
    #    self.state[:,self.sticker_target_ix[ix]] = self.state[:,self.sticker_source_ix[ix]]

    def apply_scramble(self, scramble):
        """Applies a sequence of moves (scramble) to the cube state."""
        # if isinstance(scramble, str):
        # #     scramble = scramble.split()
        # for m in scramble:
        #     if m[-1]=='2':
        #         for _ in range(2):
        #             self.finger(m[0])
        #     else:
        #             self.finger(m)
        for m in scramble:
            self.finger_ix(m)

    def finger_ix(self, ix,simplify=True):
        #  0. r_1 --> r_1 r_0
        # 1. r_0 --> r_0 r_1^{-1}
        # 2. r_1 --> r_1 r_0^{-1}
        # 3. r_0 --> r_0 r_1
        # 4: r_1 --> x_0^{-1} r_1 x_0
        # 5: r_0 ---> x_1^{-1} r_0 x_1
        # 6: r_1 --> x_1^{-1} r_1 x_1
        # 7: r_0 ---> x_0 r_0 x_0^{-1}
        # 8: r_1 --> x_0 r_1 x_0^{-1}
        # 9: r_0 --> x_1 r_0 x_1^{-1}
        # 10: r_1 --> x_1 r_1 x_1^{-1}
        # 11: r_0 --> x_0^{-1} r_0 x_0
        # Check if state is all zeros and raise error if so
        r0,r1 = self.split_into_each_generator()
        if np.all(r0 == 0) or np.all(r1 == 0):
            raise ValueError("relators cannot be all zeros: ",r0,r1)
        # Find first zero in each generator representation using argmax
        first_zero_r0 = np.argmax(r0 == 0) or len(r0)
        first_zero_r1 = np.argmax(r1 == 0) or len(r1)

        len_nonzero_r0=first_zero_r0
        len_nonzero_r1=first_zero_r1
        r0_good = r0[:len_nonzero_r0]
        r1_good = r1[:len_nonzero_r1]
        r0_inverse = np.flip(-r0_good)
        r1_inverse = np.flip(-r1_good)


        combineable = (len_nonzero_r0 + len_nonzero_r1)<=self.max_relator_length
        combineable_r0 = (len_nonzero_r0+2) <=self.max_relator_length
        combineable_r1 = (len_nonzero_r1+2)<=self.max_relator_length
        if ix ==0 and combineable:
            r1 = np.concatenate([r1_good,r0_good])
        elif ix ==1 and combineable:
            r0 = np.concatenate([r0_good,r1_inverse])
        elif ix ==2 and combineable:
            r1 = np.concatenate([r1_good,r0_inverse])
        elif ix ==3 and combineable:
            r0 = np.concatenate([r0_good,r1_good])
        elif ix==4 and combineable_r1:
            r1  = np.concatenate([[-1],r1_good,[1]])
        elif ix==5 and combineable_r0:
            r0 = np.concatenate([[-2],r0_good,[2]])
        elif ix==6 and combineable_r1:
            r1 =  np.concatenate([[-2],r1_good,[2]])
        elif ix==7 and combineable_r0:
            r0 = np.concatenate([[1],r0_good,[-1]])
        elif ix==8 and combineable_r1:
            r1 = np.concatenate([[1],r1_good,[-1]])
        elif ix==9 and combineable_r0:
            r0 = np.concatenate([[2],r0_good,[-2]])
        elif ix==10 and combineable_r1:
            r1 = np.concatenate([[2],r1_good,[-2]])
        elif ix==11 and combineable_r0:
            r0 = np.concatenate([[-1],r0_good,[1]])
        else:
            r0 = r0_good
            r1 = r1_good
        if len(r0)>self.max_relator_length or len(r1)>self.max_relator_length:
            raise ValueError("Relator length exceeded")
        # Pad r0 and r1 with zeros to reach max_relator_length
        r0 = np.pad(r0, (0, self.max_relator_length - len(r0)), mode='constant', constant_values=0)
        r1 = np.pad(r1, (0, self.max_relator_length - len(r1)), mode='constant', constant_values=0)
        self.state = np.concatenate([r0,r1])
        if simplify:
            #print("finger_ix: last state",self.state)
            self.simplify_state()
            #print("finger_ix: simplified state",self.state)


        
    

    
    
    def check_conditions_return_possibilities(self):
        """
        Returns a list of possible moves that can be performed on the current state.
        
        A move is possible if:
        1. For moves 0-3: The combined length of nonzero elements in r0 and r1 is <= max_relator_length
        2. For moves 4,6,8,10: The length of nonzero elements in r1 plus 2 is <= max_relator_length,
           or the endpoints of r1 match specific patterns
        3. For moves 5,7,9,11: The length of nonzero elements in r0 plus 2 is <= max_relator_length,
           or the endpoints of r0 match specific patterns
        
        Returns:
            list: List of integers representing possible moves that can be performed
        """
        r0,r1 = self.split_into_each_generator()
        first_zero_r0 = np.argmax(r0 == 0) or len(r0)   
        first_zero_r1 = np.argmax(r1 == 0) or len(r1)
        len_nonzero_r0=first_zero_r0
        len_nonzero_r1=first_zero_r1
        nonzero_r0 = r0[np.nonzero(r0)[0]]
        nonzero_r1 = r1[np.nonzero(r1)[0]]
        combineable = (len_nonzero_r0 + len_nonzero_r1)<=self.max_relator_length
        combineable_r0 = (len_nonzero_r0+2)<=self.max_relator_length
        combineable_r1 = (len_nonzero_r1+2)<=self.max_relator_length
        possible_moves = []
        if combineable:
             possible_moves += [0,1,2,3]
        if combineable_r1:
            possible_moves += [4,6,8,10]
        else:
            if np.any(nonzero_r1[[0,-1]] == [1,-1]):
                possible_moves += [4]
            elif np.any(nonzero_r1[[0,-1]] == [2,-2]):
                possible_moves += [6]
            elif np.any(nonzero_r1[[0,-1]] == [-1,1]):
                possible_moves += [8]
            elif np.any(nonzero_r1[[0,-1]] == [-2,2]):
                possible_moves += [10]
        if combineable_r0:
            possible_moves += [5,7,9,11]
        else:
            if np.any(nonzero_r0[[0,-1]] == [2,-2]):
                possible_moves += [5]
            elif np.any(nonzero_r0[[0,-1]] == [-1,1]):
                possible_moves += [7]
            elif np.any(nonzero_r0[[0,-1]] == [-2,2]):
                possible_moves += [9]
            elif np.any(nonzero_r0[[0,-1]] == [1,-1]):
                possible_moves += [11]
        return possible_moves

         
    def check_if_move_is_admissible(self,ix):
        r0,r1 = self.split_into_each_generator()
        first_zero_r0 = np.argmax(r0 == 0) or len(r0)   
        first_zero_r1 = np.argmax(r1 == 0) or len(r1)
        len_nonzero_r0=first_zero_r0
        len_nonzero_r1=first_zero_r1
        nonzero_r0 = r0[np.nonzero(r0)[0]]
        nonzero_r1 = r1[np.nonzero(r1)[0]]
        minusreverser0 = np.flip(-nonzero_r0)
        minusreverser1 = np.flip(-nonzero_r1)
        combineable = (len_nonzero_r0 + len_nonzero_r1)<=self.max_relator_length
        combineable_r0 = (len_nonzero_r0+2)<=self.max_relator_length
        combineable_r1 = (len_nonzero_r1+2)<=self.max_relator_length
        if ix ==0:# r_1 --> r_1 r_0
            if combineable or self.can_combine_relator_and_nonzero_relator(r1,nonzero_r0):
                return True
        elif ix ==1:# r_0 --> r_0 r_1^{-1}
            if combineable or self.can_combine_relator_and_nonzero_relator(r0,minusreverser1):
                return True
        elif ix ==2:# r_1 --> r_1 r_0^{-1}
            if combineable or self.can_combine_relator_and_nonzero_relator(r1,minusreverser0):
                return True
        elif ix ==3:# r_0 --> r_0 r_1
            if combineable or self.can_combine_relator_and_nonzero_relator(r0,nonzero_r1):
                return True
        elif ix == 4:# r_1 --> x_0^{-1} r_1 x_0
            if combineable_r1 or self.can_conjugate_relator_by_pair(nonzero_r1,[1,-1]):
                return True
        elif ix == 5:# r_0 ---> x_1^{-1} r_0 x_1
            if combineable_r0 or self.can_conjugate_relator_by_pair(nonzero_r0,[2,-2]):
                return True
        elif ix == 6:# r_1 --> x_1^{-1} r_1 x_1
            if combineable_r1 or self.can_conjugate_relator_by_pair(nonzero_r1,[2,-2]):
                return True
        elif ix == 7:# r_0 ---> x_0 r_0 x_0^{-1}
            if combineable_r0 or self.can_conjugate_relator_by_pair(nonzero_r0,[-1,1]):
                return True
        elif ix == 8:# r_1 --> x_0 r_1 x_0^{-1}
            if combineable_r1 or self.can_conjugate_relator_by_pair(nonzero_r1,[-1,1]):
                return True
        elif ix == 9:# r_0 --> x_1 r_0 x_1^{-1}
            if combineable_r0 or self.can_conjugate_relator_by_pair(nonzero_r0,[-2,2]):
                return True
        elif ix == 10:# r_1 --> x_1 r_1 x_1^{-1}
            if combineable_r1 or self.can_conjugate_relator_by_pair(nonzero_r1,[-2,2]):
                return True
        elif ix == 11:# r_0 --> x_0^{-1} r_0 x_0
            if combineable_r0 or self.can_conjugate_relator_by_pair(nonzero_r0,[1,-1]):
                return True
        return False

    def can_combine_relator_and_nonzero_relator(self,relator,nonzero_relator):
        """
        Check if we can combine relator and nonzero_relator by dragging nonzero_relator R along relator r
        """
        R_len_nonzero = len(nonzero_relator)
        r_len = len(relator)
        r_len_nonzero = len(relator[np.nonzero(relator)[0]])
        for offset in range(max(0,r_len_nonzero-R_len_nonzero),min(r_len-R_len_nonzero,r_len_nonzero)):
            padded_R = np.zeros(r_len,dtype=relator.dtype)
            padded_R[offset:offset+R_len_nonzero] = nonzero_relator
            overlap_mask = (padded_R != 0) & (relator != 0)
            sum = (relator[overlap_mask])+ np.flip(padded_R[overlap_mask])
            if not np.any(sum):
                return True
        return False

    def combine_relator_and_nonzero_relator(self,relator,nonzero_relator):
        """
        Combine relator and nonzero_relator by dragging nonzero_relator R along relator r
        """
        R_len_nonzero = len(nonzero_relator)
        r_len = len(relator)
        r_len_nonzero = len(relator[np.nonzero(relator)[0]])
        for offset in range(max(0,r_len_nonzero-R_len_nonzero),min(r_len-R_len_nonzero,r_len_nonzero)):
            padded_R = np.zeros(r_len,dtype=relator.dtype)  
            padded_R[offset:offset+R_len_nonzero] = nonzero_relator
            overlap_mask = (padded_R != 0) & (relator != 0)
            sum = relator[overlap_mask]+np.flip(padded_R[overlap_mask])
            if not np.any(sum):
                out = np.concatenate([relator[:offset],padded_R[r_len_nonzero:] ,np.zeros(r_len-(r_len-r_len_nonzero)-offset,dtype=relator.dtype)])
                return out
        return relator#otherwise return unchanged

    def do_move_to_state_flexible(self,ix):
        """
        Do relator type moves
        """
        r0,r1 = self.split_into_each_generator()
        nonzero_r0 = r0[np.nonzero(r0)[0]]
        nonzero_r1 = r1[np.nonzero(r1)[0]]
        minusreverser0 = np.flip(-nonzero_r0)
        minusreverser1 = np.flip(-nonzero_r1)
        #print("\nix:",ix,"state:",self.state)
        if ix ==0:# r_1 --> r_1 r_0
            r1=self.combine_relator_and_nonzero_relator(r1,nonzero_r0)
        elif ix ==1:# r_0 --> r_0 r_1^{-1}
            r0=self.combine_relator_and_nonzero_relator(r0,minusreverser1)
        elif ix ==2:# r_1 --> r_1 r_0^{-1}  
            r1=self.combine_relator_and_nonzero_relator(r1,minusreverser0)
        elif ix ==3:# r_0 --> r_0 r_1
            r0=self.combine_relator_and_nonzero_relator(r0,nonzero_r1)
        elif ix == 4:# r_1 --> x_0^{-1} r_1 x_0
            r1=self.contract_endpoints_of_relator_flexible(r1,[1,-1])
        elif ix == 5:# r_0 ---> x_1^{-1} r_0 x_1
            r0=self.contract_endpoints_of_relator_flexible(r0,[2,-2])
        elif ix == 6:# r_1 --> x_1^{-1} r_1 x_1
            r1=self.contract_endpoints_of_relator_flexible(r1,[2,-2])
        elif ix == 7:# r_0 ---> x_0 r_0 x_0^{-1}
            r0=self.contract_endpoints_of_relator_flexible(r0,[-1,1])
        elif ix == 8:# r_1 --> x_0 r_1 x_0^{-1}
            r1=self.contract_endpoints_of_relator_flexible(r1,[-1,1])
        elif ix == 9:# r_0 --> x_1 r_0 x_1^{-1}
            r0=self.contract_endpoints_of_relator_flexible(r0,[-2,2])
        elif ix == 10:# r_1 --> x_1 r_1 x_1^{-1}
            r1=self.contract_endpoints_of_relator_flexible(r1,[-2,2])
        elif ix == 11:# r_0 --> x_0^{-1} r_0 x_0
            r0=self.contract_endpoints_of_relator_flexible(r0,[1,-1])
        new_state = np.concatenate([r0,r1])
        if np.array_equal(new_state,self.state):
            #print("DID NOT CHANGE,", ix)
            #print(self.check_if_move_is_admissible(ix))
            #print(self.state, "move", ix)
            return self.check_if_move_is_admissible(ix)
        else:
            self.state = new_state
            self.simplify_state()
            return True

    def do_move_to_state_flexible_stateless(self,state,ix):
        """
        Vectorized version of do_move_to_state_flexible that takes a state and returns the new state.
        """
        r0 = state[:self.max_relator_length]
        r1 = state[self.max_relator_length:]
        nonzero_r0 = r0[np.nonzero(r0)[0]]
        nonzero_r1 = r1[np.nonzero(r1)[0]]
        minusreverser0 = np.flip(-nonzero_r0)
        minusreverser1 = np.flip(-nonzero_r1)

        if ix == 0:  # r_1 --> r_1 r_0
            r1 = self.combine_relator_and_nonzero_relator(r1,nonzero_r0)
        elif ix == 1:  # r_0 --> r_0 r_1^{-1}
            r0 = self.combine_relator_and_nonzero_relator(r0,minusreverser1)
        elif ix == 2:  # r_1 --> r_1 r_0^{-1}
            r1 = self.combine_relator_and_nonzero_relator(r1,minusreverser0)
        elif ix == 3:  # r_0 --> r_0 r_1
            r0 = self.combine_relator_and_nonzero_relator(r0,nonzero_r1)
        elif ix == 4:  # r_1 --> x_0^{-1} r_1 x_0
            r1 = self.contract_endpoints_of_relator_flexible(r1,[1,-1])
        elif ix == 5:  # r_0 ---> x_1^{-1} r_0 x_1
            r0 = self.contract_endpoints_of_relator_flexible(r0,[2,-2])
        elif ix == 6:  # r_1 --> x_1^{-1} r_1 x_1
            r1 = self.contract_endpoints_of_relator_flexible(r1,[2,-2])
        elif ix == 7:  # r_0 ---> x_0 r_0 x_0^{-1}
            r0 = self.contract_endpoints_of_relator_flexible(r0,[-1,1])
        elif ix == 8:  # r_1 --> x_0 r_1 x_0^{-1}
            r1 = self.contract_endpoints_of_relator_flexible(r1,[-1,1])
        elif ix == 9:  # r_0 --> x_1 r_0 x_1^{-1}
            r0 = self.contract_endpoints_of_relator_flexible(r0,[-2,2])
        elif ix == 10:  # r_1 --> x_1 r_1 x_1^{-1}
            r1 = self.contract_endpoints_of_relator_flexible(r1,[-2,2])
        elif ix == 11:  # r_0 --> x_0^{-1} r_0 x_0
            r0 = self.contract_endpoints_of_relator_flexible(r0,[1,-1])

        new_state = np.concatenate([r0,r1])
        return new_state

    def do_move_to_state_flexible_stateless_vec(self,states,ix):
        for i,state in enumerate(states):
            states[i] = self.do_move_to_state_flexible_stateless(state,ix)
        return states

    
    def combine_relator_and_nonzero_relator(self,relator,nonzero_relator):
        """
        Combine relator and nonzero_relator by dragging nonzero_relator R along relator r
        """
        R_len_nonzero = len(nonzero_relator)
        r_len = len(relator)
        r_len_nonzero = len(relator[np.nonzero(relator)[0]])
        for offset in range(max(0,r_len_nonzero-R_len_nonzero),min(r_len-R_len_nonzero+1,r_len_nonzero+1)):
            padded_R = np.zeros(r_len,dtype=relator.dtype)  
            padded_R[offset:offset+R_len_nonzero] = nonzero_relator
            overlap_mask = (padded_R != 0) & (relator != 0)
            sum = relator[overlap_mask]+np.flip(padded_R[overlap_mask])
            if not np.any(sum):
                out = np.concatenate([relator[:offset],padded_R[r_len_nonzero:] ,np.zeros(r_len-(r_len-r_len_nonzero)-offset,dtype=relator.dtype)])
                return out
        return relator#otherwise return unchanged


    def can_conjugate_relator_by_pair(self,nonzero_relator,pair):
        """
        Check if we can conjugate relator by pair
        """
        len_nonzero_relator = len(nonzero_relator)
        len_relator = self.max_relator_length
        if len_nonzero_relator == len_relator:
            return np.all(nonzero_relator[[0,-1]] == pair)
        elif len_nonzero_relator == len_relator-1:
            return (nonzero_relator[0] == pair[0]) or (nonzero_relator[-1] == pair[1])
        elif len_nonzero_relator <=len_relator-2:
            return True
        # This case should never be reached - all valid cases are handled above
        return False


    def contract_endpoints_of_relator_flexible(self, relator, pattern):
        """
        Removes the first and last nonzero elements of the relator if they match the given pattern.
        It should automatically simplify?

        Args:
            relator: Array of shape (state_size,) containing the relator to check
            pattern: List/array of [first,last] values to match against endpoints

        Returns:
            New relator with endpoints removed if pattern matched
        """
        # Get nonzero mask and indices
        nonzero_mask = relator != 0
        nonzero_indices = np.nonzero(nonzero_mask)[0]
        len_nonzero_relator = len(nonzero_indices)
        if len_nonzero_relator == 0:
            raise ValueError("Relator is empty")

        # Get first and last nonzero elements
        last_idx = nonzero_indices[-1]
        first_element = relator[0]
        last_element = relator[last_idx]
        nonzero_relator = relator[0:last_idx+1]

        # Check if endpoints match pattern
        can_contract_start = (first_element == pattern[0])
        can_contract_end = (last_element == pattern[1]) #and (0 != last_idx)# don't need this
        just_prepend_and_append = len_nonzero_relator<=len(relator)-2
        
        # Create output array
        new_relator = np.copy(relator)

        if can_contract_start and can_contract_end:
            # Remove both endpoints
            new_relator = np.zeros_like(relator)
            new_relator[:len_nonzero_relator-2] = nonzero_relator[1:-1]
        elif can_contract_start:
            # Remove first element and append pattern[1]
            new_relator[0:len_nonzero_relator-1] = nonzero_relator[1:]
            new_relator[len_nonzero_relator-1] = -pattern[1]
        elif can_contract_end:
            # Prepend pattern[0] and remove last element
            new_relator[0] = -pattern[0]
            new_relator[1:len_nonzero_relator] = nonzero_relator[0:-1]
        elif just_prepend_and_append:
            new_relator[0] = -pattern[0]
            new_relator[1:len_nonzero_relator+1] = relator[0:last_idx+1]
            new_relator[len_nonzero_relator+1] = -pattern[1]

        return new_relator

    def ONLY_FOR_TESTING_contract_endpoints_of_relator_flexible_vec(self, relators_vec, pattern):
        """
        Removes the first and last nonzero elements of each state if they match the given pattern.

        Args:
            states: Array of shape (batch_size, state_size) containing states to check
            pattern: List/array of [first,last] values to match against endpoints

        Returns:
            New states with move done
        """
        for i,relator in enumerate(relators_vec):
            relators_vec[i] = self.contract_endpoints_of_relator_flexible(relator,pattern)
        return relators_vec


    # def check_conditions_return_possibilities(self):
    #     possible_moves = []
    #     r0,r1 = self.split_into_each_generator()
    #     nonzero_r0 = r0[np.nonzero(r0)[0]]
    #     nonzero_r1 = r1[np.nonzero(r1)[0]]
    #     combineable = (len_nonzero_r0 + len_nonzero_r1)<=self.max_relator_length
    #     combineable_r1 = (len_nonzero_r1+2)<=self.max_relator_length
    #     combineable_r0 = (len_nonzero_r0+2)<=self.max_relator_length
    #     if combineable:
    #          possible_moves += [0,1,2,3]
    #     if combineable_r1:
    #         possible_moves += [4,6,8,10]
    #     else:
    #         if np.all(nonzero_r1[[0,-1]] == [1,-1]):
    #             possible_moves += [4]
    #         elif np.all(nonzero_r1[[0,-1]] == [2,-2]):
    #             possible_moves += [6]
    #         elif np.all(nonzero_r1[[0,-1]] == [-1,1]):
    #             possible_moves += [8]
    #         elif np.all(nonzero_r1[[0,-1]] == [-2,2]):
    #             possible_moves += [10]
    #     if combineable_r0:
    #         possible_moves += [5,7,9,11]
    #     else:
    #         if np.all(nonzero_r0[[0,-1]] == [2,-2]):
    #             possible_moves += [5]
    #         elif np.all(nonzero_r0[[0,-1]] == [-1,1]):
    #             possible_moves += [7]
    #         elif np.all(nonzero_r0[[0,-1]] == [-2,2]):
    #             possible_moves += [9]
    #         elif np.all(nonzero_r0[[0,-1]] == [1,-1]):
    #             possible_moves += [11]
    #     return possible_moves

    
    def scrambler(self, scramble_length):
        """
        Generates a random scramble of given length and returns the cube state and scramble moves as a generator.
        Please note that index-based implementations (faster) follow commented lexical logics.
        """
        while True:
            # Reset the cube state, scramble, and return cube state and scramble moves
            self.reset()
            scramble = []
            move = None

            # for i in range(scramble_length):
            #     # self.finger(move)
            #     #old_state = self.state.copy()
            #     # possible_moves = self.check_conditions_return_possibilities()
            #     # if possible_moves:
            #     #     move = random.choice(possible_moves)
            #     #     self.finger_ix_fast(move)
            #     #     scramble.append(move)
            #        for b in range(batch_size):
            # env.reset()
            # state = torch.tensor(env.state, device='cpu')
            # yield self.state, move
            # Generate scramble of specified length
            for i in range(scramble_length):
                # 0. r_1 --> r_1 r_0
                # 1. r_0 --> r_0 r_1^{-1} 
                # 2. r_1 --> r_1 r_0^{-1}
                # 3. r_0 --> r_0 r_1
                # 4: r_1 --> x_0^{-1} r_1 x_0
                # 5: r_0 ---> x_1^{-1} r_0 x_1
                # 6: r_1 --> x_1^{-1} r_1 x_1
                # 7: r_0 ---> x_0 r_0 x_0^{-1}
                # 8: r_1 --> x_0 r_1 x_0^{-1}
                # 9: r_0 --> x_1 r_0 x_1^{-1}
                # 10: r_1 --> x_1 r_1 x_1^{-1}
                # 11: r_0 --> x_0^{-1} r_0 x_0

                # Get first/last nonzero elements for both r0 and r1
                r0, r1 = self.split_into_each_generator()
                r0_nonzero = r0[np.nonzero(r0)[0]]
                r1_nonzero = r1[np.nonzero(r1)[0]]

                # Initialize weights for all moves
                weights = np.ones(self.num_moves) * 0.05

                if len(r0_nonzero) > 0:
                    r0_first, r0_last = r0_nonzero[0], r0_nonzero[-1]
                    # Weight moves 5,7,9,11 higher if r0 can contract
                    if r0_first == 2 and r0_last == -2:  # x1 cancellation
                        weights[5] = 0.15  # x1^-1 r0 x1
                        weights[9] = 0.15  # x1 r0 x1^-1
                    if r0_first == 1 and r0_last == -1:  # x0 cancellation
                        weights[7] = 0.15  # x0 r0 x0^-1
                        weights[11] = 0.15 # x0^-1 r0 x0

                if len(r1_nonzero) > 0:
                    r1_first, r1_last = r1_nonzero[0], r1_nonzero[-1]
                    # Weight moves 4,6,8,10 higher if r1 can contract
                    if r1_first == 1 and r1_last == -1:  # x0 cancellation
                        weights[4] = 0.15  # x0^-1 r1 x0
                        weights[8] = 0.15  # x0 r1 x0^-1
                    if r1_first == 2 and r1_last == -2:  # x1 cancellation
                        weights[6] = 0.15  # x1^-1 r1 x1
                        weights[10] = 0.15 # x1 r1 x1^-1

                # Normalize weights
                weights = weights / weights.sum()

                # Choose move and apply it
                old_state = self.state.copy()
                moves_temp = list(range(self.num_moves))

                while np.array_equal(self.state, old_state):
                    # Adjust weights based on remaining moves
                    remaining_weights = weights[moves_temp]
                    # Renormalize weights for remaining moves
                    remaining_weights = remaining_weights / remaining_weights.sum()
                    
                    # Choose move from remaining moves
                    move = np.random.choice(moves_temp, p=remaining_weights)
                    self.finger_ix_fast(move)
                    moves_temp.remove(move)
                    
                    # Break if no moves left to try
                    if not moves_temp:
                        break

                scramble.append(move)

                yield self.state, move

    def scrambler_fast(self, scramble_length):
        while True:
            self.reset()
            scramble = []
            for i in range(scramble_length):
                move = random.choice(self.moves)
                self.finger_ix_fast(move)
                scramble.append(move)
                yield self.state, move
    def finger_ix_fast(self, ix,simplify=True):
        """Faster version of finger_ix that uses vectorised operations"""
        r0, r1 = self.split_into_each_generator()
        r0_nonzero = r0[np.nonzero(r0)[0]]
        r1_nonzero = r1[np.nonzero(r1)[0]]
        if ix in [1,3,5,7,9,11]:
            new_r0 = np.zeros_like(r0)
        elif ix in [0,2,4,6,8,10]:
            new_r1 = np.zeros_like(r1)


        if ix == 0:  # r_1 --> r_1 r_0
            # Check if concatenation would exceed max length
            if len(r1_nonzero) + len(r0_nonzero) <= self.max_relator_length:
                new_r1[:len(r1_nonzero)] = r1_nonzero
                new_r1[len(r1_nonzero):len(r1_nonzero)+len(r0_nonzero)] = r0_nonzero
                self.state[self.max_relator_length:] = new_r1
            
        elif ix == 1:  # r_0 --> r_0 r_1^{-1}
            if len(r0_nonzero) + len(r1_nonzero) <= self.max_relator_length:
                new_r0[:len(r0_nonzero)] = r0_nonzero
                new_r0[len(r0_nonzero):len(r0_nonzero)+len(r1_nonzero)] = -r1_nonzero[::-1]  # Inverse is negative and reversed
                self.state[:self.max_relator_length] = new_r0
            
        elif ix == 2:  # r_1 --> r_1 r_0^{-1}
            if len(r1_nonzero) + len(r0_nonzero) <= self.max_relator_length:
                new_r1[:len(r1_nonzero)] = r1_nonzero
                new_r1[len(r1_nonzero):len(r1_nonzero)+len(r0_nonzero)] = -r0_nonzero[::-1]
                self.state[self.max_relator_length:] = new_r1
            
        elif ix == 3:  # r_0 --> r_0 r_1
            if len(r0_nonzero) + len(r1_nonzero) <= self.max_relator_length:
                new_r0[:len(r0_nonzero)] = r0_nonzero
                new_r0[len(r0_nonzero):len(r0_nonzero)+len(r1_nonzero)] = r1_nonzero
                self.state[:self.max_relator_length] = new_r0
            
        # Handle conjugation moves (4-11) with vectorised operations
        elif ix == 4:  # r_1 --> x_0^{-1} r_1 x_0
            if len(r1_nonzero) + 2 > self.max_relator_length:
                if np.all(r1_nonzero[[0,-1]] == [1,-1]):
                    new_r1[:len(r1_nonzero)-2] = r1_nonzero[1:-1]
                    self.state[self.max_relator_length:] = new_r1
            else:
                new_r1[0] = -1  # x_0^{-1} on left
                new_r1[1:len(r1_nonzero)+1] = r1_nonzero
                new_r1[len(r1_nonzero)+1] = 1  # x_0 on right
                self.state[self.max_relator_length:] = new_r1
            
        elif ix == 5:  # r_0 --> x_1^{-1} r_0 x_1
            if len(r0_nonzero) + 2 > self.max_relator_length:
                if np.all(r0_nonzero[[0,-1]] == [2,-2]):
                    new_r0[:len(r0_nonzero)-2] = r0_nonzero[1:-1]
                    self.state[:self.max_relator_length] = new_r0
            else:
                new_r0[0] = -2  # x_1^{-1} on left
                new_r0[1:len(r0_nonzero)+1] = r0_nonzero
                new_r0[len(r0_nonzero)+1] = 2  # x_1 on right
                self.state[:self.max_relator_length] = new_r0
            
        elif ix == 6:  # r_1 --> x_1^{-1} r_1 x_1
            if len(r1_nonzero) + 2 > self.max_relator_length:
                if np.all(r1_nonzero[[0,-1]] == [2,-2]):
                    new_r1[:len(r1_nonzero)-2] = r1_nonzero[1:-1]
                    self.state[self.max_relator_length:] = new_r1
            else:   
                new_r1[0] = -2  # x_1^{-1} on left
                new_r1[1:len(r1_nonzero)+1] = r1_nonzero
                new_r1[len(r1_nonzero)+1] = 2  # x_1 on right
                self.state[self.max_relator_length:] = new_r1

        elif ix == 7:  # r_0 --> x_0 r_0 x_0^{-1}
            if len(r0_nonzero) + 2 > self.max_relator_length:
                if np.all(r0_nonzero[[0,-1]] == [-1,1]):
                    new_r0[:len(r0_nonzero)-2] = r0_nonzero[1:-1]
                    self.state[:self.max_relator_length] = new_r0
            else:   
                new_r0[0] = 1  # x_0 on left
                new_r0[1:len(r0_nonzero)+1] = r0_nonzero
                new_r0[len(r0_nonzero)+1] = -1  # x_0^{-1} on right
                self.state[:self.max_relator_length] = new_r0

        elif ix == 8:  # r_1 --> x_0 r_1 x_0^{-1}
            if len(r1_nonzero) + 2 > self.max_relator_length:
                if np.all(r1_nonzero[[0,-1]] == [-1,1]):
                    new_r1[:len(r1_nonzero)-2] = r1_nonzero[1:-1]
                    self.state[self.max_relator_length:] = new_r1
            else:   
                new_r1[0] = 1  # x_0 on left
                new_r1[1:len(r1_nonzero)+1] = r1_nonzero
                new_r1[len(r1_nonzero)+1] = -1  # x_0^{-1} on right
                self.state[self.max_relator_length:] = new_r1

        elif ix == 9:  # r_0 --> x_1 r_0 x_1^{-1}
            if len(r0_nonzero) + 2 > self.max_relator_length:
                if np.all(r0_nonzero[[0,-1]] == [-2,2]):
                    new_r0[:len(r0_nonzero)-2] = r0_nonzero[1:-1]
                    self.state[:self.max_relator_length] = new_r0
            else:
                new_r0[0] = 2  # x_1 on left
                new_r0[1:len(r0_nonzero)+1] = r0_nonzero
                new_r0[len(r0_nonzero)+1] = -2  # x_1^{-1} on right
                self.state[:self.max_relator_length] = new_r0

        elif ix == 10:  # r_1 --> x_1 r_1 x_1^{-1}
            if len(r1_nonzero) + 2 > self.max_relator_length:
                if np.all(r1_nonzero[[0,-1]] == [-2,2]):
                    new_r1[:len(r1_nonzero)-2] = r1_nonzero[1:-1]
                    self.state[self.max_relator_length:] = new_r1
            else:
                new_r1[0] = 2  # x_1 on left
                new_r1[1:len(r1_nonzero)+1] = r1_nonzero
                new_r1[len(r1_nonzero)+1] = -2  # x_1^{-1} on right
                self.state[self.max_relator_length:] = new_r1

        elif ix == 11:  # r_0 --> x_0^{-1} r_0 x_0
            if len(r0_nonzero) + 2 > self.max_relator_length:
                if np.all(r0_nonzero[[0,-1]] == [1,-1]):
                    new_r0[:len(r0_nonzero)-2] = r0_nonzero[1:-1]
                    self.state[:self.max_relator_length] = new_r0
            else:
                new_r0[0] = -1  # x_0^{-1} on left
                new_r0[1:len(r0_nonzero)+1] = r0_nonzero
                new_r0[len(r0_nonzero)+1] = 1  # x_0 on right
                self.state[:self.max_relator_length] = new_r0
        #else:
        #    # Fallback to original implementation for other moves
        #    self.finger_ix(ix)
        #print('simplify?',simplify)
        if simplify:
            #print("finger_ix_fast: last state",self.state)
            self.simplify_state()
            #print("finger_ix_fast: simplified state",self.state)

    def finger_ix_fast_vec(self, states, ix,simplify=True):
        """
        vectorised version of finger_ix_fast that operates on multiple states at once.
        
        Args:
            states: Array of shape (batch_size, state_size) containing multiple cube states
            ix: Move index to apply to all states
            
        Returns:
            Updated states after applying the move
        """
        # Split states into r0 and r1 components
        r0 = states[:, :self.max_relator_length]
        r1 = states[:, self.max_relator_length:]
        r0_nonzero_mask = r0 != 0
        r1_nonzero_mask = r1 != 0
        r0_counts = r0_nonzero_mask.sum(axis=1)
        r1_counts = r1_nonzero_mask.sum(axis=1) 
        r0_indices = np.arange(self.max_relator_length)[None, :]
        r1_indices = np.arange(self.max_relator_length)[None, :]
        
        # Create output states array
        new_states = np.copy(states)
            # Start of Selection
        if ix == 0:  # r_1 --> r_1 r_0
            # Get non-zero elements and their counts for each state
            # Create mask for valid states where concatenation won't exceed max length
            mask = (r0_counts + r1_counts) <= self.max_relator_length
        
            # Initialize new r1 arrays for valid states
            new_r1 = np.zeros_like(states)
            
            # For each valid state, create indices for putting r1 values
            #np.put_along_axis(new_r1, r1_indices,
            #                r1 * r1_nonzero_mask, axis=1)
            new_r1[:,:self.max_relator_length] = r1

            # Create indices for putting r0 values after r1
            r0_start_positions = r1_counts[:, None]
            r0_newindices = r1_indices + r0_start_positions
            #r0_valid_indices = (r0_indices < (r0_counts[:, None] + r1_counts[:, None])) & (r0_indices < self.max_relator_length)
            np.put_along_axis(new_r1, r0_newindices,
                            r0 * r0_nonzero_mask, axis=1)
            
            # Update the new_states only where the mask is True, leaving others unchanged
            new_states[mask, self.max_relator_length:] = new_r1[mask,:self.max_relator_length]

        if ix == 1:  #  r_0 --> r_0 r_1^{-1}
            # Get non-zero elements and their counts for each state
            # Create mask for valid states where concatenation won't exceed max length
            mask = (r0_counts + r1_counts) <= self.max_relator_length
        
            # Initialize new r1 arrays for valid state
            new_r0 = np.zeros_like(states)
            
            # For each valid state, create indices for putting r1 values
            #np.put_along_axis(new_r0, r0_indices,
            #                r0 * r0_nonzero_mask, axis=1)
            new_r0[:,:self.max_relator_length] = r0

            reversed_r1 = -1*self.reverse_padded_vectors(r1)
            # Create indices for putting r0 values after r1
            r1_start_positions = r0_counts[:, None]
            r1_newindices = r0_indices + r1_start_positions
            #r0_valid_indices = (r0_indices < (r0_counts[:, None] + r1_counts[:, None])) & (r0_indices < self.max_relator_length)
            np.put_along_axis(new_r0, r1_newindices,
                            reversed_r1, axis=1)
            
            # Update the new_states only where the mask is True, leaving others unchanged
            new_states[mask, :self.max_relator_length] = new_r0[mask,:self.max_relator_length]

        if ix == 2:  # r_1 --> r_1 r_0^{-1}
            # Get non-zero elements and their counts for each state
            # Create mask for valid states where concatenation won't exceed max length
            mask = (r0_counts + r1_counts) <= self.max_relator_length
        
            # Initialize new r1 arrays for valid states
            new_r1 = np.zeros_like(states)
            
            # For each valid state, create indices for putting r1 values
            #np.put_along_axis(new_r1, r1_indices,
            #                r1 * r1_nonzero_mask, axis=1)
            new_r1[:,:self.max_relator_length] = r1

            reversed_r0 = -1*self.reverse_padded_vectors(r0)
            # Create indices for putting r0 values after r1
            r0_start_positions = r1_counts[:, None]
            r0_newindices = r1_indices + r0_start_positions
            #r0_valid_indices = (r0_indices < (r0_counts[:, None] + r1_counts[:, None])) & (r0_indices < self.max_relator_length)
            np.put_along_axis(new_r1, r0_newindices,
                            reversed_r0, axis=1)
            


            # Update the new_states only where the mask is True, leaving others unchanged
            new_states[mask, self.max_relator_length:] = new_r1[mask,:self.max_relator_length]
        
        if ix == 3:  # r_0 --> r_0 r_1
            # Get non-zero elements and their counts for each state
            # Create mask for valid states where concatenation won't exceed max length
            mask = (r0_counts + r1_counts) <= self.max_relator_length
        
            # Initialize new r1 arrays for valid states
            new_r0 = np.zeros_like(states)
            
            # For each valid state, create indices for putting r1 values
            new_r0[:,:self.max_relator_length] = r0

            # Create indices for putting r0 values after r1
            r1_start_positions = r0_counts[:, None]
            r1_newindices = r0_indices + r1_start_positions
            #r0_valid_indices = (r0_indices < (r0_counts[:, None] + r1_counts[:, None])) & (r0_indices < self.max_relator_length)
            np.put_along_axis(new_r0, r1_newindices,
                            r1 * r1_nonzero_mask, axis=1)
            
            # Update the new_states only where the mask is True, leaving others unchanged
            new_states[mask, :self.max_relator_length] = new_r0[mask,:self.max_relator_length]

        if ix ==4:# r_1 --> x_0^{-1} r_1 x_0
            # Create mask for valid states where adding 2 elements won't exceed max length
            mask = (r1_counts + 2) <= self.max_relator_length
            new_r1 = np.zeros((r1.shape[0],self.max_relator_length+2))
            
            # Shift original r1 values right by 1 position
            new_r1[:,1:self.max_relator_length] = r1[:,:-1]
            
            # Add x_0^{-1} at start and x_0 at end
            new_r1[:, 0] = -1  # Add x_0^{-1} at start
            new_r1[np.arange(len(r1_counts)), r1_counts + 1] = 1  # Add x_0 after r1
            
            # Update states where mask is True
            new_states[mask, self.max_relator_length:] = new_r1[mask, :self.max_relator_length]
            # For states that don't meet the mask condition, try contracting
            unmasked = ~mask
            if np.any(unmasked):
               # Try to contract unmasked states with [-1,1] pattern
               contracted_r1 = self.contract_endpoints(r1[unmasked], [1,-1])
               new_states[unmasked, self.max_relator_length:] = contracted_r1

        if ix == 5: # r_0 --> x_1^{-1} r_0 x_1
            # Create mask for valid states where adding 2 elements won't exceed max length
            mask = (r0_counts + 2) <= self.max_relator_length
            new_r0 = np.zeros((r0.shape[0],self.max_relator_length+2),dtype=self.DTYPE)
            
            # Shift original r0 values right by 1 position
            new_r0[:,1:self.max_relator_length] = r0[:,:-1]
            
            # Add x_1^{-1} at start and x_1 at end
            new_r0[:, 0] = -2  # Add x_1^{-1} at start
            # Add x_1 after each r0 sequence using vectorised indexing
            new_r0[np.arange(len(r0_counts)), r0_counts + 1] = 2  # Add x_1 after r0
            
            # Update states where mask is True
            new_states[mask, :self.max_relator_length] = new_r0[mask, :self.max_relator_length]
            unmasked = ~mask
            if np.any(unmasked):
               # Try to contract unmasked states with [-1,1] pattern
               contracted_r0 = self.contract_endpoints(r0[unmasked], [2,-2])
               new_states[unmasked, :self.max_relator_length] = contracted_r0 

        if ix == 6: # r_1 --> x_1^{-1} r_1 x_1  
            #Create mask for valid states where adding 2 elements won't exceed max length
            mask = (r1_counts + 2) <= self.max_relator_length
            new_r1 = np.zeros((r1.shape[0],self.max_relator_length+2))
            
            # Shift original r1 values right by 1 position
            new_r1[:,1:self.max_relator_length] = r1[:,:-1]
            
            # Add x_0^{-1} at start and x_0 at end
            new_r1[:, 0] = -2  # Add x_0^{-1} at start
            new_r1[np.arange(len(r1_counts)), r1_counts + 1] = 2 # Add x_0 after r1
            
            # Update states where mask is True
            new_states[mask, self.max_relator_length:] = new_r1[mask, :self.max_relator_length]
            unmasked = ~mask
            if np.any(unmasked):
               contracted_r1 = self.contract_endpoints(r1[unmasked], [2,-2])
               new_states[unmasked, self.max_relator_length:] = contracted_r1 

        if ix == 7: # r_0 --> x_0 r_0 x_0^{-1}  
            # Create mask for valid states where adding 2 elements won't exceed max length
            mask = (r0_counts + 2) <= self.max_relator_length
            new_r0 = np.zeros((r0.shape[0],self.max_relator_length+2))
            new_r0[:,1:self.max_relator_length] = r0[:,:-1]
            new_r0[:, 0] = 1  # Add x_1^{-1} at start
            new_r0[np.arange(len(r0_counts)), r0_counts + 1] =-1  # Add x_1 after r0
            new_states[mask, :self.max_relator_length] = new_r0[mask, :self.max_relator_length]
            unmasked = ~mask
            if np.any(unmasked):
               contracted_r0 = self.contract_endpoints(r0[unmasked], [-1,1])
               new_states[unmasked, :self.max_relator_length] = contracted_r0 

        if ix == 8: # r_1 --> x_0 r_1 x_0^{-1}
            # Create mask for valid states where adding 2 elements won't exceed max length
            mask = (r1_counts + 2) <= self.max_relator_length
            new_r1 = np.zeros((r1.shape[0],self.max_relator_length+2))
            new_r1[:,1:self.max_relator_length] = r1[:,:-1]
            new_r1[:, 0] = 1  # Add x_0^{-1} at start
            new_r1[np.arange(len(r1_counts)), r1_counts + 1] = -1  # Add x_0 after r1
            new_states[mask, self.max_relator_length:] = new_r1[mask, :self.max_relator_length]
            unmasked = ~mask
            if np.any(unmasked):
               contracted_r1 = self.contract_endpoints(r1[unmasked], [-1,1])
               new_states[unmasked, self.max_relator_length:] = contracted_r1 

        if ix==9: # r_0 --> x_1 r_0 x_1^{-1}
            # Create mask for valid states where adding 2 elements won't exceed max length
            mask = (r0_counts + 2) <= self.max_relator_length
            new_r0 = np.zeros((r0.shape[0],self.max_relator_length+2))
            new_r0[:,1:self.max_relator_length] = r0[:,:-1]
            new_r0[:, 0] = 2  # Add x_1^{-1} at start
            new_r0[np.arange(len(r0_counts)), r0_counts + 1] =-2  # Add x_1 after r0
            new_states[mask, :self.max_relator_length] = new_r0[mask, :self.max_relator_length]
            unmasked = ~mask
            if np.any(unmasked):
               contracted_r0 = self.contract_endpoints(r0[unmasked], [-2,2])
               new_states[unmasked, :self.max_relator_length] = contracted_r0 

        if ix == 10: # r_1 --> x_1 r_1 x_1^{-1}
            # Create mask for valid states where adding 2 elements won't exceed max length
            mask = (r1_counts + 2) <= self.max_relator_length
            new_r1 = np.zeros((r1.shape[0],self.max_relator_length+2))
            new_r1[:,1:self.max_relator_length] = r1[:,:-1]
            new_r1[:, 0] = 2  # Add x_1^{-1} at start
            new_r1[np.arange(len(r1_counts)), r1_counts + 1] = -2  # Add x_1 after r1
            new_states[mask, self.max_relator_length:] = new_r1[mask, :self.max_relator_length]
            unmasked = ~mask
            if np.any(unmasked):
               contracted_r1 = self.contract_endpoints(r1[unmasked], [-2,2])
               new_states[unmasked, self.max_relator_length:] = contracted_r1 
        
        if ix==11: # r_0 --> x_0^{-1} r_0 x_0
            # Create mask for valid states where adding 2 elements won't exceed max length
            mask = (r0_counts + 2) <= self.max_relator_length
            new_r0 = np.zeros((r0.shape[0],self.max_relator_length+2))
            new_r0[:,1:self.max_relator_length] = r0[:,:-1]
            new_r0[:, 0] = -1  # Add x_0^{-1} at start
            new_r0[np.arange(len(r0_counts)), r0_counts + 1] = 1  # Add x_0 after r0
            new_states[mask, :self.max_relator_length] = new_r0[mask, :self.max_relator_length]
            unmasked = ~mask
            if np.any(unmasked):
               contracted_r0 = self.contract_endpoints(r0[unmasked], [1,-1])
               new_states[unmasked, :self.max_relator_length] = contracted_r0 

        #8: r_1 --> x_0 r_1 x_0^{-1}
        #9: r_0 --> x_1 r_0 x_1^{-1}
        #10: r_1 --> x_1 r_1 x_1^{-1}
        #11: r_0 --> x_0^{-1} r_0 x_0

        if simplify:
            new_states = self.simplify_state_vec(new_states)
        return new_states

   
    def contract_endpoints(self, states, pattern):
        """
        Removes the first and last nonzero elements of each state if they match the given pattern.

        Args:
            states: Array of shape (batch_size, state_size) containing states to check
            pattern: List/array of [first,last] values to match against endpoints

        Returns:
            New states with endpoints removed where pattern matched
        """
        # Get nonzero masks and counts
        nonzero_mask = states != 0
        nonzero_counts = nonzero_mask.sum(axis=1)

        # Get first nonzero element for each state
        first_nonzero_indices = np.argmax(nonzero_mask, axis=1)
        first_elements = states[np.arange(len(states)), first_nonzero_indices]

        # Get last nonzero element for each state
        # Flip the mask and find first True to get last nonzero position
        last_nonzero_indices = states.shape[1] - 1 - np.argmax(np.fliplr(nonzero_mask), axis=1)
        last_elements = states[np.arange(len(states)), last_nonzero_indices]

        # Find which states can be contracted
        can_contract = (first_elements == pattern[0]) & (last_elements == pattern[1])

        # Create output array
        new_states = np.copy(states)

        if np.any(can_contract):
            # Get indices of states that can be contracted
            contract_indices = np.where(can_contract)[0]

            # For each state that can be contracted
            for idx in contract_indices:
                # Remove first and last elements by shifting everything left
                nonzero_count = nonzero_counts[idx]
                new_state = np.zeros_like(states[idx])
                new_state[:nonzero_count-2] = states[idx][1:nonzero_count-1]
                new_states[idx] = new_state

        return new_states


    def reverse_padded_vectors(self, padded_vectors):
        # Create a mask for non-zero elements
        mask = padded_vectors != 0

        # Get the counts of non-zero elements per row
        lengths = mask.sum(axis=1)

        # Create indices array for each row
        row_indices = np.arange(padded_vectors.shape[0])[:, None]
        col_indices = np.arange(padded_vectors.shape[1])[None, :]

        # Create reversed indices for non-zero elements
        reversed_indices = lengths[:, None] - 1 - col_indices

        # Create mask for valid reversed indices
        valid_mask = (reversed_indices >= 0) & mask

        # Create output array of zeros
        result = np.zeros_like(padded_vectors)

        # Fill in reversed values
        result[row_indices, col_indices] = np.where(valid_mask,
            padded_vectors[row_indices, reversed_indices],
            0)
        return result
 
    def simplify_state(self):
        #if we have [-x,x] we can replace with [], and pad the relator with zero
        
        # Split state into r0 and r1 relators
        r0, r1 = self.split_into_each_generator()
        
        # For each relator, find and cancel adjacent pairs that sum to 0
        def simplify_relator(r):
            # Get non-zero elements 
            nonzero_mask = r != 0
            if not np.any(nonzero_mask):
                return r  # Return original if all zeros
                
            nonzero = r[nonzero_mask]
            if len(nonzero) <= 1:
                return r  # Return original if 0 or 1 elements
            
            # Keep simplifying until no more changes can be made
            while True:
                # Find first pair that sums to 0
                cancel_mask = np.zeros(len(nonzero), dtype=bool)
                cancel_mask[:-1] = (nonzero[:-1] + nonzero[1:] == 0)
                
                # If no cancellations found, we're done
                if not np.any(cancel_mask):
                    break
                
                # Find index of first cancellation
                first_cancel_idx = np.argmax(cancel_mask)
                
                # Create mask keeping everything except the first cancellation pair
                keep_mask = np.ones(len(nonzero), dtype=bool)
                keep_mask[first_cancel_idx:first_cancel_idx+2] = False
                
                # Keep only non-cancelled elements
                nonzero = nonzero[keep_mask]
                if len(nonzero) <= 1:
                    break
            
            # Create final result
            result = np.zeros_like(r)
            if len(nonzero) > 0:
                result[:len(nonzero)] = nonzero
            return result
        # Apply simplification to both relators
        r0_simplified = simplify_relator(r0)
        r1_simplified = simplify_relator(r1)
        
        # Update state
        self.state[:self.max_relator_length] = r0_simplified
        self.state[self.max_relator_length:] = r1_simplified

    
    # def simplify_state_vec(self,states):
    #     env = AC_presentation(max_relator_length=self.max_relator_length)
    #     states_final = np.zeros_like(states)
    #     for i in range(len(states)):
    #         env.state = states[i].copy()
    #         env.simplify_state()
    #         states_final[i] = env.state
    #     return states_final

    
    def simplify_state_vec(self, states):
        """Vectorized version of simplify_state that operates on a batch of states without explicit loops over states."""
        # Split states into r0 and r1
        r0 = states[:, :self.max_relator_length]
        r1 = states[:, self.max_relator_length:]
            
        # Simplify both halves of the states
        r0_simplified = self._iterative_simplify_vectorized(r0) 
        r1_simplified = self._iterative_simplify_vectorized(r1)
        
        # Recombine
        simplified_states = np.hstack([r0_simplified, r1_simplified])
        return simplified_states

    def _iterative_simplify_vectorized(self, relators):
        """
        Perform iterative simplification on a batch of relators simultaneously.
        This mimics the stack-like iterative removal of zero-sum adjacent pairs,
        but does it in a vectorized manner across the entire batch.
        """
        batch_size, max_len = relators.shape
        current = np.copy(relators)
        
        # Keep simplifying until no more changes
        while True:
            # Get nonzero elements and their counts for each row
            nonzero_mask = current != 0
            nonzero_counts = np.sum(nonzero_mask, axis=1)
            
            # Find adjacent pairs that sum to zero
            adjacent_sums = current[:, :-1] + current[:, 1:]
            valid_pairs = (adjacent_sums == 0) & nonzero_mask[:, :-1] & nonzero_mask[:, 1:]
            
            # If no cancellations found anywhere, we're done
            if not np.any(valid_pairs):
                break
                
            # For each row, find first cancellation pair
            first_cancel_idx = np.argmax(valid_pairs, axis=1)
            
            # Create new array with cancellations removed
            new_current = np.zeros_like(current)
            for i in range(batch_size):
                if np.any(valid_pairs[i]):
                    # Get elements excluding the cancelled pair
                    row = current[i][nonzero_mask[i]]
                    cancel_idx = first_cancel_idx[i]
                    kept = np.concatenate([row[:cancel_idx], row[cancel_idx+2:]])
                    # Place kept elements at start of row
                    new_current[i, :len(kept)] = kept
                else:
                    new_current[i] = current[i]
                    
            current = new_current
            
        return current




    def apply_all_moves_to_all_states(self,states):
        """
        Applies all possible moves to each state in the batch of states.
        
        Args:
            states: Array of shape (batch_size, state_size) containing multiple cube states
            
        Returns:
            Array of shape (batch_size, num_moves, state_size) containing all resulting states
        """
        batch_size = states.shape[0]
        num_moves = 12
        
        # Create output array with shape (batch_size, num_moves, state_size)
        all_states = np.zeros((batch_size, num_moves, states.shape[1]), dtype=states.dtype)
        
        # For each move
        for move_idx in range(num_moves):
            # Copy original states
            all_states[:,move_idx] = np.copy(states)
            # Apply move to this slice
            self.finger_ix_fast_vec(all_states[:,move_idx], move_idx)
            
        return all_states

    # def _iterative_simplify_vectorized_torch(self, relators):
    #     """
    #     PyTorch version of _iterative_simplify_vectorized that operates on a batch of relators.
    #     """
    #     batch_size, max_len = relators.shape
    #     current = relators.clone()
        
    #     # Keep simplifying until no more changes
    #     while True:
    #         # Get nonzero elements and their counts for each row
    #         nonzero_mask = current != 0
    #         nonzero_counts = torch.sum(nonzero_mask, dim=1)
            
    #         # Find adjacent pairs that sum to zero
    #         adjacent_sums = current[:, :-1] + current[:, 1:]
    #         valid_pairs = (adjacent_sums == 0) & nonzero_mask[:, :-1] & nonzero_mask[:, 1:]
            
    #         # If no cancellations found anywhere, we're done
    #         if not torch.any(valid_pairs):
    #             break
                
    #         # For each row, find first cancellation pair
    #         first_cancel_idx = torch.argmax(valid_pairs.to(torch.int8), dim=1)
    #         # Create new tensor with cancellations removed
    #         new_current = torch.zeros_like(current)
    #         for i in range(batch_size):
    #             if torch.any(valid_pairs[i]):
    #                 # Get elements excluding the cancelled pair
    #                 row = current[i][nonzero_mask[i]]
    #                 cancel_idx = first_cancel_idx[i]
    #                 kept = torch.cat([row[:cancel_idx], row[cancel_idx+2:]])
    #                 # Place kept elements at start of row
    #                 new_current[i, :len(kept)] = kept
    #             else:
    #                 new_current[i] = current[i]
                    
    # def _iterative_simplify_vectorized_torch(self, relators):
    #     """
    #     Perform iterative simplification on a batch of relators simultaneously.
    #     This mimics the stack-like iterative removal of zero-sum adjacent pairs,
    #     but does it in a vectorized manner across the entire batch.
    #     """
    #     batch_size, max_len = relators.shape
    #     current = relators.clone()
    #     current_extended = current.clone()
        
    #     # Keep simplifying until no more changes
    #     i=0
    #     while True:
    #         current_extended = torch.cat([current_extended, torch.zeros_like(current)], dim=1)
    #         # Get nonzero elements and their counts for each row
    #         nonzero_mask = current_extended != 0
            
    #         # Find adjacent pairs that sum to zero
    #         adjacent_sums = current_extended[:, :-1] + current_extended[:, 1:]
    #         valid_pairs = (adjacent_sums == 0) & nonzero_mask[:, :-1] & nonzero_mask[:, 1:]
                
    #         # If no cancellations found anywhere, we're done
    #         if not torch.any(valid_pairs):
    #             break
                    
    #         # For each row, find first cancellation pair
    #         first_cancel_idx = torch.argmax(valid_pairs.to(torch.int8), dim=1)
            
    #         # Create a tensor of positions
    #         positions = torch.arange(current_extended.size(1), device=current_extended.device).unsqueeze(0).expand(current_extended.size(0), -1)
            
    #         # Only modify rows that have valid pairs to remove
    #         has_valid_pairs = torch.any(valid_pairs, dim=1)
            
    #         # Initialize to_keep as all True
    #         to_keep = torch.ones_like(current_extended, dtype=torch.bool)
            
    #         # Only mark positions for removal in rows that have valid pairs
    #         to_remove = (positions == first_cancel_idx.unsqueeze(1)) | (positions == (first_cancel_idx + 1).unsqueeze(1))
    #         to_keep[has_valid_pairs] = ~to_remove[has_valid_pairs]
            
    #         # Get the kept elements
    #         kept_values = torch.where(to_keep, current_extended, torch.zeros_like(current_extended))
            
    #         # Create mask of non-zero values 
    #         nonzero_mask = kept_values != 0
            
    #         # Get indices of non-zero elements for each row
    #         # Sort by (is_zero, position) to preserve lexicographic order within non-zeros
    #         positions = torch.arange(current_extended.size(1), device=current_extended.device)
    #         sort_keys = (~nonzero_mask).to(torch.int64) * current_extended.size(1) + positions
    #         nonzero_indices = sort_keys.argsort(dim=1)
            
    #         # Sort values putting non-zeros first
    #         sorted_values = torch.gather(kept_values, 1, nonzero_indices)
            
    #         current_extended = sorted_values
    #         i+=1
    #     return current_extended[:, :current.size(1)]
    # def _iterative_simplify_vectorized_torch(self, relators):
    #     """
    #     Vectorized simplification of relator batches using preallocated tensors and fused operations.
    #     """
    #     batch_size, max_len = relators.shape
    #     device = relators.device
        
    #     # Preallocate tensors we'll reuse
    #     current = relators.clone() 
    #     buffer = torch.zeros((batch_size, max_len*2), dtype=relators.dtype, device=device)
    #     positions = torch.arange(max_len*2, device=device).expand(batch_size, -1)
        
    #     while True:
    #         # Copy current into first half of buffer
    #         buffer[:, :max_len] = current
    #         buffer[:, max_len:] = 0
            
    #         # Find cancellations using fused operations
    #         nonzero = buffer != 0
    #         sums = buffer[:, :-1] + buffer[:, 1:] 
    #         valid_pairs = (sums == 0) & nonzero[:, :-1] & nonzero[:, 1:]
            
    #         if not valid_pairs.any():
    #             break
                
    #         # Find first cancellation per row
    #         cancel_idx = torch.argmax(valid_pairs.to(torch.int8), dim=1)
    #         has_valid = valid_pairs.any(dim=1)
            
    #         # Mask out cancelled pairs
    #         to_remove = (positions == cancel_idx.unsqueeze(1)) | (positions == (cancel_idx + 1).unsqueeze(1))
    #         keep_mask = ~(to_remove & has_valid.unsqueeze(1))
            
    #         # Compact remaining elements
    #         kept = torch.where(keep_mask, buffer, torch.zeros_like(buffer))
    #         nonzero = kept != 0
    #         sort_keys = (~nonzero) * (max_len*2) + positions
    #         sorted_indices = sort_keys.argsort(dim=1)
    #         current = torch.gather(kept, 1, sorted_indices)[:, :max_len]

    #     return current


            
#  while True:
#             # Get nonzero elements and their counts for each row
#             nonzero_mask = current != 0
#             nonzero_counts = torch.sum(nonzero_mask, dim=1)
            
#             # Find adjacent pairs that sum to zero
#             adjacent_sums = current[:, :-1] + current[:, 1:]
#             valid_pairs = (adjacent_sums == 0) & nonzero_mask[:, :-1] & nonzero_mask[:, 1:]

#             # If no cancellations found anywhere, we're done
#             if not torch.any(valid_pairs):
#                 break
                
#             # For each row, find first cancellation pair
#             first_cancel_idx = torch.argmax(valid_pairs.to(torch.int8), dim=1)
            
#             # Create new array with cancellations removed
#             new_current = torch.zeros_like(current)
#             for i in range(batch_size):
#                 if torch.any(valid_pairs[i]):
#                     # Get elements excluding the cancelled pair
#                     row = current[i][nonzero_mask[i]]
#                     cancel_idx = first_cancel_idx[i]
#                     kept = torch.cat([row[:cancel_idx], row[cancel_idx+2:]])
#                     # Place kept elements at start of row
#                     new_current[i, :len(kept)] = kept
#                 else:
#                     new_current[i] = current[i]
                    
#             current = new_current
            
#         return current

if __name__ == "__main__":
    import AC_tests


import torch
import os
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import math
# Assume Cube3 and TrainConfig are defined elsewhere
class ScrambleGeneratorAC(IterableDataset):
    def __init__(
            self,
            max_depth=1,#TrainConfig.max_depth,
            total_samples=1,#TrainConfig.num_steps * TrainConfig.batch_size_per_depth
            seed=None,
            env=AC_presentation()
        ):
        super(ScrambleGeneratorAC, self).__init__()
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
            X = np.zeros((self.max_depth+1, self.env.state_dim), dtype=int)
            X[0,:] = self.env.goal
            y = np.zeros((self.max_depth,), dtype=int)
            for j in range(self.max_depth):
                state, last_move = next(self.generator)
                X[j+1, :] = state
                y[j] = last_move
            yield X, y
