import numpy as np
import torch
import os
import time
from AC_training import *
"""
Must define the environment before using this file.
"""

def get_AKn_state(n, max_relator_length=100):
    """Returns the state representation for AK(n) presentation.
    
    AK(n) = ⟨x,y|xn=yn+1,xyx=yxy⟩ where n≥2
    
    Args:
        n: Integer ≥ 2 specifying which AK group
        max_relator_length: Maximum length to pad relators to
    
    Returns:
        numpy array of shape (2, max_relator_length) containing the two relators
    """
    # Initialize state array
    state = np.zeros((2, max_relator_length), dtype=np.int32)
    
    # First relator: x^n = y^(n+1)
    # x^n part: n ones
    state[0, :n] = 1
    # y^(n+1) part: n+1 negative twos
    state[0, n:2*n+1] = -2
    
    # Second relator: xyx = yxy
    state[1, :6] = [1, 2, 1, -2, -1, -2]
    state=state.reshape(2*max_relator_length)
    return state




# Function to get all states n moves away from solved state
def get_states_n_moves_away_AC(env,n):
    #env=AC_presentation()
    if n > 6:
        raise ValueError("n should not be greater than 6 to avoid excessive computation time")
    
    states = []
    def dfs(depth,state):
        #states.append(np.concatenate([state.copy(),np.array([])]))#depth
        states.append(state.copy())#depth
        if depth == n:
            return
        for move in env.moves:
            env.state = state.copy()
            env.do_move_to_state_flexible(move)
            dfs(depth + 1,env.state.copy())
            #env.finger_ix_fast(env.inverse_moves[move])  # Undo the move
    
    env.reset()  # Start from solved state
    dfs(0,env.state.copy())
    return np.array(states)



# Function to hash a cube state
# def hash_state(state):
#     return hash(tuple(state))

@torch.jit.script
def hash_vectors_torch(vectors: torch.Tensor, mod: int = 2**61-1, prime:int=101, force_no_collision: bool=False, monitor_collision: bool=True) -> torch.Tensor:
    """
    Hash batches of vectors containing numbers 0-20
    Input shape: (..., 20) - last dimension must be 20
    Returns: (...) - same shape as input minus last dimension
    """
    #assert ((vectors >= 0) & (vectors <= 20)).all(), "Values must be between 0 and 20"

    # Convert to int64 to prevent overflow during multiplication
    vectors = vectors.to(torch.int64)+2

    # Powers of prime for each position
    n = vectors.shape[-1]
    
    powers = torch.arange(vectors.shape[-1], device=vectors.device, dtype=torch.int64)
    multipliers = prime ** powers

    # Compute hash: sum((num + 1) * prime^pos) % mod
    hash_vals = ((vectors + 1) * multipliers) % mod
    hash_vals = hash_vals.sum(dim=-1) % mod
    if monitor_collision:
        if len(torch.unique(hash_vals)) != len(vectors):
            print("Warning: hashed states collision! Change the prime.")
            print("hashed: ", len(torch.unique(hash_vals)))
            print("unhashed: ", len(vectors))
            if force_no_collision:
                raise ValueError("Collision detected. If you want to continue anyway, set force_no_collision=False.")
        else:
            print("No collision! Good prime.")

    return hash_vals



def is_state_in_short_distance(state,hashed_states_close_to_finish):
    """
    Check if a given state is in the list of states 4 moves away from solved.
    
    Args:
    state (numpy.ndarray): The cube state to check.
    
    Returns:
    bool: True if the state is in the list, False otherwise.
    """
    if len(state.shape) != 1:
        raise ValueError("state must be a 1D array")
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    state = state[None, :]
    hashed_input_state = hash_vectors_torch(state,monitor_collision=False)[0]
    return hashed_input_state in hashed_states_close_to_finish

# Vectorize is_state_in_short_distance check across states
def is_state_in_short_distance_torch(states,hashed_states_close_to_finish_torch):
    """
    Vectorized version of is_state_in_short_distance for batches of states.
    
    Args:
        states (torch.Tensor): Batch of cube states to check
        
    Returns:
        torch.Tensor: Boolean tensor indicating which states are in short distance
    """
    # # Convert states to numpy if they're torch tensors
    # if isinstance(states, torch.Tensor):
    #     states = states.cpu().numpy()
    
    # Handle both single state and batched states
    if len(states.shape) == 1:
        states = states[None, :]
    # Vectorize the check across the batch using vmap
    hashed_states = hash_vectors_torch(states,monitor_collision=False,force_no_collision=False).to(states.device)
    results = torch.isin(hashed_states, (hashed_states_close_to_finish_torch).to(states.device))
    
    # Convert results to torch tensor
    return results#torch.tensor(results, dtype=torch.bool, device=states.device if isinstance(states, torch.Tensor) else 'cpu')

def is_state_actually_in_list_torch(state,states_close_to_finish_torch):
    # Check if state exists in the tensor of states close to finish
    if len(state.shape) == 1:
        state = state.unsqueeze(0)  # Add batch dimension if single state
    return (state == states_close_to_finish_torch.to(state.device)).all(dim=1).any()

def is_state_actually_in_list(state,states_close_to_finish):
    # Check if state exists in the array of states close to finish
    if len(state.shape) == 1:
        state = np.expand_dims(state, 0)  # Add batch dimension if single state
    return np.any(np.all(state == states_close_to_finish, axis=1))

# def vmap_is_state_in_short_distance_torch(states):


from torch_scatter import scatter_max
def beam_search_fast(env, model, fwd_model, beam_width=100, max_depth=0, check_short_distance=False, skip_redundant_moves=True,attempts=1,start_step=0,many_initial_states=None,expensive_pick_best=True, hashed_states_close_to_finish_torch=None, states_close_to_finish_torch=None):
    """
    Beam search to solve the cube using model predictions.
    
    Args:
        env: Cube environment
        model: Trained model for move prediction
        beam_width: Number of candidates to keep at each depth
        max_depth: Maximum search depth
        check_short_distance: Whether to check if solution is found at each step
        skip_redundant_moves: Whether to skip redundant move sequences
    
    Returns:
        success: Whether solution was found
        result: Dictionary with solution info
        path: Solution path if found
    """
    if hashed_states_close_to_finish_torch is None or states_close_to_finish_torch is None:
        raise ValueError("hashed_states_close_to_finish_torch and states_close_to_finish_torch must be provided")
    device = next(model.parameters()).device
    start_time = time.time()
    num_nodes = 0
    current_width = 1
    length_trajectory = max_depth-start_step
    init_state = env.state.to(device) if isinstance(env.state,torch.Tensor) else torch.tensor(env.state,device=device,dtype=torch.int64)    
    # Pre-allocate tensors
    candidate_states = -1*torch.ones((beam_width,env.state_dim),device=device,dtype=torch.int64)
    if many_initial_states is not None:
        print("using initial states given")
        num_states = many_initial_states.shape[0]
        candidate_states[0:num_states,:] = torch.tensor(many_initial_states,device=device,dtype=torch.int64)
    else:
        print("using initial state of env")
        candidate_states[0,:] =  init_state
    candidate_paths = -1*torch.ones((beam_width,length_trajectory),device=device,dtype=torch.int64)
    candidate_log_values = torch.zeros((beam_width),device=device,dtype=torch.float64)
    paths_flat = -1*torch.ones((beam_width*env.num_moves,length_trajectory),device=device,dtype=torch.int64)
    states_flat = -1*torch.ones((beam_width*env.num_moves,candidate_states.shape[-1]),device=device,dtype=torch.int64)
    log_values_flat = torch.zeros((beam_width*env.num_moves),device=device,dtype=torch.float64)
    sorted_indices = torch.arange(beam_width*env.num_moves,device=device,dtype=torch.int64)
    #batch_p = torch.zeros((beam_width,env.num_moves),device=device,dtype=torch.float64)
    inverse_indices = torch.zeros((beam_width*env.num_moves),device=device,dtype=torch.int64)
    reverse_diffusion_probs_func = reverse_diffusion_probs_from_fwd_model_and_scores(fwd_model,env,return_states_out=True,device=device)

    # Search up to max depth
    for attempt in range(attempts):
        for depth in range(start_step,max_depth):
            num_nodes += current_width

            # Get model predictions
            with torch.no_grad():
                t = torch.tensor(1 - depth / max_depth, device=device).repeat(current_width)
                batch_scores = model(candidate_states[:current_width],t)

            #new_states = apply_all_moves_to_all_states_torch_jit(candidate_states[:current_width])
            #batch_probs = get_probabilities_from_model_output(batch_scores,candidate_states[:current_width],new_states,env,weight_contraction=TrainConfig.weight_contraction,total_relator_weight=TrainConfig.total_relator_weight,double_weight=TrainConfig.double_weight)
            batch_probs, new_states = reverse_diffusion_probs_func(candidate_states[:current_width],batch_scores,return_states_out=True)
            log_batch_p = torch.log(batch_probs + torch.finfo(batch_probs.dtype).eps)  # Add dtype-appropriate epsilon to avoid log(0)
            
            # Update log values efficiently
            log_values_flat[:current_width*env.num_moves] = candidate_log_values[:current_width].repeat_interleave(env.num_moves) + log_batch_p.reshape(-1)
            
            # Update states and paths
            states_flat[:current_width*env.num_moves] = new_states.reshape(-1, candidate_states.shape[-1])
            paths_flat[:current_width*env.num_moves] = candidate_paths[:current_width].repeat_interleave(env.num_moves,dim=0)
            paths_flat[:current_width*env.num_moves,attempt*length_trajectory+(depth-start_step)] = torch.arange(env.num_moves,device=device).tile((current_width,))

            if expensive_pick_best:
                # Find unique states and get max values
                inverse_indices[:current_width*env.num_moves] = torch.unique(states_flat[:current_width*env.num_moves], dim=0, return_inverse=True, return_counts=False)[1]
                #scatter_max returns the max values and the indices of the max values in the original array
                __, max_indices = scatter_max(log_values_flat[:current_width*env.num_moves], inverse_indices[:current_width*env.num_moves], dim=0)
                current_width = min(beam_width, len(max_indices))
            
                # Get top k efficiently
                candidate_log_values[:current_width], sorted_indices[:current_width] = torch.topk(log_values_flat[max_indices], k=current_width, largest=True, sorted=False)
                max_sorted = max_indices[sorted_indices[:current_width]]
                
                # Update candidates
                candidate_states[:current_width] = states_flat[max_sorted]
                candidate_paths[:current_width] = paths_flat[max_sorted]
                candidate_log_values[:current_width] = log_values_flat[max_sorted]
            else:
                # Simple top k selection
                current_width = min(beam_width, current_width*env.num_moves)
                sorted_indices[:current_width] = torch.argsort(log_values_flat[:current_width], descending=True)[:current_width]
                candidate_states[:current_width] = states_flat[sorted_indices[:current_width]]
                candidate_paths[:current_width] = paths_flat[sorted_indices[:current_width]]
                candidate_log_values[:current_width] = log_values_flat[sorted_indices[:current_width]]

            # Check for solutions
            check_states = is_state_in_short_distance_torch(candidate_states[:current_width],hashed_states_close_to_finish_torch)
            if torch.any(check_states):
                positive_indices = torch.where(check_states)[0]
                for idx in positive_indices:
                    state = candidate_states[idx]
                    if is_state_actually_in_list_torch(state,states_close_to_finish_torch):
                        good_path = candidate_paths[idx]
                        good_path = good_path[:(attempt)*length_trajectory+(depth-start_step)+1]
                        good_path = good_path.to('cpu').detach().numpy() # Move to CPU to avoid memory accumulation
                        return True, {
                            'solution': good_path,
                            'num_nodes_generated': num_nodes,
                            'time': time.time() - start_time
                        }, good_path, attempt
                    else:
                        print("bad hash: ",state)
                        
    return False, {
        'solution': None,
        'num_nodes_generated': num_nodes,
        'time': time.time() - start_time
    }, None, attempt

