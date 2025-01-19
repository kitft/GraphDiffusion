import numpy as np
import torch
import os
import time

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



# Initialize the Rubik's Cube environment
# cube = Cube3()

def get_move_inverse(move):
    return env.inverse_moves[move]
# Function to get all states n moves away from solved state
def get_states_n_moves_away_AC(n):
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
            #env.finger_ix_fast(get_move_inverse(move))  # Undo the move
    
    env.reset()  # Start from solved state
    dfs(0,env.state.copy())
    return np.array(states)



# Function to hash a cube state
def hash_state(state):
    return hash(tuple(state))

def hash_vectors_torch(vectors: torch.Tensor, mod: int = 2**61-1) -> torch.Tensor:
    """
    Hash batches of vectors containing numbers 0-20
    Input shape: (..., 20) - last dimension must be 20
    Returns: (...) - same shape as input minus last dimension
    """
    #assert ((vectors >= 0) & (vectors <= 20)).all(), "Values must be between 0 and 20"

    # Convert to int64 to prevent overflow during multiplication
    vectors = vectors.to(torch.int64)+2

    # Powers of prime for each position
    #prime = 31
    prime = 101
    n = vectors.shape[-1]
    
    powers = torch.arange(vectors.shape[-1], device=vectors.device, dtype=torch.int64)
    multipliers = prime ** powers

    # Compute hash: sum((num + 1) * prime^pos) % mod
    hash_vals = ((vectors + 1) * multipliers) % mod
    hash_vals = hash_vals.sum(dim=-1) % mod

    return hash_vals



def is_state_in_short_distance(state):
    """
    Check if a given state is in the list of states 4 moves away from solved.
    
    Args:
    state (numpy.ndarray): The cube state to check.
    
    Returns:
    bool: True if the state is in the list, False otherwise.
    """
    hashed_input_state = hash_state(state)
    return hashed_input_state in hashed_states_close_to_finish

# Vectorize is_state_in_short_distance check across states
def is_state_in_short_distance_torch(states):
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
    hashed_states = torch.vmap(hash_vectors_torch)(states).to(device)
    results = torch.isin(hashed_states, (hashed_states_close_to_finish_torch).to(device))
    
    # Convert results to torch tensor
    return results#torch.tensor(results, dtype=torch.bool, device=states.device if isinstance(states, torch.Tensor) else 'cpu')

def is_state_actually_in_list_torch(state):
    # Check if state exists in the tensor of states close to finish
    if len(state.shape) == 1:
        state = state.unsqueeze(0)  # Add batch dimension if single state
    return (state == torch_states_close_to_finish.to(device)).all(dim=1).any()

def is_state_actually_in_list(state):
    # Check if state exists in the array of states close to finish
    if len(state.shape) == 1:
        state = np.expand_dims(state, 0)  # Add batch dimension if single state
    return np.any(np.all(state == states_close_to_finish, axis=1))

# def vmap_is_state_in_short_distance_torch(states):


from torch_scatter import scatter_max
def beam_search_fast(env, model, beam_width=100, max_depth=TrainConfig.max_depth, check_short_distance=False, skip_redundant_moves=True,attempts=1,start_step=0,many_initial_states=None,expensive_pick_best=True):
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
    device = next(model.parameters()).device
    start_time = time.time()
    num_nodes = 0
    current_width = 1
    length_trajectory = max_depth-start_step

    # Pre-allocate tensors
    candidate_states = -1*torch.ones((beam_width,env.state_dim),device=device,dtype=torch.int64)
    if many_initial_states is not None:
        print("using initial states given")
        num_states = many_initial_states.shape[0]
        candidate_states[0:num_states,:] = torch.tensor(many_initial_states,device=device,dtype=torch.int64)
    else:
        print("using initial state of env")
        candidate_states[0,:] = torch.tensor(env.state,device=device,dtype=torch.int64)
    candidate_paths = -1*torch.ones((beam_width,length_trajectory),device=device,dtype=torch.int64)
    candidate_log_values = torch.zeros((beam_width),device=device,dtype=torch.float64)
    paths_flat = -1*torch.ones((beam_width*env.num_moves,length_trajectory),device=device,dtype=torch.int64)
    states_flat = -1*torch.ones((beam_width*env.num_moves,candidate_states.shape[-1]),device=device,dtype=torch.int64)
    log_values_flat = torch.zeros((beam_width*env.num_moves),device=device,dtype=torch.float64)
    sorted_indices = torch.arange(beam_width*env.num_moves,device=device,dtype=torch.int64)
    batch_p = torch.zeros((beam_width,env.num_moves),device=device,dtype=torch.float64)
    inverse_indices = torch.zeros((beam_width*env.num_moves),device=device,dtype=torch.int64)

    # Search up to max depth
    for attempt in range(attempts):
        for depth in range(start_step,max_depth):
            num_nodes += current_width

            # Get model predictions
            with torch.no_grad():
                t = torch.tensor(1 - depth / max_depth, device=device).repeat(current_width)
                batch_scores = model(candidate_states[:current_width],t)
            
            new_states = apply_all_moves_to_all_states_torch_jit(candidate_states[:current_width])
            batch_probs = get_probabilities_from_model_output(batch_scores,candidate_states[:current_width],new_states,env,weight_contraction=TrainConfig.weight_contraction,total_relator_weight=TrainConfig.total_relator_weight,double_weight=TrainConfig.double_weight)
            log_batch_p = torch.log(batch_probs)
            
            # Update log values efficiently
            log_values_flat[:current_width*env.num_moves] = candidate_log_values[:current_width].tile(env.num_moves) + log_batch_p.transpose(0,1).reshape(-1)
            
            # Update states and paths
            states_flat[:current_width*env.num_moves] = new_states.reshape(-1, candidate_states.shape[-1])
            paths_flat[:current_width*env.num_moves] = candidate_paths[:current_width].tile((env.num_moves,1))
            paths_flat[:current_width*env.num_moves,attempt*length_trajectory+(depth-start_step)] = torch.arange(env.num_moves,device=device).repeat_interleave(current_width)

            if expensive_pick_best:
                # Find unique states and get max values
                inverse_indices[:current_width*env.num_moves] = torch.unique(states_flat[:current_width*env.num_moves], dim=0, return_inverse=True, return_counts=False)[1]
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
            check_states = is_state_in_short_distance_torch(candidate_states[:current_width])
            if torch.any(check_states):
                positive_indices = torch.where(check_states)[0]
                for idx in positive_indices:
                    state = candidate_states[idx]
                    if is_state_actually_in_list_torch(state):
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



def calculate_validation_loss(dataloader_val, model, num_batches=10):
    """
    Calculate validation loss over specified number of batches.
    
    Args:
        dataloader_val: Validation dataloader
        model: Model to evaluate
        num_batches: Number of batches to use for validation
    
    Returns:
        Average loss over all batches
    """
    device = next(model.parameters()).device
    model = model.to(device)
    model.eval() # Put in eval mode - for dropout/norm if present
    
    total_loss = 0
    total_trajectories = 0
    
    # Process batches in chunks of 10 to avoid memory issues
    chunk_size = min(10, num_batches)
    num_chunks = (num_batches + chunk_size - 1) // chunk_size
    
    with torch.no_grad():
        for chunk in range(num_chunks):
            x_testing = []
            start_batch = chunk * chunk_size
            end_batch = min((chunk + 1) * chunk_size, num_batches)
            
            # Collect batches for this chunk
            for i, batch in enumerate(dataloader_val):
                if i >= end_batch:
                    break
                if i >= start_batch:
                    x_testing.append(batch[0])
                    
            if not x_testing:  # Skip if no batches collected
                continue
                
            x_testing = torch.cat(x_testing, dim=0).to(device)
            num_steps = TrainConfig.max_depth
            batch_t = torch.linspace(1/num_steps, 1, steps=num_steps, device=device)
            batch_t = batch_t.unsqueeze(0).repeat(x_testing.shape[0], 1)
            
            loss = custom_loss_discrete(model, x_testing, batch_t)
            total_loss += loss.item() * x_testing.shape[0]
            total_trajectories += x_testing.shape[0]
            
            # Free memory
            del x_testing, batch_t
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_trajectories
    print(f"Validation loss: {avg_loss:.4f} (calculated over {total_trajectories} trajectories)")
    return avg_loss

