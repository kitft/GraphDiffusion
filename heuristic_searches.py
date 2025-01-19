import time
import torch
import numpy as np
from envsAndScramble import *
from NN_models import *
from cube_training import *


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






# Initialize the Rubik's Cube environment
# cube = Cube3()

def get_move_inverse(move):
    return (move[0]+"'") if len(move)==1 else move[0]
# Function to get all states n moves away from solved state
def get_states_n_moves_away_CUBE(env,n):
    cube=Cube2() if env.__class__.__name__ == 'Cube2' else Cube3()
    if n > 6:
        raise ValueError("n should not be greater than 6 to avoid excessive computation time")
    
    states = []
    def dfs(depth):
        if depth == n:
            states.append(cube.state.copy())
            return
        for move in cube.moves:
            cube.finger(move)
            dfs(depth + 1)
            cube.finger(get_move_inverse(move))  # Undo the move
    
    cube.reset()  # Start from solved state
    dfs(0)
    return states

def get_move_inverse_GROUP(move):
    if move%2==0:
        return move+1
    else:
        return move-1


def get_states_n_moves_away_GROUP(env,n):
    if n > 6:
        raise ValueError("n should not be greater than 6 to avoid excessive computation time")
    state_init = env.identity
    state = state_init.copy()
    states = []
    def dfs(depth,state):
        if depth == n:
            states.append(state.copy())
            return
        for move in range(env.num_moves):
            state = env.apply_move_fast(state,move)
            dfs(depth + 1,state)
            state = env.apply_move_fast(state,get_move_inverse_GROUP(move))  # Undo the move
    
    dfs(0,state)
    return states





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
    vectors = vectors.to(torch.int64)

    # Powers of prime for each position
    prime = 31
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
    # Check if state exists in the list of states close to finish
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()
    state = np.array(state)
    if len(state.shape) == 1:
        state = state[np.newaxis, :]  # Add batch dimension if single state
    return np.any(np.all(state == states_close_to_finish, axis=1))


from torch_scatter import scatter_max
#!pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
def beam_search_fast_AC(env, model, beam_width=100, max_depth=TrainConfig.max_depth, check_short_distance=False, skip_redundant_moves=True,attempts=1,start_step=0,many_initial_states=None,expensive_pick_best=True):
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

    STICKER_SOURCE_IX = torch.tensor(env.STICKER_SOURCE_IX,device=device,dtype=torch.int64)
    STICKER_TARGET_IX = torch.tensor(env.STICKER_TARGET_IX,device=device,dtype=torch.int64)
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
    #log_values_for_sort_flat= -10000000*torch.ones((beam_width*env.num_moves),device=device,dtype=torch.float64)
    #log_batch_probs = torch.zeros((beam_width*env.num_moves),device=device,dtype=torch.float64)
    # Search up to max depth
    for attempt in range(attempts):
        #if attempt>0:
        #    candidate_paths = torch.cat((candidate_paths,-1*torch.ones((beam_width,max_depth),device=device,dtype=torch.int64)))
        for depth in range(start_step,max_depth):
            #print(f"CW: {current_width}, mem allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, mem peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB, mem reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            #if not candidates:
            #    break
            num_nodes += current_width

            # Get model predictions
            with torch.no_grad():
                t = torch.tensor(1 - depth / max_depth, device=device).repeat(current_width)
                #print("c",candidate_states[current_width-1:current_width+1])
                #print(candidate_states[:current_width].shape)
                #print(candidate_states.shape)
                batch_p[:current_width] = model(candidate_states[:current_width],t)
                # Take log of probabilities to avoid underflow
                batch_p[:current_width] = torch.log(batch_p[:current_width] / batch_p[:current_width].sum(axis=-1, keepdims=True))

            #log_batch_p = torch.log(batch_probs)

            #new_log_values = candidate_log_values[:current_width].unsqueeze(1).repeat(1,env.num_moves) + log_batch_p
            #log_values_flat[:current_width*env.num_moves] = candidate_log_values[:current_width].repeat_interleave(env.num_moves) + batch_p[:current_width].view(-1)
            log_values_flat[:current_width*env.num_moves] = candidate_log_values[:current_width].tile(env.num_moves) + batch_p[:current_width].transpose(0,1).reshape(-1)  # Transpose and reshape to flatten column-by-column
            
            # Apply all moves to all states
            #states_flat[:current_width*env.num_moves] = apply_all_moves_to_all_states(candidate_states[:current_width],STICKER_SOURCE_IX,STICKER_TARGET_IX).view(-1, candidate_states.shape[-1])
            #each move occupies current_width rows in turn in states_flat.
            states_flat[:] = apply_all_moves_to_all_states_no_reshape(candidate_states[:current_width],states_flat,STICKER_SOURCE_IX,STICKER_TARGET_IX)
            # Update candidates_next_depth with new states
            #paths_flat[:current_width*env.num_moves] = candidate_paths[:current_width].repeat_interleave(env.num_moves,dim=0)
            paths_flat[:current_width*env.num_moves] = candidate_paths[:current_width].tile((env.num_moves,1))
            #paths_flat[:current_width*env.num_moves,depth] = torch.arange(env.num_moves,device=device).tile((current_width,))#.repeat(current_width,1)
            #this does 1,1,1,1,1,1,2,2,2,2,2 etc
            paths_flat[:current_width*env.num_moves,attempt*length_trajectory+(depth-start_step)] = torch.arange(env.num_moves,device=device).repeat_interleave(current_width)
            #next_paths[:current_width,:,depth] = torch.arange(env.num_moves,device=device).unsqueeze(0)#.repeat(current_width,1)
            #next_paths[:current_width,:,depth] = torch.arange(env.num_moves,device=device).unsqueeze(0)#.repeat(current_width,1)
            # Create candidates_next_depth more efficiently by avoiding list comprehension
            # and minimizing CPU/GPU transfers
            if expensive_pick_best:
         
                # Find unique states and corresponding indices
                inverse_indices[:current_width*env.num_moves] = torch.unique(states_flat[:current_width*env.num_moves], dim=0, return_inverse=True, return_counts=False)[1]

                # Get max values for each unique state
                __, max_indices = scatter_max(log_values_flat[:current_width*env.num_moves], inverse_indices[:current_width*env.num_moves], dim=0)
                current_width = min(beam_width,  len(max_indices))
            
                candidate_log_values[:current_width], sorted_indices[:current_width] = torch.topk(log_values_flat[max_indices], k=current_width, largest=True,sorted=False)

                #sorted_indices[:current_width] = torch.argsort(log_values_flat[max_indices], descending=True)[:current_width]
                
                max_sorted = max_indices[sorted_indices[:current_width]]
                #del max_indices, cw_buffer, discard_buffer
                candidate_states[:current_width] = states_flat[max_sorted]
                candidate_paths[:current_width] = paths_flat[max_sorted]
                candidate_log_values[:current_width] = log_values_flat[:current_width]
                candidate_log_values[:current_width] = log_values_flat[max_sorted]
         
            else:
                # Sort by value and keep top beam_width candidates
                current_width = min(beam_width, current_width*env.num_moves)
                sorted_indices[:current_width] = torch.argsort(log_values_flat[:current_width], descending=True)[:current_width]
 
                candidate_states[:current_width] = states_flat[sorted_indices[:current_width]]
                candidate_paths[:current_width] = paths_flat[sorted_indices[:current_width]]
                candidate_log_values[:current_width] = log_values_flat[sorted_indices[:current_width]]
      

            check_states = is_state_in_short_distance_torch(candidate_states[:current_width])
            if torch.any(check_states):
                # Get indices of positive check_states
                positive_indices = torch.where(check_states)[0]
                
                # Loop through positive indices
                for idx in positive_indices:
                    # Get state and verify with exact check
                    state = candidate_states[idx]
                    if is_state_actually_in_list_torch(state):
                        # Found valid solution
                        good_path = candidate_paths[idx]
                        good_path = good_path[:(attempt)*length_trajectory+(depth-start_step)+1]
                        good_path = good_path.to('cpu').detach().numpy()#MUST BE MOVED TO CPU - otherwise memory accumulates on each iteration
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
    }, None,attempt#,candidate_log_values



