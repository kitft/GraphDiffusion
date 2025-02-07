import torch 
from AC_env import *
import random




def OLD_apply_all_moves_to_all_states_torch(states):
    """
    Applies all possible moves to each state in the batch of states using PyTorch.
    
    Args:
        states: Tensor of shape (batch_size, state_size) containing multiple cube states
        
    Returns:
        Tensor of shape (batch_size, num_moves, state_size) containing all resulting states
    """
    batch_size = states.shape[0]
    num_moves = 12
    
    # Create output tensor with shape (batch_size, num_moves, state_size)
    all_states = torch.zeros(batch_size, num_moves, states.shape[1], dtype=states.dtype, device=states.device)
    
    # For each move
    for move_idx in range(num_moves):
        # Copy original states
        #all_states[:,move_idx] = states.clone()
        # Apply move to this slice
        all_states[:,move_idx] = finger_ix_fast_vec_torch(states, move_idx)
        
    return all_states
def apply_all_moves_to_all_states_torch(states):
    """
    Applies all possible moves to each state in the batch of states using PyTorch.
    Uses same batched approach as finger_ix_fast_vec_torch_list_of_moves.
    
    Args:
        states: Tensor of shape (batch_size, state_size) containing multiple cube states
        
    Returns:
        Tensor of shape (batch_size, num_moves, state_size) containing all resulting states
    """
    batch_size = states.shape[0]
    num_moves = 12
    device = states.device
    max_rel_length = states.shape[-1]//2
    
    # Create output tensor
    all_states = torch.zeros(batch_size, num_moves, states.shape[1], dtype=states.dtype, device=device)
    
    # Split into r0 and r1 components
    r0 = states[:, :max_rel_length]
    r1 = states[:, max_rel_length:]
    
    # Create minus reversed versions once for basic moves
    minusreverse_r0 = minus_reverse_torch(r0)
    minusreverse_r1 = minus_reverse_torch(r1)
    
    # Handle moves 0-3 efficiently by batching
    # Prepare inputs for single combine_relator_and_relator2_torch_vmap call
    first_relators = torch.zeros(batch_size * 4, max_rel_length, device=device, dtype=states.dtype)
    second_relators = torch.zeros_like(first_relators)
    
    # Set up which relators get combined based on move type
    first_relators[0::4] = r1  # move 0: r1 for r_1 --> r_1 r_0
    first_relators[1::4] = r0  # move 1: r0 for r_0 --> r_0 r_1^{-1}
    first_relators[2::4] = r1  # move 2: r1 for r_1 --> r_1 r_0^{-1}
    first_relators[3::4] = r0  # move 3: r0 for r_0 --> r_0 r_1
    
    second_relators[0::4] = r0             # move 0: r0
    second_relators[1::4] = minusreverse_r1  # move 1: -r1^rev
    second_relators[2::4] = minusreverse_r0  # move 2: -r0^rev
    second_relators[3::4] = r1             # move 3: r1
    
    # Single call to combine relators
    combined = combine_relator_and_relator2_torch_vmap(first_relators, second_relators)
    
    # Update outputs based on move type
    all_states[:, 0] = torch.cat([r0, combined[0::4]], dim=-1)  # move 0
    all_states[:, 1] = torch.cat([combined[1::4], r1], dim=-1)  # move 1
    all_states[:, 2] = torch.cat([r0, combined[2::4]], dim=-1)  # move 2
    all_states[:, 3] = torch.cat([combined[3::4], r1], dim=-1)  # move 3
    
    # Handle conjugation moves 4-11
    conjugations = {
        4: (1, torch.tensor([1,-1], device=device)),  # r1, x0^-1 r1 x0
        5: (0, torch.tensor([2,-2], device=device)),  # r0, x1^-1 r0 x1
        6: (1, torch.tensor([2,-2], device=device)),  # r1, x1^-1 r1 x1
        7: (0, torch.tensor([-1,1], device=device)),  # r0, x0 r0 x0^-1
        8: (1, torch.tensor([-1,1], device=device)),  # r1, x0 r1 x0^-1
        9: (0, torch.tensor([-2,2], device=device)),  # r0, x1 r0 x1^-1
        10: (1, torch.tensor([-2,2], device=device)), # r1, x1 r1 x1^-1
        11: (0, torch.tensor([1,-1], device=device))  # r0, x0^-1 r0 x0
    }
    
    for move in range(4, 12):
        rel_idx, conj_pattern = conjugations[move]
        if rel_idx == 0:
            r0_out = contract_endpoints_of_relator_flexible_torch_vmap(r0, conj_pattern)
            all_states[:,move] = torch.cat([r0_out, r1], dim=-1)
        else:
            r1_out = contract_endpoints_of_relator_flexible_torch_vmap(r1, conj_pattern)
            all_states[:,move] = torch.cat([r0, r1_out], dim=-1)
            
    return all_states






def OLD_finger_ix_fast_vec_torch(states: torch.Tensor, ix: int, simplify: bool = True) -> torch.Tensor:
    """
    PyTorch version of finger_ix_fast_vec that operates on multiple states at once.
    
    Args:
        states: Tensor of shape (batch_size, state_size) containing multiple cube states
        ix: Move index to apply to all states
        
    Returns:
        Updated states after applying the move
    """
    max_relator_length = states.shape[1] // 2
    # Split states into r0 and r1 components
    r0 = states[:, :max_relator_length]
    r1 = states[:, max_relator_length:]
    r0_nonzero_mask = r0 != 0
    r1_nonzero_mask = r1 != 0
    r0_counts = r0_nonzero_mask.sum(dim=1)
    r1_counts = r1_nonzero_mask.sum(dim=1)
    r0_indices = torch.arange(max_relator_length, device=states.device)[None, :]
    r1_indices = torch.arange(max_relator_length, device=states.device)[None, :]

    # Create output states tensor
    new_states = states.clone()

    if ix == 0:  # r_1 --> r_1 r_0
        mask = (r0_counts + r1_counts) <= max_relator_length
        new_r1 = torch.zeros_like(states)
        new_r1[:,:max_relator_length] = r1
        r0_start_positions = r1_counts[:, None]
        r0_newindices = r0_indices + r0_start_positions
        new_r1.scatter_(1, r0_newindices, r0 * r0_nonzero_mask)
        new_states[mask, max_relator_length:] = new_r1[mask,:max_relator_length]

    elif ix == 1:  # r_0 --> r_0 r_1^{-1}
        mask = (r0_counts + r1_counts) <= max_relator_length
        new_r0 = torch.zeros_like(states)
        new_r0[:,:max_relator_length] = r0
        #reversed_r1 = -1 * torch.flip(r1 * r1_nonzero_mask, [1])
        reversed_r1 = -1* reverse_padded_vectors_torch(r1)
        r1_start_positions = r0_counts[:, None]
        r1_newindices = r0_indices + r1_start_positions
        new_r0.scatter_(1, r1_newindices, reversed_r1)
        new_states[mask, :max_relator_length] = new_r0[mask,:max_relator_length]

    elif ix == 2:  # r_1 --> r_1 r_0^{-1}
        mask = (r0_counts + r1_counts) <= max_relator_length
        new_r1 = torch.zeros_like(states)
        new_r1[:,:max_relator_length] = r1
        #reversed_r0 = -1 * torch.flip(r0 * r0_nonzero_mask, [1])
        reversed_r0 = -1* reverse_padded_vectors_torch(r0)
        r0_start_positions = r1_counts[:, None]
        r0_newindices = r1_indices + r0_start_positions
        new_r1.scatter_(1, r0_newindices, reversed_r0)
        new_states[mask, max_relator_length:] = new_r1[mask,:max_relator_length]

    elif ix == 3:  # r_0 --> r_0 r_1
        mask = (r0_counts + r1_counts) <= max_relator_length
        new_r0 = torch.zeros_like(states)
        new_r0[:,:max_relator_length] = r0
        r1_start_positions = r0_counts[:, None]
        r1_newindices = r0_indices + r1_start_positions
        new_r0.scatter_(1, r1_newindices, r1 * r1_nonzero_mask)
        new_states[mask, :max_relator_length] = new_r0[mask,:max_relator_length]

    elif ix == 4:  # r_1 --> x_0^{-1} r_1 x_0
        mask = (r1_counts + 2) <= max_relator_length
        new_r1 = torch.zeros((r1.shape[0], max_relator_length+2), device=states.device,dtype=states.dtype)
        new_r1[:,1:max_relator_length] = r1[:,:-1]
        new_r1[:, 0] = -1
        new_r1[torch.arange(len(r1_counts), device=states.device,dtype=torch.int64), r1_counts + 1] = 1
        new_states[mask, max_relator_length:] = new_r1[mask, :max_relator_length]
        unmasked = ~mask
        if torch.any(unmasked):
           contracted_r1 = contract_endpoints_torch(r1[unmasked], torch.tensor([1,-1], device=states.device))
           new_states[unmasked, max_relator_length:] = contracted_r1

    elif ix == 5:  # r_0 --> x_1^{-1} r_0 x_1
        mask = (r0_counts + 2) <= max_relator_length
        new_r0 = torch.zeros((r0.shape[0], max_relator_length+2), device=states.device, dtype=states.dtype)
        new_r0[:,1:max_relator_length] = r0[:,:-1]
        new_r0[:, 0] = -2
        new_r0[torch.arange(len(r0_counts), device=states.device), r0_counts + 1] = 2
        new_states[mask, :max_relator_length] = new_r0[mask, :max_relator_length]
        unmasked = ~mask
        if torch.any(unmasked):
           contracted_r0 = contract_endpoints_torch(r0[unmasked], torch.tensor([2,-2], device=states.device))
           new_states[unmasked, :max_relator_length] = contracted_r0

    elif ix == 6:  # r_1 --> x_1^{-1} r_1 x_1
        mask = (r1_counts + 2) <= max_relator_length
        new_r1 = torch.zeros((r1.shape[0], max_relator_length+2), device=states.device,dtype=states.dtype)
        new_r1[:,1:max_relator_length] = r1[:,:-1]
        new_r1[:, 0] = -2
        new_r1[torch.arange(len(r1_counts), device=states.device), r1_counts + 1] = 2
        new_states[mask, max_relator_length:] = new_r1[mask, :max_relator_length]
        unmasked = ~mask
        if torch.any(unmasked):
           contracted_r1 = contract_endpoints_torch(r1[unmasked], torch.tensor([2,-2], device=states.device))
           new_states[unmasked, max_relator_length:] = contracted_r1

    elif ix == 7:  # r_0 --> x_0 r_0 x_0^{-1}
        mask = (r0_counts + 2) <= max_relator_length
        new_r0 = torch.zeros((r0.shape[0], max_relator_length+2), device=states.device,dtype=states.dtype)
        new_r0[:,1:max_relator_length] = r0[:,:-1]
        new_r0[:, 0] = 1
        new_r0[torch.arange(len(r0_counts), device=states.device), r0_counts + 1] = -1
        new_states[mask, :max_relator_length] = new_r0[mask, :max_relator_length]
        unmasked = ~mask
        if torch.any(unmasked):
           contracted_r0 = contract_endpoints_torch(r0[unmasked], torch.tensor([-1,1], device=states.device))
           new_states[unmasked, :max_relator_length] = contracted_r0

    elif ix == 8:  # r_1 --> x_0 r_1 x_0^{-1}
        mask = (r1_counts + 2) <= max_relator_length
        new_r1 = torch.zeros((r1.shape[0], max_relator_length+2), device=states.device,dtype=states.dtype)
        new_r1[:,1:max_relator_length] = r1[:,:-1]
        new_r1[:, 0] = 1
        new_r1[torch.arange(len(r1_counts), device=states.device), r1_counts + 1] = -1
        new_states[mask, max_relator_length:] = new_r1[mask, :max_relator_length]
        unmasked = ~mask
        if torch.any(unmasked):
           contracted_r1 = contract_endpoints_torch(r1[unmasked], torch.tensor([-1,1], device=states.device))
           new_states[unmasked, max_relator_length:] = contracted_r1

    elif ix == 9:  # r_0 --> x_1 r_0 x_1^{-1}
        mask = (r0_counts + 2) <= max_relator_length
        new_r0 = torch.zeros((r0.shape[0], max_relator_length+2), device=states.device,dtype=states.dtype)
        new_r0[:,1:max_relator_length] = r0[:,:-1]
        new_r0[:, 0] = 2
        new_r0[torch.arange(len(r0_counts), device=states.device), r0_counts + 1] = -2
        new_states[mask, :max_relator_length] = new_r0[mask, :max_relator_length]
        unmasked = ~mask
        if torch.any(unmasked):
           contracted_r0 = contract_endpoints_torch(r0[unmasked], torch.tensor([-2,2], device=states.device))
           new_states[unmasked, :max_relator_length] = contracted_r0

    elif ix == 10:  # r_1 --> x_1 r_1 x_1^{-1}
        mask = (r1_counts + 2) <= max_relator_length
        new_r1 = torch.zeros((r1.shape[0], max_relator_length+2), device=states.device,dtype=states.dtype)
        new_r1[:,1:max_relator_length] = r1[:,:-1]
        new_r1[:, 0] = 2
        new_r1[torch.arange(len(r1_counts), device=states.device), r1_counts + 1] = -2
        new_states[mask, max_relator_length:] = new_r1[mask, :max_relator_length]
        unmasked = ~mask
        if torch.any(unmasked):
           contracted_r1 = contract_endpoints_torch(r1[unmasked], torch.tensor([-2,2], device=states.device)    )
           new_states[unmasked, max_relator_length:] = contracted_r1

    elif ix == 11:  # r_0 --> x_0^{-1} r_0 x_0
        mask = (r0_counts + 2) <= max_relator_length
        new_r0 = torch.zeros((r0.shape[0], max_relator_length+2), device=states.device,dtype=states.dtype)
        new_r0[:,1:max_relator_length] = r0[:,:-1]
        new_r0[:, 0] = -1
        new_r0[torch.arange(len(r0_counts), device=states.device), r0_counts + 1] = 1
        new_states[mask, :max_relator_length] = new_r0[mask, :max_relator_length]
        unmasked = ~mask
        if torch.any(unmasked):
           contracted_r0 = contract_endpoints_torch(r0[unmasked], torch.tensor([1,-1], device=states.device)   )
           new_states[unmasked, :max_relator_length] = contracted_r0

    if simplify:
        new_states = simplify_state_vec_torch(new_states)
    return new_states
    
    
def simplify_state_vec_torch(states):
    """PyTorch version of simplify_state_vec that operates on a batch of states."""
    # Split states into r0 and r1
    max_relator_length = states.shape[-1] // 2
    r0 = states[:, :max_relator_length]
    r1 = states[:, max_relator_length:]
        
    # Simplify both halves of the states
    r0_simplified = _iterative_simplify_vectorized_torch(r0)
    r1_simplified = _iterative_simplify_vectorized_torch(r1)
    
    # Recombine
    simplified_states = torch.hstack([r0_simplified, r1_simplified])
    return simplified_states

def _iterative_simplify_vectorized_torch(relators):
    """
    Vectorized simplification of relator batches using preallocated tensors and fused operations.
    Only processes rows that still have valid cancellation pairs.
    """
    batch_size, max_len = relators.shape
    device = relators.device
    
    # Preallocate tensors we'll reuse
    current = relators.clone()
    buffer = torch.zeros(batch_size, max_len*2, dtype=relators.dtype, device=device)
    positions = torch.arange(max_len*2, device=device).expand(batch_size, -1)
    
    # Track which rows still need processing
    active_rows = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    while active_rows.any():
        # Copy current into first half of buffer (only for active rows)
        buffer[active_rows, :max_len] = current[active_rows]
        buffer[active_rows, max_len:] = 0
        
        # Find cancellations using fused operations (only for active rows)
        nonzero = buffer[active_rows] != 0
        sums = buffer[active_rows, :-1] + buffer[active_rows, 1:]
        valid_pairs = (sums == 0) & nonzero[:, :-1] & nonzero[:, 1:]
        
        # Update active rows - if a row has no valid pairs, we're done with it
        has_valid = valid_pairs.any(dim=1)
        active_indices = torch.nonzero(active_rows).squeeze(1)
        active_rows[active_indices[~has_valid]] = False
        
        if not has_valid.any():
            break
            
        # Get indices for remaining active rows
        still_active = active_indices[has_valid]
        
        # Find first cancellation per remaining active row
        cancel_idx = torch.argmax(valid_pairs[has_valid].to(torch.int8), dim=1)
        
        # Mask out cancelled pairs
        to_remove = (positions[still_active] == cancel_idx.unsqueeze(1)) | (positions[still_active] == (cancel_idx + 1).unsqueeze(1))
        keep_mask = ~to_remove
        
        # Compact remaining elements
        kept = torch.where(keep_mask, buffer[still_active], torch.zeros_like(buffer[still_active]))
        nonzero = kept != 0
        sort_keys = (~nonzero) * (max_len*2) + positions[still_active]
        sorted_indices = sort_keys.argsort(dim=1)
        current[still_active] = torch.gather(kept, 1, sorted_indices)[:, :max_len]

    return current

# test_vecs = torch.tensor([[-1,  1,  1,  1, -2,  1,  1, -2, -2, -1,  2,  1,  2, -1,  2, -1,  2, -1,
#          -2, -1,  2, -1, -1, -1,  1],
#         [-1,  1,  1,  1, -2,  1,  1, -2, -2, -1,  2,  1,  2, -1,  2, -1,  2, -1,
#          -2, -1,  2, -1, -1, -1,  1],
#         [-1,  1,  1,  1, -2,  1,  1, -2, -2, -1,  2,  1,  2, -1,  2, -1,  2, -1,
#          -2, -1,  2, -1, -1, -1,  1]])[0:3]

# test_result = _iterative_simplify_vectorized_torch(test_vecs)
# print("Original test vectors:")
# print(test_vecs)
# print("\nSimplified result:")
# print(test_result)

def reverse_padded_vectors_torch(padded_vectors):
    # Create a mask for non-zero elements
    mask = padded_vectors != 0

    # Get the counts of non-zero elements per row
    lengths = mask.sum(dim=1)

    # Create indices array for each row
    row_indices = torch.arange(padded_vectors.shape[0], device=padded_vectors.device)[:, None]
    col_indices = torch.arange(padded_vectors.shape[1], device=padded_vectors.device)[None, :]

    # Create reversed indices for non-zero elements
    reversed_indices = lengths[:, None] - 1 - col_indices

    # Create mask for valid reversed indices
    valid_mask = (reversed_indices >= 0) & mask

    # Create output tensor of zeros
    result = torch.zeros_like(padded_vectors)

    # Fill in reversed values
    result = torch.where(valid_mask,
        padded_vectors.gather(1, torch.clamp(reversed_indices, min=0)),
        torch.zeros_like(padded_vectors))

    return result


def contract_endpoints_torch(states, pattern):
    """
    Removes the first and last nonzero elements of each state if they match the given pattern.

    Args:
        states: Tensor of shape (batch_size, state_size) containing states to check
        pattern: List/array of [first,last] values to match against endpoints

    Returns:
        New states with endpoints removed where pattern matched
    """
    # Get nonzero masks and counts
    nonzero_mask = states != 0
    nonzero_counts = torch.sum(nonzero_mask, dim=1)

    # Get first nonzero element for each state
    first_nonzero_indices = torch.argmax(nonzero_mask.long(), dim=1)
    first_elements = states[torch.arange(len(states)), first_nonzero_indices]

    # Get last nonzero element for each state
    # Flip the mask and find first True to get last nonzero position
    last_nonzero_indices = states.shape[1] - 1 - torch.argmax(torch.fliplr(nonzero_mask).long(), dim=1)
    last_elements = states[torch.arange(len(states)), last_nonzero_indices]

    # Find which states can be contracted
    can_contract = (first_elements == pattern[0]) & (last_elements == pattern[1])

    # Create output tensor
    new_states = states.clone()

    if torch.any(can_contract):
        # Get indices of states that can be contracted
        contract_indices = torch.where(can_contract)[0]

        # For each state that can be contracted
        for idx in contract_indices:
            # Remove first and last elements by shifting everything left
            nonzero_count = nonzero_counts[idx]
            new_state = torch.zeros_like(states[idx])
            new_state[:nonzero_count-2] = states[idx][1:nonzero_count-1]
            new_states[idx] = new_state

    return new_states

def minus_reverse_torch(r1):
    # r1 has shape (batch_size, seq_len)
    minusreverser1 = -torch.flip(r1, dims=[-1])
    nonzero_mask = minusreverser1 != 0
    nonzero_counts = torch.sum(nonzero_mask, dim=1)
    minusreverser1_full = torch.zeros_like(r1)
    
    # Create indices for placing nonzeros
    batch_size = r1.shape[0]
    batch_indices = torch.arange(batch_size,device=r1.device).unsqueeze(1)
    seq_indices = torch.arange(r1.shape[1],device=r1.device).unsqueeze(0)
    
    # Place nonzeros at start using mask
    valid_indices = seq_indices < nonzero_counts.unsqueeze(1)
    minusreverser1_full[valid_indices] = minusreverser1[nonzero_mask][:(valid_indices.sum())]
    
    return minusreverser1_full
def left_justify_states(rs):
    """
    Pushes all nonzero elements to the left and zeros to the right.
    
    Args:
        rs: Tensor of shape (batch_size, state_size) containing states to justify
        
    Returns:
        States with nonzeros on left and zeros on right, same shape as input
    """
    # Get nonzero values and their original positions
    nonzero_mask = rs != 0
    nonzero_values = rs[nonzero_mask]
    nonzero_counts = torch.sum(nonzero_mask, dim=1)
    justified = torch.zeros_like(rs)
    batch_size = rs.shape[0]
    seq_indices = torch.arange(rs.shape[1],device=rs.device).unsqueeze(0)
    
    # Place all nonzeros at start, pushing zeros to right
    valid_indices = seq_indices < nonzero_counts.unsqueeze(1)
    justified[valid_indices] = nonzero_values[:(valid_indices.sum())]
    
    return justified

def right_justify_states(rs):
    """
    Pushes all nonzero elements to the right and zeros to the left
    
    Args:
        rs: Tensor of shape (batch_size, state_size) containing states to justify
        
    Returns:
        States with zeros on left and nonzeros on right, same shape as input
    """
    # Get nonzero values and their original positions
    nonzero_mask = rs != 0
    nonzero_values = rs[nonzero_mask]
    nonzero_counts = torch.sum(nonzero_mask, dim=1)
    justified = torch.zeros_like(rs)
    batch_size = rs.shape[0]
    seq_indices = torch.arange(rs.shape[1],device=rs.device).unsqueeze(0)
    
    # Place nonzeros at end by offsetting indices by sequence length minus nonzero count
    offsets = rs.shape[1] - nonzero_counts
    valid_indices = seq_indices >= offsets.unsqueeze(1)
    justified[valid_indices] = nonzero_values[:(valid_indices.sum())]
    
    return justified

def finger_ix_fast_vec_torch_list_of_moves(states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
    """
    Do relator type moves using PyTorch for a batch of moves 0-3.
    
    Args:
        states: Tensor of shape (batch_size, state_size) containing cube states
        moves: Tensor of shape (batch_size,) containing move indices 0-11
        
    Returns:
        Tensor of same shape as states with moves applied
        
    Notes:
        - Optimized version that batches moves 0-3 together
        - Not a true vmap (vectorized map) implementation
        - Simplification is NOT applied to output states
        - Moves 0-3: Combine relators (r0,r1) in different orders
        - Moves 4-11: Contract endpoints of relators with conjugation
    """
    max_rel_length = states.shape[-1]//2
    r0, r1 = torch.split(states, max_rel_length, dim=-1)
    batch_size = states.shape[0]
    device = states.device

    # Handle moves 0-3 efficiently by batching
    basic_moves_mask = moves < 4
    if basic_moves_mask.any():
        # Get indices where basic moves occur
        basic_move_indices = basic_moves_mask.nonzero().squeeze(-1)
        basic_moves = moves[basic_move_indices]
        
        # Prepare relators for combination
        r0_basic = r0[basic_move_indices]
        r1_basic = r1[basic_move_indices]
        
        # Create minus reversed versions once
        minusreverse_r0 = minus_reverse_torch(r0_basic)
        minusreverse_r1 = minus_reverse_torch(r1_basic)
        
        # Create output tensors
        r0_out = r0.clone()
        r1_out = r1.clone()
        
        # Handle each basic move type
        move_0_mask = basic_moves == 0  # r_1 --> r_1 r_0
        move_1_mask = basic_moves == 1  # r_0 --> r_0 r_1^{-1}
        move_2_mask = basic_moves == 2  # r_1 --> r_1 r_0^{-1}
        move_3_mask = basic_moves == 3  # r_0 --> r_0 r_1
        # Combine all relators in one batch operation
        # Prepare inputs for single combine_relator_and_relator2_torch_vmap call
        first_relators = torch.zeros_like(r0_basic)
        second_relators = torch.zeros_like(r0_basic)
        
        # Set up which relators get combined based on move type
        first_relators[move_0_mask] = r1_basic[move_0_mask]  # r1 for move 0
        first_relators[move_1_mask] = r0_basic[move_1_mask]  # r0 for move 1  
        first_relators[move_2_mask] = r1_basic[move_2_mask]  # r1 for move 2
        first_relators[move_3_mask] = r0_basic[move_3_mask]  # r0 for move 3

        second_relators[move_0_mask] = r0_basic[move_0_mask]  # r0 for move 0
        second_relators[move_1_mask] = minusreverse_r1[move_1_mask]  # -r1^rev for move 1
        second_relators[move_2_mask] = minusreverse_r0[move_2_mask]  # -r0^rev for move 2  
        second_relators[move_3_mask] = r1_basic[move_3_mask]  # r1 for move 3

        # Single call to combine relators
        combined = combine_relator_and_relator2_torch_vmap(first_relators, second_relators)
        
        # Update outputs based on move type
        r1_out[basic_move_indices[move_0_mask]] = combined[move_0_mask]
        r0_out[basic_move_indices[move_1_mask]] = combined[move_1_mask]
        r1_out[basic_move_indices[move_2_mask]] = combined[move_2_mask]
        r0_out[basic_move_indices[move_3_mask]] = combined[move_3_mask]

    # Handle conjugation moves 4-11
    conj_moves_mask = ~basic_moves_mask
    if conj_moves_mask.any():
        conj_move_indices = conj_moves_mask.nonzero().squeeze(-1)
        conj_moves = moves[conj_move_indices]
        
        # Create output tensors if not already created
        if not basic_moves_mask.any():
            r0_out = r0.clone()
            r1_out = r1.clone()
            
        # Define conjugation patterns
        conjugations = {
            4: (1, torch.tensor([1,-1], device=device)),  # r1, x0^-1 r1 x0
            5: (0, torch.tensor([2,-2], device=device)),  # r0, x1^-1 r0 x1
            6: (1, torch.tensor([2,-2], device=device)),  # r1, x1^-1 r1 x1
            7: (0, torch.tensor([-1,1], device=device)),  # r0, x0 r0 x0^-1
            8: (1, torch.tensor([-1,1], device=device)),  # r1, x0 r1 x0^-1
            9: (0, torch.tensor([-2,2], device=device)),  # r0, x1 r0 x1^-1
            10: (1, torch.tensor([-2,2], device=device)), # r1, x1 r1 x1^-1
            11: (0, torch.tensor([1,-1], device=device))  # r0, x0^-1 r0 x0
        }
        
        for move in range(4, 12):
            move_mask = conj_moves == move
            if move_mask.any():
                rel_idx, conj_pattern = conjugations[move]
                if rel_idx == 0:
                    r0_out[conj_move_indices[move_mask]] = contract_endpoints_of_relator_flexible_torch_vmap(
                        r0[conj_move_indices[move_mask]], conj_pattern)
                else:
                    r1_out[conj_move_indices[move_mask]] = contract_endpoints_of_relator_flexible_torch_vmap(
                        r1[conj_move_indices[move_mask]], conj_pattern)
    
    return torch.cat([r0_out, r1_out], dim=-1)

def finger_ix_fast_vec_torch(states: torch.Tensor, ix: int) -> torch.Tensor:
    """
    Do relator type moves using PyTorch.
    
    Args:
        states: Tensor of shape (batch_size, state_size) containing cube states
        ix: Integer index specifying which move to apply (0-11)
        
    Returns:
        Tensor of same shape as state with move applied
        
    Notes:
        - Not a true vmap (vectorized map) implementation
        - Simplification is NOT applied to output states
        - Moves 0-3: Combine relators (r0,r1) in different orders
        - Moves 4-11: Contract endpoints of relators with conjugation
    """
    max_rel_length = states.shape[-1]//2
    r0,r1 = torch.split(states,max_rel_length,dim=-1)
    
    # No need to extract nonzero elements since we want full width
    # For minus reverse: flip, negate, and left-justify by removing zeros

    if ix ==0:# r_1 --> r_1 r_0
        r1=combine_relator_and_relator2_torch_vmap(r1,r0)
    elif ix ==1:# r_0 --> r_0 r_1^{-1}
        minusreverser1 = minus_reverse_torch(r1)
        r0=combine_relator_and_relator2_torch_vmap(r0,minusreverser1)
    elif ix ==2:# r_1 --> r_1 r_0^{-1}  
        minusreverser0 = minus_reverse_torch(r0)
        r1=combine_relator_and_relator2_torch_vmap(r1,minusreverser0)
    elif ix ==3:# r_0 --> r_0 r_1
        r0=combine_relator_and_relator2_torch_vmap(r0,r1)
    elif ix == 4:# r_1 --> x_0^{-1} r_1 x_0
        r1=contract_endpoints_of_relator_flexible_torch_vmap(r1,torch.tensor([1,-1],device=r0.device))
    elif ix == 5:# r_0 ---> x_1^{-1} r_0 x_1
        r0=contract_endpoints_of_relator_flexible_torch_vmap(r0,torch.tensor([2,-2],device=r0.device))   
    elif ix == 6:# r_1 --> x_1^{-1} r_1 x_1
        r1=contract_endpoints_of_relator_flexible_torch_vmap(r1,torch.tensor([2,-2],device=r0.device))
    elif ix == 7:# r_0 ---> x_0 r_0 x_0^{-1}
        r0=contract_endpoints_of_relator_flexible_torch_vmap(r0,torch.tensor([-1,1],device=r0.device))
    elif ix == 8:# r_1 --> x_0 r_1 x_0^{-1}
        r1=contract_endpoints_of_relator_flexible_torch_vmap(r1,torch.tensor([-1,1],device=r0.device))
    elif ix == 9:# r_0 --> x_1 r_0 x_1^{-1}
        r0=contract_endpoints_of_relator_flexible_torch_vmap(r0,torch.tensor([-2,2],device=r0.device))
    elif ix == 10:# r_1 --> x_1 r_1 x_1^{-1}
        r1=contract_endpoints_of_relator_flexible_torch_vmap(r1,torch.tensor([-2,2],device=r0.device))
    elif ix == 11:# r_0 --> x_0^{-1} r_0 x_0
        r0=contract_endpoints_of_relator_flexible_torch_vmap(r0,torch.tensor([1,-1],device=r0.device))
    out_state = torch.cat([r0,r1],dim=-1)
    #if ix>=4:
    #    out_state = simplify_state_vec_torch(out_state)
    return out_state

#OLD VERSION OF LISTOF MOVES
def OLD_finger_ix_fast_vec_torch_list_of_moves(states,ixs_all):
    #unique,reverse_ixs = torch.unique(ixs_all,return_inverse=True)#reverse_ixs is the size of ixs_all, and indexes to unique
    out_states = torch.zeros_like(states,device=states.device)
    moves = torch.arange(12,device=states.device)
    for ix in moves:
        mask = (ixs_all==ix)
        if mask.any():
            out_states[mask] = finger_ix_fast_vec_torch(states[mask],ix)
    return out_states

def finger_ix_fast_vec_torch_list_of_moves_all_pick(states,ixs_all):
    #out_states = torch.zeros_like(states,device=states.device)
    out_states = apply_all_moves_to_all_states_torch(states)[torch.arange(states.shape[0]),ixs_all]
    return out_states
# def combine_relator_and_nonzero_relator_torch_single_vec(relator,nonzero_relator):
#     """
#     Combine relator and nonzero_relator by dragging nonzero_relator R along relator r.
#     PyTorch version that operates on a single vector.
#     """
#     # Get lengths
#     R_len_nonzero = len(nonzero_relator)
#     r_len = len(relator)
#     r_len_nonzero = len(relator[torch.nonzero(relator).squeeze()])

#     # Try each valid offset
#     for offset in range(max(0, r_len_nonzero-R_len_nonzero), min(r_len-R_len_nonzero+1, r_len_nonzero+1)):
#         # Create padded version of nonzero_relator
#         padded_R = torch.zeros_like(relator)
#         padded_R[offset:offset+R_len_nonzero] = nonzero_relator
        
#     # Check for valid cancellation
#     overlap_mask = (padded_R != 0) & (relator != 0)
#     sum = relator[overlap_mask] + torch.flip(padded_R[overlap_mask], [0])
    
#     if not torch.any(sum):
#         # Valid cancellation found - construct output
#         out = torch.zeros_like(relator)
#         out[:offset] = relator[:offset]
#         out[offset:offset+len(padded_R[r_len_nonzero:])] = padded_R[r_len_nonzero:]
#         return out

#     return relator  # No valid cancellation found - return unchanged


# def combine_relator_and_relator2_torch_vmap(relators, relators2):
#     """
#     Combine relator and nonzero_relator by dragging nonzero_relator R along relator r.
#     PyTorch version that operates on batches of vectors without loops.
    
#     Args:
#         relators: Tensor of shape (batch_size, state_size) containing relators
#         nonzero_relators: Tensor of shape (batch_size, state_size) containing nonzero relators
        
#     Returns:
#         New relators with combinations applied where valid
#     """
#     batch_size, r_len = relators.shape
    
#     # Get lengths for each relator
#     nonzero_mask = relators != 0
#     r_len_nonzero = torch.sum(nonzero_mask, dim=1)
#     R_len_nonzero = torch.sum(relators2 != 0, dim=1)
    
#     # Calculate valid offset ranges for each relator
#     min_offsets = torch.maximum(torch.zeros_like(r_len_nonzero), 
#                               r_len_nonzero - R_len_nonzero)
#     max_offsets = torch.minimum(r_len - R_len_nonzero + 1,
#                               r_len_nonzero + 1)
    
#     # Create offset matrix (batch_size x max_possible_offsets)
#     max_offset_range = torch.max(max_offsets - min_offsets).item()
#     offset_matrix = (torch.arange(max_offset_range, device=relators.device)
#                     .unsqueeze(0).expand(batch_size, -1))
#     valid_offsets = (offset_matrix >= min_offsets.unsqueeze(1)) & \
#                    (offset_matrix < max_offsets.unsqueeze(1))
    
#     # Create double-width relators and padded_R
#     wide_relators = torch.zeros((batch_size, r_len*2), device=relators.device)
#     wide_relators[:, :r_len] = relators
    
#     #padded_R = torch.zeros((batch_size, r_len), device=relators.device)
#     padded_R = relators2
    
#     # For each offset, shift padded_R and check for cancellations
#     best_offset = torch.full((batch_size,), -1, device=relators.device)
#     best_result = wide_relators.clone()
    
#     for offset in range(max_offset_range):
#         # Only process batches where this offset is valid
#         valid_batch = valid_offsets[:, offset]
#         if not torch.any(valid_batch):
#             continue
            
#         # Shift padded_R by offset
#         shifted_R = torch.roll(padded_R[valid_batch], offset, dims=1)
        
#         # Check for cancellations
#         overlap_mask = (shifted_R != 0) & (wide_relators[valid_batch] != 0)
#         sums = wide_relators[valid_batch][overlap_mask] + torch.flip(shifted_R[overlap_mask], [0])
        
#         # Where sum is zero, we found a valid cancellation
#         valid_cancel = ~torch.any(sums, dim=-1) 
        
#         # Update best results for batches with valid cancellations
#         update_mask = valid_batch.clone()
#         update_mask[valid_batch] &= valid_cancel
        
#         if torch.any(update_mask):
#             # Only update if this is the first valid cancellation found
#             first_time = best_offset[update_mask] == -1
#             if torch.any(first_time):
#                 update_mask[update_mask.clone()] &= first_time
#                 best_offset[update_mask] = offset
#                 best_result[update_mask] = shifted_R[first_time]
    
#     # Return original relator where no valid cancellation was found
#     no_valid = best_offset == -1
#     best_result[no_valid] = relators[no_valid]
    
#     # Truncate back to original width
#     return best_result[:, :r_len]

#def combine_relator_and_relator2_torch_vmap2(relators, relators2):
def combine_relator_and_relator2_torch_vmap(relators, relators2):
    """
    More efficient version that right-justifies relator1, left-justifies relator2,
    and searches for cancellations by pulling them apart.
    ie. do r1 -> r1 r2
    
    Args:
        relators: Tensor of shape (batch_size, state_size) containing relators
        relators2: Tensor of shape (batch_size, state_size) containing second relators
        
    Returns:
        New relators with combinations applied where valid
    """
    batch_size, r_len = relators.shape
    device = relators.device
    #print("relators:",relators) 
    #print("relators2:",relators2)
    # Get nonzero lengths and masks
    rL_nonzero = relators != 0
    rR_nonzero = relators2 != 0
    rL_len_nonzero = torch.sum(rL_nonzero, dim=1)
    rR_len_nonzero = torch.sum(rR_nonzero, dim=1)
    
    # Right justify relator1 by rolling each row
    #roll_amounts = r_len - r1_len_nonzero
    justified_rL = right_justify_states(relators)
    
    # Left justify relator2 (already left justified if using zeros padding)
    justified_rR = relators2
    
    # Calculate valid offset range
    # Offset measures how much overlap between right-justified relator1 and left-justified relator2
    # For example with relator1=xxx00 and relator2=yyy00:
    # offset=3 means maximum overlap: xxx00
    #                                yyy00
    # offset=2 means overlap of 2:   xxx00
    #                                 yyy00 
    # offset=1 means overlap of 1:   xxx00
    #                                  yyy00
    # offset=0 means no overlap:     xxx00
    #                                   yyy00
    max_offsets = torch.minimum(rL_len_nonzero, rR_len_nonzero) # Maximum overlap is min of nonzero lengths
    min_offsets = torch.maximum(torch.zeros_like(rL_len_nonzero), rL_len_nonzero+rR_len_nonzero-r_len)
    max_offset_range = torch.max(max_offsets).item()
    
    # Track best results
    best_offset = torch.full((batch_size,), -1, device=device)
    best_result = justified_rL.clone()
    
    # Start with maximum overlap and reduce
    #print("max_offset_range",max_offset_range)
    for offset in torch.arange(max_offset_range, -1, -1, device=device):
        #print("offset",offset)
        
        # Determine which batches to process at this offset
        valid_batch = (offset <= max_offsets) & (offset >= min_offsets)
        #print("valid_batch",valid_batch)
        if not torch.any(valid_batch):
            continue
            
        # Get overlap region: last offset elements of r1 and first offset elements of r2
        overlap_size = offset
        rL_free = justified_rL[valid_batch, :r_len-overlap_size]
        #print("r1_free",r1_free.shape)
        rL_overlap = justified_rL[valid_batch, -overlap_size:]
        #print("r1_overlap",r1_overlap.shape)
        rR_overlap = justified_rR[valid_batch, :overlap_size]
        #print("r2_overlap",r2_overlap.shape)
        remaining_rR = justified_rR[valid_batch, overlap_size:]
        #print("remaining_r2",remaining_r2.shape)
        if overlap_size == 0:
            # Empty overlap case - no need to calculate sums
            sums = torch.zeros(valid_batch.sum().item(),0, device=device)
            valid_cancel = torch.ones(valid_batch.sum().item(), device=device,dtype=torch.bool)
        else:
            sums = rL_overlap + torch.flip(rR_overlap, [-1])
            valid_cancel = ~torch.any(sums, dim=-1)

        
        
        #print("valid_cancel",valid_cancel)
    
        #print("sums",sums.shape)
        
        # Check for cancellations
        #overlap_mask = (shifted_r2 != 0) & (justified_r1[valid_batch, -overlap_size:] != 0)
        #sums = justified_r1[valid_batch][overlap_mask] + torch.flip(shifted_r2[overlap_mask], [0])
        #print("sums",sums.shape)
        #print("sums",sums)
        
        # Where sum is zero, we found a valid cancellation
        
        
        # Update best results for first valid cancellation found
        # print("\n overlap_size",overlap_size)
        # print("sums.shape",sums.shape)
        # print("r1_overlap",r1_overlap.shape)
        # print("best_offset",best_offset.shape)
        # print("valid_batch",valid_batch.shape)
        # print("valid_cancel",valid_cancel.shape)
        # Create mask for valid batches that haven't found a valid cancellation yet
        update_mask = torch.zeros_like(valid_batch, dtype=torch.bool, device=valid_batch.device)
        update_mask[valid_batch] = valid_cancel & (best_offset[valid_batch] == -1)
       
        if torch.any(update_mask):
            best_offset[update_mask] = offset
            temp_result = torch.zeros(update_mask.sum(),r_len,device=device,dtype=relators.dtype)
            temp_result[:,:min(r_len - overlap_size + r_len-overlap_size,r_len)] = left_justify_states(torch.cat([rL_free[update_mask[valid_batch]],remaining_rR[update_mask[valid_batch]]],dim=-1))[:,:r_len]
            best_result[update_mask] = temp_result
    # Return original relator where no valid cancellation was found
    no_valid = best_offset == -1
    best_result[no_valid] = relators[no_valid]
    
    # Left justify the final result to ensure consistent format
    #print("best_result",best_result)
    best_result = left_justify_states(best_result)
    
    return best_result


def contract_endpoints_of_relator_flexible_torch_vmap(relators, pattern):
    """
    Removes the first and last nonzero elements of each relator if they match the given pattern.
    PyTorch version that operates on a batch of vectors.

    Args:
        relators: Tensor of shape (batch_size, state_size) containing relators to check
        pattern: Tensor of [first,last] values to match against endpoints

    Returns:
        New relators with endpoints removed where pattern matched
    """
    # Get nonzero counts and elements for all relators
    nonzero_mask = relators != 0
    nonzero_counts = torch.sum(nonzero_mask, dim=1)
    max_rel_length = relators.shape[1]
    
    # Get first and last nonzero elements
    first_nonzero_indices = torch.argmax(nonzero_mask.long(), dim=1)
    first_elements = relators[torch.arange(len(relators),device=relators.device), first_nonzero_indices]
    
    last_nonzero_indices = relators.shape[1] - 1 - torch.argmax(torch.fliplr(nonzero_mask).long(), dim=1)
    last_elements = relators[torch.arange(len(relators),device=relators.device), last_nonzero_indices]
    #print("first_elements",first_elements.shape,first_elements)
    #print("last_elements",last_elements.shape,last_elements)

    # Create output tensor
    new_relators = relators.clone()

    # Path 1: nonzero_length <= max_rel_length-2
    #print("nonzero_counts",nonzero_counts, "max_rel_length",max_rel_length)
    short_mask = nonzero_counts <= (max_rel_length - 2)
    if torch.any(short_mask):
        # Shift everything right by 1
        shifted = relators[short_mask]
        #print("shifted",shifted)
        shifted = torch.roll(shifted, shifts=(0,1), dims=(0,1))
        #print("shifted2",shifted)
        shifted[:, 0] = -pattern[0]
        #print("shifted3",shifted)
        nonzero_mask = shifted != 0
        nonzero_counts_shifted = torch.sum(nonzero_mask, dim=1)
        batch_indices = torch.arange(len(nonzero_counts_shifted))
        shifted[batch_indices, nonzero_counts_shifted] = -pattern[1]
        #print("shifted4",shifted)
        #print("shifted",shifted)
        shifted = _iterative_simplify_vectorized_torch(shifted)
        new_relators[short_mask] = shifted
    #print("new_relators",new_relators)

    # # Path 2: nonzero_length == max_rel_length-1
    # mid_mask = nonzero_counts == (max_rel_length - 1)
    # if torch.any(mid_mask):
    #     start_match = first_elements[mid_mask] == pattern[0]
    #     end_match = last_elements[mid_mask] == pattern[1]
        
    #     # Handle start matches
    #     start_only = mid_mask.clone()
    #     start_only[mid_mask] &= start_match
    #     if torch.any(start_only):
    #         indices = torch.arange(1, max_rel_length, device=relators.device)
    #         new_relators[start_only,0:-1] = torch.index_select(relators[start_only], 1, indices)
    #         nonzero_mask = new_relators[start_only] != 0
    #         nonzero_counts = torch.sum(nonzero_mask, dim=1)
    #         batch_indices = torch.arange(start_only.sum())
    #         new_relators[start_only][batch_indices, nonzero_counts] = -pattern[1]

    #     # Handle end matches
    #     end_only = mid_mask.clone()
    #     end_only[mid_mask] &= end_match & ~start_match
    #     if torch.any(end_only):
    #         indices = torch.arange(0, max_rel_length-1, device=relators.device)
    #         shifted = torch.index_select(relators[end_only], 1, indices)
    #         shifted = torch.cat([
    #             -pattern[0].expand(shifted.shape[0], 1),
    #             shifted
    #         ], dim=1)
    #         new_relators[end_only] = shifted

    #print("nonzero_counts",nonzero_counts, "max_rel_length",max_rel_length)
    mid_mask = nonzero_counts == (max_rel_length - 1)
    if torch.any(mid_mask):
        start_match = first_elements[mid_mask] == pattern[0]
        end_match = last_elements[mid_mask] == pattern[1]
        
        # Handle both matches
        both_match = mid_mask.clone()
        both_match[mid_mask] &= start_match & end_match
        if torch.any(both_match):
            indices = torch.arange(1, max_rel_length-2, device=relators.device)
            new_relators[both_match,0:-3] = torch.index_select(relators[both_match], 1, indices)
            new_relators[both_match,-3:]=0

        # Handle start match only
        start_only = mid_mask.clone()
        start_only[mid_mask] &= start_match & ~end_match
        if torch.any(start_only):
            indices = torch.arange(1, max_rel_length-1, device=relators.device)
            new_relators[start_only,0:-2] = torch.index_select(relators[start_only], 1, indices)
            new_relators[start_only,-2] = -pattern[1]
            new_relators[start_only,-1] = 0

        # Handle end match only
        end_only = mid_mask.clone()
        end_only[mid_mask] &= ~start_match & end_match
        if torch.any(end_only):
            indices = torch.arange(0, max_rel_length-2, device=relators.device)
            shifted = torch.index_select(relators[end_only], 1, indices)
            shifted = torch.cat([
                -pattern[0].expand(shifted.shape[0], 1),
                shifted
            ], dim=1)
            new_relators[end_only,:-1] = shifted
            new_relators[end_only,-1] = 0


    # Path 3: nonzero_length == max_rel_length
    full_mask = nonzero_counts == max_rel_length
    if torch.any(full_mask):
        start_match = first_elements[full_mask] == pattern[0]
        end_match = last_elements[full_mask] == pattern[1]
        
        # Handle both matches
        both_match = full_mask.clone()
        both_match[full_mask] &= start_match & end_match
        if torch.any(both_match):
            indices = torch.arange(1, max_rel_length-1, device=relators.device)
            new_relators[both_match,0:-2] = torch.index_select(relators[both_match], 1, indices)
            new_relators[both_match,-2:] *= 0

        # Handle start match only
        start_only = full_mask.clone()
        start_only[full_mask] &= start_match & ~end_match
        if torch.any(start_only):
            indices = torch.arange(1, max_rel_length, device=relators.device)
            new_relators[start_only,0:-1] = torch.index_select(relators[start_only], 1, indices)
            new_relators[start_only,-1] = -pattern[1]


        # Handle end match only
        end_only = full_mask.clone()
        end_only[full_mask] &= ~start_match & end_match
        if torch.any(end_only):
            indices = torch.arange(0, max_rel_length-1, device=relators.device)
            shifted = torch.index_select(relators[end_only], 1, indices)
            shifted = torch.cat([
                -pattern[0].expand(shifted.shape[0], 1),
                shifted
            ], dim=1)
            new_relators[end_only] = shifted
    return new_relators

def get_random_scramble_torch(state, scramble_length):
    current_state = torch.tensor(state, device='cpu',dtype=torch.int64)
    moves=torch.zeros((scramble_length,),device='cpu',dtype=torch.long)
    for i in range(scramble_length):
        last_state = current_state.clone()
        moves_temp = list(range(12))
        while torch.equal(current_state,last_state):
            move = random.choice(moves_temp)
            current_state = finger_ix_fast_vec_torch(current_state.unsqueeze(0), move).squeeze(0)
            moves_temp.remove(move)
        moves[i] = move
    return current_state,moves

def compute_weights_from_state(state,env,weight_contraction=3,total_relator_weight=0.16,double_weight=None):
    if double_weight is None:
        double_weight = weight_contraction**1.5
    double_weight_tensor = torch.tensor(double_weight, device=state.device)
    device = state.device
    weights = torch.ones(12, device=device) * 0.05
    r0, r1 = state[:env.max_relator_length], state[env.max_relator_length:]
    r0_nonzero = r0[r0 != 0]
    r1_nonzero = r1[r1 != 0]
    weight_higher = weight_contraction
    if len(r0_nonzero) > 0:
        r0_first, r0_last = r0_nonzero[0], r0_nonzero[-1]
        
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

        # x1 cancellation for r0
        weights[5] *= torch.min(torch.where(r0_first == 2, weight_higher, 1.0) * torch.where(r0_last == -2, weight_higher, 1.0), double_weight_tensor)  # x1^-1 r0 x1
        weights[9] *= torch.min(torch.where(r0_first == -2, weight_higher, 1.0) * torch.where(r0_last == 2, weight_higher, 1.0), double_weight_tensor)  # x1 r0 x1^-1
        
        # x0 cancellation for r0
        weights[11] *= torch.min(torch.where(r0_first == 1, weight_higher, 1.0) * torch.where(r0_last == -1, weight_higher, 1.0), double_weight_tensor)  # x0^-1 r0 x0
        weights[7] *= torch.min(torch.where(r0_first == -1, weight_higher, 1.0) * torch.where(r0_last == 1, weight_higher, 1.0), double_weight_tensor)  # x0 r0 x0^-1

    if len(r1_nonzero) > 0:
        r1_first, r1_last = r1_nonzero[0], r1_nonzero[-1]
        
        # x0 cancellation for r1
        weights[4] *= torch.min(torch.where(r1_first == 1, weight_higher, 1.0) * torch.where(r1_last == -1, weight_higher, 1.0), double_weight_tensor)  # x0^-1 r1 x0
        weights[8] *= torch.min(torch.where(r1_first == -1, weight_higher, 1.0) * torch.where(r1_last == 1, weight_higher, 1.0), double_weight_tensor)  # x0 r1 x0^-1
        
        # x1 cancellation for r1
        weights[6] *= torch.min(torch.where(r1_first == 2, weight_higher, 1.0) * torch.where(r1_last == -2, weight_higher, 1.0), double_weight_tensor)  # x1^-1 r1 x1
        weights[10] *= torch.min(torch.where(r1_first == -2, weight_higher, 1.0) * torch.where(r1_last == 2, weight_higher, 1.0), double_weight_tensor)  # x1 r1 x1^-1

    # Normalize weights
    total_conj_weight = 1-total_relator_weight
    weights[4:] = total_conj_weight*weights[4:] / weights[4:].sum()
    weights[:4] = torch.ones(4, device= device) * (total_relator_weight/4)
    return weights

def compute_weights_from_state_vec(states, env, weight_contraction=3, total_relator_weight=0.16,double_weight = None):
    """Vectorized version of compute_weights_from_state that works on batches of states"""
    if double_weight is None:
        double_weight = weight_contraction**1.5
    double_weight_tensor = torch.tensor(double_weight, device=states.device)
    batch_size = states.shape[0]
    device = states.device
    weights = torch.ones((batch_size, 12), device=device,dtype=float) * 0.05
    weight_higher = weight_contraction

    # Split states into r0 and r1
    r0 = states[:, :env.max_relator_length]  
    r1 = states[:, env.max_relator_length:]

    # Get first and last nonzero elements for r0 and r1
    r0_mask = r0 != 0
    r1_mask = r1 != 0
    
    # Get indices of first and last nonzero elements
    r0_first_idx = torch.argmax(r0_mask.float(), dim=1)
    r1_first_idx = torch.argmax(r1_mask.float(), dim=1)
    r0_last_idx = r0.shape[1] - 1 - torch.argmax(r0_mask.float().flip(1), dim=1)
    r1_last_idx = r1.shape[1] - 1 - torch.argmax(r1_mask.float().flip(1), dim=1)

    # Get values of first and last nonzero elements
    r0_has_nonzero = r0_mask.any(dim=1)
    r1_has_nonzero = r1_mask.any(dim=1)
    
    r0_first = torch.zeros(batch_size, device=device,dtype=torch.int64)
    r0_last = torch.zeros(batch_size, device=device,dtype=torch.int64)
    r1_first = torch.zeros(batch_size, device=device,dtype=torch.int64)
    r1_last = torch.zeros(batch_size, device=device,dtype=torch.int64)
    
    r0_first[r0_has_nonzero] = r0[r0_has_nonzero, r0_first_idx[r0_has_nonzero]]
    r0_last[r0_has_nonzero] = r0[r0_has_nonzero, r0_last_idx[r0_has_nonzero]]
    r1_first[r1_has_nonzero] = r1[r1_has_nonzero, r1_first_idx[r1_has_nonzero]]
    r1_last[r1_has_nonzero] = r1[r1_has_nonzero, r1_last_idx[r1_has_nonzero]]

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

    # Handle r0 weights
    # x1 cancellation
    weights[:, 5] *= torch.min(torch.where(r0_first == 2, weight_higher, 1.0) * torch.where(r0_last == -2, weight_higher, 1.0), double_weight_tensor)  # x1^-1 r0 x1
    weights[:, 9] *= torch.min(torch.where(r0_first == -2, weight_higher, 1.0) * torch.where(r0_last == 2, weight_higher, 1.0), double_weight_tensor)  # x1 r0 x1^-1
    
    # x0 cancellation
    weights[:, 11] *= torch.min(torch.where(r0_first == 1, weight_higher, 1.0) * torch.where(r0_last == -1, weight_higher, 1.0), double_weight_tensor)  # x0^-1 r0 x0
    weights[:, 7] *= torch.min(torch.where(r0_first == -1, weight_higher, 1.0) * torch.where(r0_last == 1, weight_higher, 1.0), double_weight_tensor)  # x0 r0 x0^-1

    # Handle r1 weights
    # x0 cancellation
    weights[:, 4] *= torch.min(torch.where(r1_first == 1, weight_higher, 1.0) * torch.where(r1_last == -1, weight_higher, 1.0), double_weight_tensor)  # x0^-1 r1 x0
    weights[:, 8] *= torch.min(torch.where(r1_first == -1, weight_higher, 1.0) * torch.where(r1_last == 1, weight_higher, 1.0), double_weight_tensor)  # x0 r1 x0^-1
    
    # x1 cancellation
    weights[:, 6] *= torch.min(torch.where(r1_first == 2, weight_higher, 1.0) * torch.where(r1_last == -2, weight_higher, 1.0), double_weight_tensor)  # x1^-1 r1 x1
    weights[:, 10] *= torch.min(torch.where(r1_first == -2, weight_higher, 1.0) * torch.where(r1_last == 2, weight_higher, 1.0), double_weight_tensor)  # x1 r1 x1^-1

    # Normalize weights
    total_conj_weight = 1 - total_relator_weight
    weights_conj_sum = weights[:, 4:].sum(dim=1, keepdim=True)
    weights[:, 4:] = total_conj_weight * weights[:, 4:] / weights_conj_sum
    weights[:, :4] = total_relator_weight / 4

    return weights

def get_probabilities_from_model_output(scores,states_in,states_out,env,weight_contraction=3,total_relator_weight=0.03,double_weight=20,fwd_model = compute_weights_from_state_vec):
    batch_dim = states_in.shape[0]
    state_dim = env.state_dim
    num_moves = env.num_moves
    moves = torch.arange(num_moves)
    inverse_moves = torch.tensor(env.inverse_moves)
    #with torch.no_grad():
    #    predictions = model(states, t)#.cpu().detach().numpy()
    ##    model.eval()
    if fwd_model is compute_weights_from_state_vec:
        weights_from_gx_to_x = compute_weights_from_state_vec(states_out.reshape(batch_dim*num_moves,state_dim),
                                                          env,weight_contraction=weight_contraction,total_relator_weight=total_relator_weight,
                                                          double_weight=double_weight).reshape(batch_dim,num_moves,num_moves)[:,moves,inverse_moves]
    else:
        weights_from_gx_to_x = fwd_model(states_out.reshape(batch_dim*num_moves,state_dim)).reshape(batch_dim,num_moves,num_moves)[:,moves,inverse_moves]
    #print(weights_from_gx_to_x.shape)
    # these weights are q in qtilde(y,x) = score * q(x,y) - probaility of moving from y to x, from gx to x.
    #they are NOT normalised to be probabilities,
    #compute_one = compute_weights_from_state(states_out[0],env,weight_contraction=3,total_relator_weight=0.16)
    #print(compute_one)
    #print(compute_one.sum(axis=-1))
    #print(weights_from_gx_to_x.sum(axis=-1))
    #assert torch.allclose(weights_from_gx_to_x.sum(axis=-1),torch.ones(batch_dim,dtype=weights_from_gx_to_x.dtype,device=weights_from_gx_to_x.device))
    #print("weights_from_gx_to_x: ",weights_from_gx_to_x.shape)
    did_not_change_mask = (states_out == states_in.unsqueeze(1).repeat(1,num_moves,1)).all(dim=-1)
    weights_from_gx_to_x[did_not_change_mask] = 0
    #print("weights_from_gx_to_x: ",weights_from_gx_to_x.shape)
    #weights_from_gx_to_x = weights_from_gx_to_x#/weights_from_gx_to_x.sum(axis=-1,keepdims=True)
    
    new_transition_kernel = weights_from_gx_to_x * scores#[did_not_change_mask]
    probabilities = new_transition_kernel/new_transition_kernel.sum(axis=-1,keepdims=True)


    #probabilities = predictions/predictions.sum(axis=-1, keepdims=True)
    return probabilities#, states_out

def mask_impossible_moves(states_in,env,return_states_out=False):
    """
    returns a mask of shape (states_in.shape[0],env.num_moves)
    where the i-th entry is True if the i-th move is possible for the i-th state, and False otherwise

    if return_states_out, it returns states_out, which is the result of applying all moves to all states
    """
    states_out = apply_all_moves_to_all_states_torch_jit(states_in.reshape(-1,env.state_dim)).reshape(states_in.shape[0],env.num_moves,env.state_dim)
    if return_states_out:
        return ~(states_out == states_in.unsqueeze(1).repeat(1,env.num_moves,1)).all(dim=-1), states_out
    else:
        return ~(states_out == states_in.unsqueeze(1).repeat(1,env.num_moves,1)).all(dim=-1)


def block_inverse_moves_func(last_5_moves,weights,inverse_moves):
    last_5_inverses = inverse_moves[last_5_moves]
    # Find latest occurrence of [5,7,9,11] and [4,6,8,10] in last 5 moves
    # and zero out their weights if no relator moves (0,1,2,3) occur after them
    r0_conjugates = torch.tensor([5,7,9,11], device=weights.device)
    r1_conjugates = torch.tensor([4,6,8,10], device=weights.device)
    relator_moves = torch.tensor([0,1,2,3], device=weights.device)
    
    # Get indices where last_5_inverses matches r0_conjugates
    r0_matches = (last_5_inverses.unsqueeze(-1) == r0_conjugates).nonzero()
    if len(r0_matches) > 0:
        # Get most recent match
        latest_r0_idx = r0_matches[:,0].max()
        latest_r0_move = r0_conjugates[r0_matches[r0_matches[:,0] == latest_r0_idx, 1][0]]
        
        # Check if any relator moves occur after the latest r0 conjugate
        moves_after = last_5_inverses[latest_r0_idx+1:]
        if not any((moves_after.unsqueeze(-1) == relator_moves).any(-1)):
            weights[latest_r0_move] *= 1e-3
            
    # Get indices where last_5_inverses matches r1_conjugates
    r1_matches = (last_5_inverses.unsqueeze(-1) == r1_conjugates).nonzero()
    if len(r1_matches) > 0:
        # Get most recent match
        latest_r1_idx = r1_matches[:,0].max()
        latest_r1_move = r1_conjugates[r1_matches[r1_matches[:,0] == latest_r1_idx, 1][0]]
        
        # Check if any relator moves occur after the latest r1 conjugate
        moves_after = last_5_inverses[latest_r1_idx+1:]
        if not any((moves_after.unsqueeze(-1) == relator_moves).any(-1)):
            weights[latest_r1_move] *= 1e-3
    return weights


def scrambler_torch(env, scramble_length, batch_size=1, device='cpu',return_apply_all=False,weight_contraction=3,total_relator_weight=0.16, start_state=None,block_inverse_moves=False,double_weight=None):
    """
    Generates batches of scrambled states and their corresponding moves.
    Returns tensors of shape:
    - states: (batch_size, scramble_length, state_size)
    - moves: (batch_size, scramble_length)
    """
    inverse_moves = torch.tensor(env.inverse_moves)
    if start_state is not None:
        start_state = torch.tensor(start_state, device=device,dtype=torch.int64)
    else:
        env.reset()
        start_state = torch.tensor(env.state, device=device,dtype=torch.int64)
    while True:
        # Initialize tensors to store batch
        states = torch.zeros((batch_size, scramble_length+1, env.state.size), device=device,dtype=start_state.dtype)
        moves = torch.zeros((batch_size, scramble_length), dtype=torch.long, device=device)
        states[:,0] = start_state.clone().detach()
        
        # Generate scrambles for each item in batch
        for b in range(batch_size):
            #state = copy.deepcopy(start_state)
            state = start_state.clone().detach()
            
            # Generate scramble of specified length
            for s in range(scramble_length):
                #  0. r_1 --> r_1 r_0
                # 1. r_0 --> r_0 r_1^{-1}
                # 2. r_1 --> r_1 r_0^{-1}
                # 3. r_0 --> r_0 r_1
                # 4: r_1 --> x_0^{-1} r_1 x_0
                #        5: r_0 ---> x_1^{-1} r_0 x_1
                # 6: r_1 --> x_1^{-1} r_1 x_1
                #        7: r_0 ---> x_0 r_0 x_0^{-1}
                # 8: r_1 --> x_0 r_1 x_0^{-1}
                #       9: r_0 --> x_1 r_0 x_1^{-1}
                # 10: r_1 --> x_1 r_1 x_1^{-1}
                #        11: r_0 --> x_0^{-1} r_0 x_0
                # Choose random move
                moves_temp = torch.arange(env.num_moves, device=device)
                weights = compute_weights_from_state(state,env,weight_contraction,total_relator_weight,double_weight=double_weight)
                if s>5 and block_inverse_moves:
                    weights = block_inverse_moves_func(moves[b,s-5:s],weights,inverse_moves)
                # Choose move based on weights and try moves until state changes
                while torch.equal(state, states[b,s]):
                    # Renormalize weights for remaining moves
                    
                    # Choose move from remaining moves
                    move = torch.multinomial(weights, num_samples=1).item()#values do not need at add to 1, so no need to normalise weights
                    state = finger_ix_fast_vec_torch(state.unsqueeze(0), move).squeeze(0)
                    weights[move]=0
                    
                    # Break if no moves left to try
                    if moves_temp.sum()<= 1e-6:
                        break
                        
                # Store results
                states[b,s+1] = state 
                moves[b,s] = move
        
        if return_apply_all:
            all_next_states = apply_all_moves_to_all_states_torch(states.reshape(batch_size*(scramble_length+1),env.state.size)).reshape(batch_size,(scramble_length+1),env.num_moves,env.state.size)
        else:
            all_next_states = None
        yield states, moves, all_next_states

import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import math

def fwd_diffusion_probs_func_from_model(fwd_model,env,return_states_out=False,device=None):
    # if device is None:
    #     if hasattr(fwd_model, 'device'):
    #         device = fwd_model.device
    #     else:
    #         device = 'cpu'
    def func_for_fwd(states):
        learned_policy = fwd_model(states)
        if return_states_out:
            good_moves_mask,states_out = mask_impossible_moves(states,env,return_states_out=return_states_out)
            learned_policy = learned_policy*good_moves_mask
            learned_policy = learned_policy/learned_policy.sum(dim=-1,keepdims=True)
            return learned_policy,states_out
        else:
            good_moves_mask = mask_impossible_moves(states,env,return_states_out=return_states_out)
            learned_policy = learned_policy*good_moves_mask
            learned_policy = learned_policy/learned_policy.sum(dim=-1,keepdims=True)
            return learned_policy
    return func_for_fwd

def reverse_diffusion_probs_from_fwd_model_and_scores(fwd_model,env,return_states_out=False,device=None):
    if device is None:
        if hasattr(fwd_model, 'parameters'):
            device = next(fwd_model.parameters()).device
        else:
            device = 'cpu'
    moves = torch.arange(env.num_moves,device=device)
    inverse_moves = torch.tensor(env.inverse_moves,device=device)
    def func_for_reverse(states,scores, return_states_out=return_states_out):
        scores = scores.to(device)
        states = states.to(device)
        good_moves_mask,states_out = mask_impossible_moves(states,env,return_states_out=True)
        weights_from_gx_to_x = fwd_model(states_out.reshape(-1,env.state_dim)).reshape(-1,env.num_moves,env.num_moves)[:,moves,inverse_moves]
        learned_fwd_policy = weights_from_gx_to_x*good_moves_mask*scores
        learned_fwd_policy = learned_fwd_policy/learned_fwd_policy.sum(dim=-1,keepdims=True)
        if return_states_out:
            return learned_fwd_policy,states_out
        else:
            return learned_fwd_policy
    return func_for_reverse

def reverse_diffusion_probs_from_fwd_model_and_scores_time(fwd_model,env,return_states_out=False,device=None):
    """
    fwd_model is a function that takes in a state and a time, and returns a probability distribution over moves.
    note that the fwd_model should not return out_states. We don't need the out states, as fwd_model is already itself processsing (gx) for all x inputted into the reverse diffusive process.
    """
    if device is None:
        if hasattr(fwd_model, 'parameters'):
            device = next(fwd_model.parameters()).device
        else:
            device = 'cpu'
    moves = torch.arange(env.num_moves,device=device)
    inverse_moves = torch.tensor(env.inverse_moves,device=device)
    
    def func_for_reverse(states,times,scores, return_states_out=return_states_out):
        #times are indexed from 0 to max_depth-1, as fwd_model adds 1.
        scores = scores.to(device)
        states = states.to(device)
        times = times.to(device)
        #print(states.shape)
        good_moves_mask,states_out = mask_impossible_moves(states,env,return_states_out=True)
        with torch.no_grad():# is this the right time coordinate?
            weights_from_gx_to_x = fwd_model(states_out.reshape(-1,env.state_dim),times.repeat_interleave(env.num_moves)).reshape(-1,env.num_moves,env.num_moves)[:,moves,inverse_moves]
        learned_fwd_policy = weights_from_gx_to_x*good_moves_mask*scores
        learned_fwd_policy = learned_fwd_policy/learned_fwd_policy.sum(dim=-1,keepdims=True)
        if return_states_out:
            return learned_fwd_policy,states_out
        else:
            return learned_fwd_policy
    return func_for_reverse

#def scrambler_torch_batch(env, scramble_length, batch_size=1, device='cpu', return_apply_all=False, weight_contraction=3, total_relator_weight=0.16, start_state=None, block_inverse_moves=False, double_weight=None):
def scrambler_torch_batch(env, scramble_length, batch_size=1, device='cpu', return_apply_all=False,start_state=None, model=None,compute_all_states=True):
    """
    Generates batches of scrambled states and their corresponding moves, computing weights and applying moves in batches.
    Returns tensors of shape:
    - states: (batch_size, scramble_length, state_size)
    - moves: (batch_size, scramble_length)
    """
    #inverse_moves = torch.tensor(env.inverse_moves)
    if start_state is not None:
        start_state = torch.tensor(start_state, device=device, dtype=torch.int64)
    else:
        env.reset()
        start_state = torch.tensor(env.state, device=device, dtype=torch.int64)

    if False:
        fwd_diffusion_function = lambda x: compute_weights_from_state_vec(x, env, weight_contraction, total_relator_weight, double_weight=double_weight)
    else:
        fwd_diffusion_function = fwd_diffusion_probs_func_from_model(model,env,return_states_out=compute_all_states)

    while True:
        yield generate_trajectory_torch_batch(batch_size,scramble_length, env,device,start_state,return_apply_all,fwd_diffusion_function, compute_all_states=compute_all_states)

# def generate_trajectory_torch_batch(batch_size,scramble_length, env,device,start_state, return_apply_all,fwd_diffusion_function, compute_all_states=True):
#     """
#     Generates a single batch of scrambled states and their corresponding moves.

#     Args:
#         batch_size (int): Number of trajectories to generate in parallel
#         scramble_length (int): Length of each scramble sequence
#         env: The cube environment
#         device (str): Device to place tensors on ('cpu' or 'cuda')
#         start_state (torch.Tensor): Initial state to start scrambling from
#         return_apply_all (bool): Whether to return all possible next states
#         fwd_diffusion_function (callable): Function that computes move probabilities for each state (including masking bad moves)

#     Returns:
#         tuple:
#             - states (torch.Tensor): States for each step of each trajectory (batch_size, scramble_length+1, state_size)
#             - moves (torch.Tensor): Moves taken at each step (batch_size, scramble_length)
#             - all_next_states (torch.Tensor or None): If return_apply_all is True, contains all possible next states
#     """
#     # Initialize tensors to store batch
#     states = torch.zeros((batch_size, scramble_length+1, env.state.size), device=device, dtype=start_state.dtype)
#     moves = torch.zeros((batch_size, scramble_length), dtype=torch.long, device=device)
#     states[:,0] = start_state.clone().detach()
    
#     # Generate scrambles for all items in batch simultaneously
#     current_states = start_state.clone().detach().expand(batch_size, -1)
    
#     for s in range(scramble_length):
#         # Compute weights for all states in batch using vectorized computation
#         #weights = compute_weights_from_state_vec(current_states, env, weight_contraction, total_relator_weight, double_weight=double_weight)
#         if compute_all_states:
#             weights_BM, all_next_states_BMS = fwd_diffusion_function(current_states)
#             batch_moves_B = torch.multinomial(weights_BM, num_samples=1).squeeze(-1)
#             current_states = all_next_states_BMS[torch.arange(batch_size),batch_moves_B,:]
#         else:
#             weights_BM = fwd_diffusion_function(current_states)
#             #all_next_states_BMS = apply_all_moves_to_all_states(current_states.reshape(-1, env.state_dim),STICKER_SOURCE_IX,STICKER_TARGET_IX).view(-1, env.num_moves, env.state_dim)
#             batch_moves_B = torch.multinomial(weights_BM, num_samples=1).squeeze(-1)
#             current_states = apply_list_of_moves_to_states(current_states, batch_moves_B, env.sticker_source_ix_torch, env.sticker_target_ix_torch)
#         # Apply moves to all states at once
#         #next_states_BS = finger_ix_fast_vec_torch_list_of_moves(current_states, batch_moves_B)
#         # Store results
#         raise Exception("Not fixed - the all_next_states stuff is wrong")
        
#         states[:,s+1] = current_states
#         moves[:,s] = batch_moves_B
    
#     if return_apply_all:
#         all_next_states = all_next_states_BMS
#     else:
#         all_next_states = None
#     return states, moves, all_next_states




def generate_trajectory_torch_batch_AC_options(batch_size,scramble_length, env,device,start_state, return_apply_all,fwd_diffusion_function, compute_all_states=True, apply_time = False, start_with_both = False, start_states_in_ball = 0):
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
    swapped_start_state = torch.cat((start_state[env.max_relator_length:],start_state[:env.max_relator_length]))
    if start_with_both:
        states[:,0] = torch.where(
            torch.rand(batch_size, device=device)[:, None] < 0.5,
            start_state.clone(),
            swapped_start_state.clone()
        )
    else:
        states[:,0] = start_state.clone()
    if start_states_in_ball != 0:
        # For each state, randomly choose how many moves to apply (0 to start_states_in_ball)
        num_moves_per_state = torch.randint(0, start_states_in_ball + 1, (batch_size,), device=device)
        current_states = states[:,0].clone()
        
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
                    current_states[mask] = finger_ix_fast_vec_torch_list_of_moves(
                        current_states[mask], 
                        moves_to_apply[mask,i]
                    )
        states[:,0] = current_states.clone()


    
    # Generate scrambles for all items in batch simultaneously
    current_states = states[:,0].clone().detach()
    
    for s in range(scramble_length):
        # Compute weights for all states in batch using vectorized computation
        #weights = compute_weights_from_state_vec(current_states, env, weight_contraction, total_relator_weight, double_weight=double_weight)
        if apply_time:
            time_input = torch.tensor([(s)/scramble_length], device=device).expand(batch_size)
        if return_apply_all and not compute_all_states:
            raise Exception("Not fixed - the all_next_states stuff is wrong")
            all_next_states[:,s,:,:] = apply_all_moves_to_all_states(current_states,env.sticker_source_ix_torch,env.sticker_target_ix_torch)
        if compute_all_states:
            if apply_time:
                weights_BM, all_next_states_BMS = fwd_diffusion_function(current_states,time_input)# all next states of current_states, which is [:,s]. so should go in [s]
            else:
                weights_BM, all_next_states_BMS = fwd_diffusion_function(current_states)
            weights_BM = weights_BM.detach()
            all_next_states_BMS = all_next_states_BMS.detach()
            batch_moves_B = torch.multinomial(weights_BM, num_samples=1).squeeze(-1)
            current_states = all_next_states_BMS[torch.arange(batch_size),batch_moves_B,:].detach()
            
            if return_apply_all:
                all_next_states[:,s,:,:] = all_next_states_BMS
        else:
            raise Exception("Not fixed - the all_next_states stuff is wrong")
            if apply_time:
                weights_BM = fwd_diffusion_function(current_states,time_input).detach()
            else:
                weights_BM = fwd_diffusion_function(current_states).detach()
            batch_moves_B = torch.multinomial(weights_BM, num_samples=1).squeeze(-1)
            current_states = finger_ix_fast_vec_torch_list_of_moves(current_states, batch_moves_B).detach()

        states[:,s+1] = current_states
        moves[:,s] = batch_moves_B

        
    if return_apply_all and not compute_all_states:# at the end
        raise Exception("Not fixed - the all_next_states stuff is wrong")
        all_next_states[:,scramble_length,:,:] = apply_all_moves_to_all_states(current_states,env.sticker_source_ix_torch,env.sticker_target_ix_torch)
        all_next_states = all_next_states
    return states.detach(), moves.detach(), all_next_states.detach() if all_next_states is not None else None

# def generate_trajectory_torch_batch_RL(batch_size,scramble_length, env,device,start_state,fwd_diffusion_function):
#     """
#     Generates batches of scrambled states and their corresponding moves, computing weights and applying moves in batches.
#     Returns tensors of shape:
#     - states: (batch_size, scramble_length+1, state_size)
#     - moves: (batch_size, scramble_length)
#     - weights_each_step: (batch_size, scramble_length, num_moves)
#     """
#     # Initialize tensors to store batch
#     states = torch.zeros((batch_size, scramble_length+1, env.state.size), device=device, dtype=start_state.dtype)
#     moves = torch.zeros((batch_size, scramble_length), dtype=torch.long, device=device)
#     states[:,0] = start_state.clone().detach()
#     weights_each_step = torch.zeros((batch_size, scramble_length, env.num_moves), device=device, dtype=torch.float)
    
#     # Generate scrambles for all items in batch simultaneously
#     current_states = start_state.clone().detach().expand(batch_size, -1)
    
#     for s in range(scramble_length):
#         # Compute weights for all states in batch using vectorized computation
#         #weights = compute_weights_from_state_vec(current_states, env, weight_contraction, total_relator_weight, double_weight=double_weight)
#         time_step = torch.tensor(s/scramble_length, device=device, dtype=torch.float).expand(batch_size)
#         weights_BM = fwd_diffusion_function(current_states,time_step)
#         batch_moves_B = torch.multinomial(weights_BM, num_samples=1).squeeze(-1)
        
#         # Apply moves to all states at once
#         next_states_BS = finger_ix_fast_vec_torch_list_of_moves(current_states, batch_moves_B)
#         # Store results
#         current_states = next_states_BS
#         states[:,s+1] = current_states.clone().detach()
#         moves[:,s] = batch_moves_B.clone().detach()
#         weights_each_step[:,s] = weights_BM.clone().detach()
    
#     return states, moves, weights_each_step



def apply_scramble_torch(state, scramble):
    for move in scramble:
        state = finger_ix_fast_vec_torch(state.unsqueeze(0), move).squeeze(0)
    return state

class ScrambleGeneratorTORCH(IterableDataset):
    def __init__(
            self,
            max_depth=1,#TrainConfig.max_depth,
            total_samples=1,#TrainConfig.num_steps * TrainConfig.batch_size_per_depth
            seed=None,
            env=AC_presentation(max_relator_length=25),
            batch_size=1,
            return_apply_all=False,
            #weight_contraction=3,
            #block_inverse_moves=False,
            ##total_relator_weight=0.16,
            #double_weight=None
            model=lambda x: torch.ones((x.shape[0],12))/12,
            device='cpu'
        ):
        super(ScrambleGeneratorTORCH, self).__init__()
        self.max_depth = max_depth
        self.total_samples = total_samples
        self.seed = seed
        self.env = env
        self.batch_size = batch_size
        self.return_apply_all = return_apply_all
        #self.generator = scrambler_torch(self.env,self.max_depth,self.batch_size,return_apply_all=return_apply_all,weight_contraction=weight_contraction,total_relator_weight=total_relator_weight,block_inverse_moves=block_inverse_moves,double_weight=double_weight)
        #self.generator = scrambler_torch_batch(self.env,self.max_depth,self.batch_size,return_apply_all=return_apply_all,
        #                                       weight_contraction=weight_contraction,total_relator_weight=total_relator_weight,
        #                                       block_inverse_moves=block_inverse_moves,double_weight=double_weight)
        self.generator = scrambler_torch_batch(self.env,self.max_depth,self.batch_size,return_apply_all=return_apply_all,model=model,device=device)
    

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            iter_start = 0
            iter_end = self.total_samples
            worker_seed = self.seed
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.total_samples / float(self.batch_size*worker_info.num_workers)))
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
            #X = np.zeros((self.max_depth, self.env.state_dim), dtype=int)
            #y = np.zeros((self.max_depth,), dtype=int)
            #for j in range(self.max_depth):
            #    state, last_move = next(self.generator)
            #    y[j] = last_move
            ##    X[j, :] = state
            #yield X, y
            states, moves, all_next_states = next(self.generator)
            yield states, moves, all_next_states

apply_all_moves_to_all_states_torch_jit = torch.jit.script(apply_all_moves_to_all_states_torch)
