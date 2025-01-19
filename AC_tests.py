"""
Extensive tests for various implementations of the Andrews-Curtis group environment.

These tests are non-deterministic as we test on random (large) batches of states.
"""

from envsAndScramble import *
import importlib
import time
import numpy as np
import torch
from AC_env import *

# Test state from file_context_0
test_state = np.array([-2, 1, 2, 2, 2, -1, -2, -2, 1, 2, -1, 2, 2, 1, -2, -2, -2, -1, 2, 0, 0, 0, 0, 0,
                       0, -1, 2, 2, 1, -2, -2, -2, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print("\nTesting all moves on specific state:")
print("Initial state:", test_state)

# Convert to torch tensor for batched operations
test_state_torch = torch.from_numpy(test_state).unsqueeze(0)

ac = AC_presentation(max_relator_length=25)

# Test each move with different methods
for move in range(12):
    print(f"\nTesting move {move}:")
    
    # Test with finger_ix_fast_vec_torch
    torch_result = finger_ix_fast_vec_torch(test_state_torch, move)
    print(f"finger_ix_fast_vec_torch result: {torch_result[0].numpy()}")
    
    # Test with OLD_finger_ix_fast_vec_torch
    #old_torch_result = OLD_finger_ix_fast_vec_torch(test_state_torch, move)
    #print(f"OLD_finger_ix_fast_vec_torch result: {old_torch_result[0].numpy()}")
    
    # Test with AC_presentation's do_move_to_state_flexible
    ac.state = test_state.copy()
    ac.do_move_to_state_flexible(move)
    print(f"do_move_to_state_flexible result: {ac.state}")
    
    # Compare results
    results_match = np.array_equal(torch_result[0].numpy(), ac.state)
    print(f"All results match: {results_match}")
    if not results_match:
        print("WARNING: Results don't match!")
        exit()


# ac = AC_presentation(max_relator_length=25)
# print("Testing admissibility of moves by checking if inverse moves undo the last move of a scramble")
# this section tests finger_ix, which does not necessarily ensure moves are invertible!
# # Track failures for each move type
# failures = {i: 0 for i in range(12)}

# # Test 100 random scrambles
# for test_num in range(1000):
#     ac.reset()
#     scramble = []
#     prev_state = ac.state.copy()
#     curr_state = ac.state.copy()
#     old_prev_state = ac.state.copy()
    
#     # Keep scrambling until state stops changing
#     for i in range(1000):
#         old_prev_state = prev_state.copy()
#         prev_state = curr_state.copy()
#         #possible_moves = ac.check_conditions_return_possibilities()
#         #if not possible_moves:
#         #    break
            
#         move = np.random.choice(ac.moves)
#         ac.finger_ix(move)
#         curr_state = ac.state.copy()
#         scramble.append(move)
        
#         if np.array_equal(prev_state, curr_state):
#             scramble.pop() # Remove move that did nothing
#             #print("stopped at ",i)
#             break
            
#     if len(scramble) == 0:#or scramble[-1] !=5:
#         continue
        
#     # Get last effective move and try its inverse
#     last_move = scramble[-1]
#     inverse_move = ac.inverse_moves[last_move]
    
#     # Try the inverse move
#     prev_state = ac.state.copy()

#     admissible = ac.check_if_move_is_admissible(inverse_move)
#     if not admissible:
#         print(ac.state)
#         print(last_move,inverse_move)
#         #ac.finger_ix(inverse_move)
#         #if np.array_equal(prev_state, ac.state):
#         # Move was admissible but did nothing - count as failure
#         failures[inverse_move] += 1
#     if inverse_move in range(12):
#         ac.state = np.copy(prev_state)
#         ac.do_move_to_state_flexible(inverse_move)
#         if np.array_equal(ac.state, prev_state) or not np.array_equal(ac.state, old_prev_state):
#             print(last_move,inverse_move)
#             print("old",old_prev_state)
#             print("prev",prev_state)
#             print("new-",ac.state," should be same as old")
#             exit()
#         ac.do_move_to_state_flexible(last_move)
#         if not np.array_equal(ac.state, prev_state):
#             print("new- should be same as prev",ac.state)
#             exit()

# print(f"\nFailure counts for each move type for max_relator_length={ac.max_relator_length} and n scrambles={test_num}:")
# for move in range(12):
#     print(f"Move {move}: {failures[move]} failures")


#test pytorch version of do_move_to_state_flexible:
print("\nTesting PyTorch single vector version of do_move_to_state_flexible:")

# Create test instance
ac = AC_presentation(max_relator_length=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test 100 random scrambled states
n_states_scrambled = 100
n_random_moves = 50
for test_idx in range(n_states_scrambled):
    ac.reset()
    
    # Generate random scrambled state
    num_moves = np.random.randint(5, 15)
    for _ in range(num_moves):
        move = np.random.choice(ac.moves)
        ac.finger_ix(move)
    
    # Store scrambled state
    scrambled_state = ac.state.copy()
    

    # Test 30 random moves on this scrambled state
    for _ in range(n_random_moves):
        # Start from scrambled state
        ac.state = scrambled_state.copy()
        state = torch.from_numpy(ac.state).to(device)
        
        # Get numpy result
        np_state = ac.state.copy()
        #print("move:",move)
        ac.do_move_to_state_flexible(move)
        np_result = ac.state.copy()

        # Get numpy result for inverse move
        ac.do_move_to_state_flexible(ac.inverse_moves[move])
        np_result_inverse = ac.state.copy()
        
        # Get torch result
        state_multiple = torch.from_numpy(scrambled_state).to(device).unsqueeze(0).repeat(3,1)
        torch_results = finger_ix_fast_vec_torch(state_multiple, move)
        torch_result = torch_results[0]
        torch_result_inverse = finger_ix_fast_vec_torch(torch_results, ac.inverse_moves[move])[0]
        
        # Compare results
        if not (np.array_equal(torch_result.cpu().numpy(), np_result) and 
                np.array_equal(torch_result_inverse.cpu().numpy(), np_result_inverse)):
            ac.state = scrambled_state.copy()
            print("Admissible move?:",ac.check_if_move_is_admissible(move))
            ac.state = np_state.copy()
            print("Admissible inverse move?:",ac.check_if_move_is_admissible(ac.inverse_moves[move]))
            print(f"Mismatch in test {test_idx} for move {move}:")
            print(f"Input state: {scrambled_state}")
            print(f"Numpy move result: {np_result}")
            print(f"Numpy inverse move result: {np_result_inverse}")
            print(f"Torch move result: {torch_result.cpu().numpy()}")
            print(f"Torch inverse move result: {torch_result_inverse.cpu().numpy()}")
            raise ValueError("Results don't match!")

    
    # # Test 30 random moves on this scrambled state
    # for _ in range(30):
    #     # Start from scrambled state
    #     ac.state = scrambled_state.copy()
    #     state = torch.from_numpy(ac.state).to(device)
    #     move = np.random.randint(0, 12)
        
    #     # Get numpy result
    #     np_state = ac.state.copy()
    #     print("move:",move)
    #     ac.do_move_to_state_flexible(move)
    #     np_result = ac.state
    #     ac.state = np_state  # Reset state
        
    #     # Get torch result
    #     state_multiple = state.unsqueeze(0).repeat(3,1)
    #     torch_result = finger_ix_fast_vec_torch(state_multiple, move)[0]
        
    #     # Compare results
    #     if not np.array_equal(torch_result.cpu().numpy(), np_result):
    #         print(f"Mismatch in test {test_idx} for move {move}:")
    #         print(f"Input state: {state}")
    #         print(f"Numpy result: {np_result}")
    #         print(f"Torch result: {torch_result}")
    #         raise ValueError("Results don't match!")

print(f"All single vector tests passed! ({n_states_scrambled} scrambled states x {n_random_moves} moves each)")


print("\nTesting batched moves vs sequential moves...")

# Test parameters
num_tests = 100
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for test_idx in range(num_tests):
    # Create batch of different scrambled states
    batch_states = []
    np_states = []
    for i in range(batch_size):
        # Scramble with random number of moves (1-10)
        ac.reset()
        num_scramble = np.random.randint(1, 11)
        for _ in range(num_scramble):
            move = np.random.randint(0, 12)
            ac.do_move_to_state_flexible(move)
        batch_states.append(ac.state.copy())
        np_states.append(ac.state.copy())
    
    # Convert to torch tensor
    batch_states = torch.tensor(np.stack(batch_states), device=device)
    
    # Pick random move to test
    move = np.random.randint(0, 12)
    
    # Get batched torch result
    torch_results = finger_ix_fast_vec_torch(batch_states, move)
    
    # Get numpy results by looping
    np_results = []
    for state in np_states:
        ac.state = state
        ac.do_move_to_state_flexible(move)
        np_results.append(ac.state.copy())
    np_results = np.stack(np_results)
    
    # Compare results
    if not np.allclose(torch_results.cpu().numpy(), np_results):
        print(f"Mismatch in batch test {test_idx} for move {move}:")
        mismatch_idx = np.where(~np.all(np.isclose(torch_results.cpu().numpy(), np_results), axis=1))[0]
        print(f"First mismatch at batch index {mismatch_idx[0]}:")
        print(f"Input state: {batch_states[mismatch_idx[0]].cpu().numpy()}")
        print(f"Numpy result: {np_results[mismatch_idx[0]]}")
        print(f"Torch result: {torch_results[mismatch_idx[0]].cpu().numpy()}")
        raise ValueError("Batch results don't match!")

print(f"All batch tests passed! ({num_tests} tests with batch size {batch_size})")


# Speed comparison test
print("\nSpeed comparison test:")

num_trials = 1000
batch_size = 256
max_relator_length = 25
ac = AC_presentation(max_relator_length=max_relator_length)
# Generate random states and moves
states = torch.randint(-2, 3, (batch_size, max_relator_length*2), device=device)
states = left_justify_states(states)
states = simplify_state_vec_torch(states)
# Drop states where either half is empty
r0, r1 = states[:, :max_relator_length], states[:, max_relator_length:]
r0_nonzero = torch.any(r0 != 0, dim=1)
r1_nonzero = torch.any(r1 != 0, dim=1)
valid_mask = r0_nonzero & r1_nonzero
states = states[valid_mask].repeat(2,1)
states = states[:batch_size]
if len(states) < batch_size:
    print("Warning: not enough valid states, reducing batch size to", len(states))
    batch_size = len(states)

moves = np.random.randint(0, 12, num_trials)

finger_ix_fast_vec_torch = torch.jit.script(finger_ix_fast_vec_torch,example_inputs=(states,10))
OLD_finger_ix_fast_vec_torch = torch.jit.script(OLD_finger_ix_fast_vec_torch,example_inputs=(states,10))
# Time torch batched version
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()
for move in moves:
    _ = finger_ix_fast_vec_torch(states, move)
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch_time = time.time() - start

# Time torch OLD batched version
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()
for move in moves:
    _ = OLD_finger_ix_fast_vec_torch(states, move)
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch_old_time = time.time() - start

print(f"Torch OLD batched time: {torch_old_time:.3f}s")
print(f"Speedup vs OLD: {torch_old_time/torch_time:.1f}x")


# Time numpy loop version
start = time.time()
for move in moves:
    for state in states.cpu().numpy():
        ac.state = state
        ac.do_move_to_state_flexible(move)
numpy_time = time.time() - start

print(f"Torch batched time: {torch_time:.3f}s")
print(f"Numpy loop time: {numpy_time:.3f}s") 
print(f"Speedup: {numpy_time/torch_time:.1f}x")




# Test speed of individual moves
print("\nTesting speed of individual moves:")

# Time each move separately
move_times = {}
for move in range(12):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_trials):
        _ = finger_ix_fast_vec_torch(states, move)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    move_times[move] = time.time() - start

# Print results sorted by time
print("\nMove execution times (sorted fastest to slowest):")
sorted_moves = sorted(move_times.items(), key=lambda x: x[1])
baseline = sorted_moves[0][1]  # Fastest move time
for move, t in sorted_moves:
    relative = t/baseline
    print(f"Move {move:2d}: {t:.3f}s ({relative:.2f}x baseline)")



# Create an AC_presentation instance
ac = AC_presentation(max_relator_length=25)
ac.reset()












# Test scrambler generator
print("\n\n\n\n\nTesting AC_presentation scrambler:")
max_depth = 3
generator = ac.scrambler(max_depth)
# for i in range(5):  # Generate 5 samples
#     state, move = next(generator)
#     print(f"Sample {i}:")
#     print(f"State: {state}")
#     print(f"Move applied: {move}\n")


#     # Create two AC_presentation instances
#     ac1 = AC_presentation(max_relator_length=25)
#     ac2 = AC_presentation(max_relator_length=25)

#     # Reset both to initial state
#     ac1.reset()
#     ac2.reset()

#     # Define a sequence of test moves
#     test_moves = np.random.randint(0, len(ac.moves), size=100)
    
#     print(f"\n Sample {i} Testing finger_ix implementations:")
#     print("Initial states match:", np.array_equal(ac1.state, ac2.state))
    
#     # Apply moves and compare states after each
#     for i, move in enumerate(test_moves):
#         #print(f"\nApplying move {move}:")
        
#         # Apply move to first instance using original finger_ix
#         ac1.finger_ix(move)
        
#         # Apply move to second instance using fast finger_ix
#         ac2.finger_ix_fast(move)
        
#         # Compare states
#         states_match = np.array_equal(ac1.state, ac2.state)
#         #print(f"States match after move {i}: {states_match}")
        
#         if not states_match:
#             print("\n PROBLEM with move ",move)
#             print("State slow:", ac1.state)
#             print("State fast:", ac2.state)
#             print("Difference:", ac1.state - ac2.state)
#             exit()
#             break

# Benchmark comparison
print("\nBenchmarking finger_ix implementations:")

def benchmark_original(n_samples):
    start_time = time.time()
    ac = AC_presentation(max_relator_length=25)
    ac.reset()
    moves = np.random.randint(0, len(ac.moves), size=n_samples)
    for move in moves:
        ac.finger_ix(move)
    return time.time() - start_time

def benchmark_fast(n_samples): 
    start_time = time.time()
    ac = AC_presentation(max_relator_length=25)
    ac.reset()
    moves = np.random.randint(0, len(ac.moves), size=n_samples)
    for move in moves:
        ac.finger_ix_fast(move)
    return time.time() - start_time

n_samples = 1000
original_time = benchmark_original(n_samples)
fast_time = benchmark_fast(n_samples)

print(f"Time taken for {n_samples} samples:")
print(f"Original version: {original_time:.4f} seconds")
print(f"Fast version: {fast_time:.4f} seconds")
print(f"Speedup: {original_time / fast_time:.2f}x")

# Test vectorised implementation
print("\nTesting finger_ix_fast_vec implementation:")

# Create test instances
ac1 = AC_presentation(max_relator_length=7)
ac2 = AC_presentation(max_relator_length=7)
# Create batch of test states
states_testing=np.array(
                    [[1,2,-1,0,0,0,0,2,-1,1,-1,0,0,0],
                    [1,0,0,0,0,0,0,2,-1,1,-1,0,0,0],
                    [1,-2,1,0,0,0,0,2,-1,1,-1,2,0,0],
                    [1,0,0,0,0,0,0,2,-1,1,-1,1,1,0]])

batch_size =len(states_testing)
test_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print("init state:",states_testing) 
test_states = ac1.simplify_state_vec(states_testing).copy()
print("simp state:",test_states)
# ac1.state = test_states[0].copy()
# ac1.simplify_state()
# print("simplify_state:",ac1.state)

for move in test_moves:
    print(f"\nTesting move {move}:")
    
    # Apply move to each state individually using finger_ix_fast
    states_individual_fast = test_states.copy()
    states_individual_slow = test_states.copy()
    for i in range(batch_size):
        #print(f"\n{i}")
        ac1.state = states_individual_fast[i].copy()
        ac1.finger_ix_fast(move)
        states_individual_fast[i] = ac1.state
        #print("fast",ac1.state)
        ac2.state = states_individual_slow[i].copy()
        ac2.finger_ix(move)
        states_individual_slow[i] = ac2.state
        #print("slow",ac2.state)
        
        
    # Apply move to all states at once using finger_ix_fast_vec
    states_vectorised = ac2.finger_ix_fast_vec(test_states.copy(), move)
    
    
    # Compare results with both fast and slow implementations
    states_match_fast = np.array_equal(states_individual_fast, states_vectorised)
    states_match_slow = np.array_equal(states_individual_slow, states_vectorised)
    print(f"States match fast implementation: {states_match_fast}")
    print(f"States match slow implementation: {states_match_slow}")
    
    if not states_match_fast or not states_match_slow:

        print("First differing state:",)
        
        # Find first mismatch with either implementation
        mismatch_idx = None
        if not states_match_fast:
            mismatch_idx = np.where(~np.all(states_individual_fast == states_vectorised, axis=1))[0][0]
            print(f"Fast Individual {mismatch_idx}:", states_individual_fast[mismatch_idx])
        if not states_match_slow:
            mismatch_idx = np.where(~np.all(states_individual_slow == states_vectorised, axis=1))[0][0]
            print(f"Slow Individual {mismatch_idx}:", states_individual_slow[mismatch_idx])
        print("Original:", test_states[mismatch_idx])
        print("vectorised:", states_vectorised[mismatch_idx])
        print("Difference from fast:", states_individual_fast[mismatch_idx] - states_vectorised[mismatch_idx])
        print("Difference from slow:", states_individual_slow[mismatch_idx] - states_vectorised[mismatch_idx])
        exit()
        break
# Benchmark vectorised vs individual implementation
print("\nBenchmarking vectorised vs individual implementations:")

simplify_for_finger_benchmark = False
max_relator_length=1000
def benchmark_individual(batch_size, n_moves):
    ac = AC_presentation(max_relator_length=max_relator_length)
    states = np.tile(ac.state, (batch_size, 1))
    moves = np.random.randint(0, len(ac.moves), size=n_moves)
    
    start_time = time.time()
    for move in moves:
        for i in range(batch_size):
            ac.state = states[i].copy()
            ac.finger_ix_fast(move,simplify=simplify_for_finger_benchmark)
            states[i] = ac.state
    return time.time() - start_time

def benchmark_vectorised(batch_size, n_moves):
    ac = AC_presentation(max_relator_length=max_relator_length)
    states = np.tile(ac.state, (batch_size, 1))
    moves = np.random.randint(0, len(ac.moves), size=n_moves)
    
    start_time = time.time()
    for move in moves:
        states = ac.finger_ix_fast_vec(states, move,simplify=simplify_for_finger_benchmark)
    return time.time() - start_time

batch_size = 100
n_moves = 1000

individual_time = benchmark_individual(batch_size, n_moves)
vectorised_time = benchmark_vectorised(batch_size, n_moves)

print(f"\nTime taken for batch_size={batch_size}, n_moves={n_moves}, simplify={simplify_for_finger_benchmark}:")
print(f"Individual finger_ix version: {individual_time:.4f} seconds")
print(f"vectorised finger_ix version: {vectorised_time:.4f} seconds") 
print(f"Speedup: {individual_time / vectorised_time:.2f}x")


# Benchmark individual moves
print("\nBenchmarking individual moves:")
batch_size = 10
n_trials = 10

def benchmark_move(ac, states, move_idx, n_trials):
    start_time = time.time()
    for _ in range(n_trials):
        states = ac.finger_ix_fast_vec(states, move_idx)
    return time.time() - start_time

ac = AC_presentation(max_relator_length=max_relator_length)
states = np.tile(ac.state, (batch_size, 1))

move_times = {}
for i in range(12):  # Test all 12 moves
    time_taken = benchmark_move(ac, states.copy(), i, n_trials)
    move_times[i] = time_taken
    print(f"Move {i}: {time_taken:.4f} seconds")

# Find fastest and slowest moves
fastest_move = min(move_times.items(), key=lambda x: x[1])
slowest_move = max(move_times.items(), key=lambda x: x[1])

print(f"\nFastest move: {fastest_move[0]} ({fastest_move[1]:.4f} seconds)")
print(f"Slowest move: {slowest_move[0]} ({slowest_move[1]:.4f} seconds)")
print(f"Slowdown ratio: {slowest_move[1]/fastest_move[1]:.2f}x")

# Group moves by type and compare average times
basic_moves = {k:v for k,v in move_times.items() if k < 4}  # First 4 moves
conjugation_moves = {k:v for k,v in move_times.items() if k >= 4}  # Last 8 moves

avg_basic = sum(basic_moves.values()) / len(basic_moves)
avg_conj = sum(conjugation_moves.values()) / len(conjugation_moves)

print(f"\nAverage times by move type:")
print(f"Basic moves (0-3): {avg_basic:.4f} seconds")
print(f"Conjugation moves (4-11): {avg_conj:.4f} seconds")
print(f"Ratio: {avg_conj/avg_basic:.2f}x")

# Test simplify_state functionality
print("\nTesting simplify_state:")

# Create test instance
ac = AC_presentation(max_relator_length=25)

# Define test cases
test_cases = [
    {
        'name': 'Basic adjacent cancellation',
        'state': [1, -1, 2, -2],  # Should cancel to empty state
        'length': 4
    },
    {
        'name': 'No cancellation needed',
        'state': [1, 2, 1, 2],
        'length': 4
    },
    {
        'name': 'Multiple cancellations',
        'state': [1, -1, 2, -2, 1, -1, 2, -2],  # Should cancel completely
        'length': 8
    },
    {
        'name': 'Mixed cancellations', 
        'state': [1, -1, 2, 1, -2, 2],  # Should simplify to [1, 2]
        'length': 6
    },
    {
        'name': 'Mixed cancellations', 
        'state':  [2, -1, 1, -1, 1, -1] ,  # Should simplify to [1, 2]
        'length': 6
    }
]

# Test individual simplify_state
for i, test in enumerate(test_cases, 1):
    print(f"\nTest case {i}: {test['name']}")
    test_state = np.zeros(50, dtype=np.int8)
    test_state[0:test['length']] = test['state']
    ac.state = test_state.copy()
    ac.simplify_state()
    print(f"Original state: {test_state}")
    print(f"Simplified state: {ac.state}")

# Test simplify_state_vec functionality
print("\nTesting simplify_state_vec:")

# Create batch of test states
test_states = np.zeros((5, 50), dtype=np.int8) 
# Add all previous test cases
for i, test in enumerate(test_cases):
    test_states[i, 0:test['length']] = test['state']

# Run vectorised numpy simplification
simplified_states = ac.simplify_state_vec(test_states)

# Run vectorised torch simplification
test_states_torch = torch.from_numpy(test_states).to(torch.int8)
simplified_states_torch = simplify_state_vec_torch(test_states_torch)
simplified_states_torch = simplified_states_torch.cpu().numpy()

# Print and verify results
for i in range(5):
    print(f"\nTest case {i+1}:")
    print(f"Original state: {test_states[i]}")
    print(f"Numpy simplified state: {simplified_states[i]}")
    print(f"Torch simplified state: {simplified_states_torch[i]}")
    
    # Verify against non-vectorised version
    ac.state = test_states[i].copy()
    ac.simplify_state()
    print(f"Non-vectorised result: {ac.state}")
    testnumpyreg = np.array_equal(simplified_states[i], ac.state)
    testtorchreg = np.array_equal(simplified_states_torch[i], ac.state)
    testnumpytorch = np.array_equal(simplified_states[i], simplified_states_torch[i])   
    print(f"Numpy results match: {testnumpyreg}")
    print(f"Torch results match: {testtorchreg}")
    print(f"Numpy and Torch match: {testnumpytorch}")
    if not testnumpyreg or not testtorchreg or not testnumpytorch:
        print("Numpy: ",simplified_states[i])
        print("Torch: ",simplified_states_torch[i])
        print("Non-vectorised: ",ac.state)
        exit()


# Benchmark comparison of simplify_state implementations
print("\nBenchmarking simplify_state implementations:")
max_relator_length_for_simplify_benchmark = 25
def create_valid_states(batch_size,max_relator_length=25,num_nonzero_low=1,num_nonzero_high=None):
    print("Rigorous generation may take some time")
    if num_nonzero_high is None:
        num_nonzero_high = max_relator_length
    # Create AC instance
    ac = AC_presentation(max_relator_length=max_relator_length)
    
    # Initialize output array
    test_states = np.zeros((batch_size, max_relator_length*2), dtype=ac.DTYPE)
    
    # For each state in batch
    for i in range(batch_size):
        if i%1000==0: 
            print(f"{i}/{batch_size}")
        # Reset AC instance
        ac.reset()
        
        # Pick target length for either r0 or r1
        target_length = np.random.randint(num_nonzero_low, num_nonzero_high+1)
        target_relator = np.random.choice(['r0', 'r1'])
        
        # Keep scrambling until we hit target length
        while True:
            # Get current lengths
            r0_len = (ac.state[:max_relator_length] != 0).sum()
            r1_len = (ac.state[max_relator_length:] != 0).sum()
            
            # Check if we've hit target
            if target_relator == 'r0' and r0_len >= target_length:
                break
            if target_relator == 'r1' and r1_len >= target_length:
                break
                
            # Apply random move
            move = np.random.choice(ac.moves)
            old_state = ac.state.copy()
            ac.finger_ix_fast(move,simplify=True)
            if np.array_equal(ac.state,old_state):
                break

            
        # Store final state
        test_states[i] = ac.state
    # rel_length= max_relator_length
    # # Create valid states with non-zero elements followed by zeros
    # test_states = np.zeros((batch_size, rel_length*2), dtype=ac1.DTYPE)
    # # Generate lengths for both r0 and r1 relators
    # r0_lengths = np.random.randint(num_nonzero_low, num_nonzero_high, size=batch_size) # Random lengths between 1-9 elements
    # r1_lengths = np.random.randint(num_nonzero_low, num_nonzero_high, size=batch_size)
    
    # # Fill r0 relator (first half)
    # for i in range(batch_size):
    #     test_states[i, :r0_lengths[i]] = np.random.choice([-2, -1, 1, 2], size=r0_lengths[i])
        
    # # Fill r1 relator (second half) 
    # for i in range(batch_size):
    #     test_states[i, rel_length:rel_length+r1_lengths[i]] = np.random.choice([-2, -1, 1, 2], size=r1_lengths[i])
    # # Simplify the states using vectorized simplification
    # test_states = ac.simplify_state_vec(test_states)
    
    return test_states
    
def benchmark_original(n_samples, test_states):
    ac = AC_presentation(max_relator_length=max_relator_length_for_simplify_benchmark)
    # Create random test states
    output_states = test_states.copy()
    start_time = time.time()
    for _ in range(n_samples):
        for i in range(batch_size):
            ac.state = test_states[i].copy()
            ac.simplify_state()
            output_states[i] = ac.state
    return time.time() - start_time,output_states
    
def benchmark_fast(n_samples, test_states):
    ac = AC_presentation(max_relator_length=max_relator_length_for_simplify_benchmark)
    start_time = time.time()
    output_states = test_states.copy()
    for _ in range(n_samples):
        output_states = ac.simplify_state_vec(test_states)
    return time.time() - start_time,output_states
    
n_samples = 2
batch_size = 10000
test_states = create_valid_states(batch_size,max_relator_length=max_relator_length_for_simplify_benchmark)
original_time,output_states_original = benchmark_original(n_samples, test_states)
fast_time,output_states_fast = benchmark_fast(n_samples, test_states)
# Check if outputs match
states_match = np.array_equal(output_states_original, output_states_fast)
print(f"\nOutputs match: {states_match}")
if not states_match:
    mismatch = np.where(output_states_original != output_states_fast)
    print(f"First mismatch at: batch {mismatch[0][0]}, position {mismatch[1][0]}")
    print('orig')
    print(test_states[0:3])
    print('slow')
    print(output_states_original[0:3])
    print('vec')
    print(output_states_fast[0:3])
    exit()

    
print(f"Time taken for {n_samples} iterations with batch size {batch_size}:")
print(f"Original version: {original_time:.4f} seconds")
print(f"Fast version: {fast_time:.4f} seconds")
print(f"Speedup: {original_time / fast_time:.2f}x")


# Test apply_all_moves_to_all_states by comparing to finger_ix_fast_vec
def test_apply_all_moves(batch_size=100,max_relator_length=25):
    ac = AC_presentation(max_relator_length=max_relator_length)
    
    # Create random test states
    test_states = create_valid_states(batch_size)
    
    # Get results from apply_all_moves_to_all_states
    all_moves_result = ac.apply_all_moves_to_all_states(test_states)
    
    # Get results from finger_ix_fast_vec one move at a time
    individual_moves_result = np.zeros_like(all_moves_result)
    for move_idx in range(len(ac.moves)):
        states_copy = test_states.copy()
        ac.finger_ix_fast_vec(states_copy, move_idx)
        individual_moves_result[:, move_idx] = states_copy
        
    # Compare results
    matches = np.array_equal(all_moves_result, individual_moves_result)
    print("\nTesting apply_all_moves_to_all_states:")
    print(f"Results match: {matches}")
    if not matches:
        print("First mismatch:")
        mismatch = np.where(all_moves_result != individual_moves_result)
        print(f"Batch {mismatch[0][0]}, Move {mismatch[1][0]}, Position {mismatch[2][0]}")
        print(f"apply_all_moves value: {all_moves_result[mismatch[0][0], mismatch[1][0], mismatch[2][0]]}")
        print(f"individual moves value: {individual_moves_result[mismatch[0][0], mismatch[1][0], mismatch[2][0]]}")
        
test_apply_all_moves()

# Test torch version against numpy version
def test_torch_vs_numpy_finger_ix(batch_size=100,max_relator_length=25  ):
    import torch
    ac = AC_presentation(max_relator_length=max_relator_length)
    
    # Create random test states
    test_states = create_valid_states(batch_size,max_relator_length=max_relator_length)
    
    # Convert to torch tensor
    test_states_torch = torch.from_numpy(test_states).clone()
    
    # Test each move
    for move_idx in range(len(ac.moves)):
        # Get numpy result
        states_numpy = test_states.copy()
        states_numpy=ac.do_move_to_state_flexible_stateless_vec(states_numpy, move_idx)
        
        # Get torch result 
        states_torch = test_states_torch.clone()
        #states_torch= finger_ix_fast_vec_torch(states_torch, move_idx,simplify=True)
        states_torch= finger_ix_fast_vec_torch(states_torch, move_idx)
        states_torch_numpy = states_torch.numpy()
        
        # Compare results
        matches = np.array_equal(states_numpy, states_torch_numpy)
        #print(f"\nTesting move {move_idx}:")
        #print(f"Results match: {matches}")
        
        if not matches:
            print("test_torch_vs_numpy_finger_ix,First mismatch, move: ",move_idx)
            mismatch = np.where(states_numpy != states_torch_numpy)
            print(f"Batch no {mismatch[0][0]}, Position {mismatch[1][0]}")
            print(f"Numpy value: {states_numpy[mismatch[0][0], mismatch[1][0]]}")
            print(f"Torch value: {states_torch_numpy[mismatch[0][0], mismatch[1][0]]}")
            print("\nFirst few states:")
            print(test_states[mismatch[0][0]])
            print("Numpy:")
            print(states_numpy[mismatch[0][0]])
            print("Torch:")
            print(states_torch_numpy[mismatch[0][0]])
            exit()
            return

    # Test create_valid_states with num_nonzero=max_relator_length-1 and compare finger_ix_fast_vec results
    print("\nTesting finger_ix_fast_vec with states created using num_nonzero=max_relator_length-1...")
    test_states = create_valid_states(batch_size=100, max_relator_length=max_relator_length, num_nonzero_low=max_relator_length-3,num_nonzero_high=max_relator_length-1)
    
    # Test each move
    for move_idx in range(len(ac.moves)):
        # Get numpy result for finger_ix_fast_vec
        states_numpy = test_states.copy()
        states_numpy=ac.do_move_to_state_flexible_stateless_vec(states_numpy, move_idx)
        
        # Get torch result for finger_ix_fast_vec
        states_torch = torch.from_numpy(test_states).clone()
        #states_torch= finger_ix_fast_vec_torch(states_torch, move_idx,simplify=True)
        states_torch= finger_ix_fast_vec_torch(states_torch, move_idx)
        states_torch_numpy = states_torch.numpy()
        
        # Compare results for finger_ix_fast_vec
        matches = np.array_equal(states_numpy, states_torch_numpy)
        
        if not matches:
            print(f"\nMismatch found for finger_ix_fast_vec with move {move_idx}:")
            mismatch = np.where(states_numpy != states_torch_numpy)
            print(f"number in list {mismatch[0][0]}, Position {mismatch[1][0]}")
            print(f"Numpy value: {states_numpy[mismatch[0][0], mismatch[1][0]]}")
            print(f"Torch value: {states_torch_numpy[mismatch[0][0], mismatch[1][0]]}")
            print("\nFirst few states:")
            print("Numpy:")
            print(states_numpy[mismatch[0][0]])
            print("Torch:")
            print(states_torch_numpy[mismatch[0][0]])
            exit()

        # Get numpy result for finger_ix_fast by looping over states
        states_numpy = np.zeros_like(test_states)
        for i in range(len(test_states)):
            ac.state = copy.deepcopy(test_states[i])  # Set state for single example
            #ac.finger_ix_fast(move_idx)  # Apply move
            ac.do_move_to_state_flexible(move_idx)  # Apply move
            states_numpy[i] = ac.state  # Store result
        
        # Get torch result for finger_ix_fast_vec
        states_torch = torch.from_numpy(test_states).clone()
        states_torch= finger_ix_fast_vec_torch(states_torch, move_idx)
        states_torch_numpy = states_torch.numpy()
        
        # Compare results
        matches = np.array_equal(states_numpy, states_torch_numpy)
        
        if not matches:
            print(f"\nMismatch found for ac.do_move_to_state_flexible vs finger_ix_fast_vec_torch with move {move_idx}:")
            mismatch = np.where(states_numpy != states_torch_numpy)
            mismatch_idx = mismatch[0][0]
            print(f"failed state: {test_states[mismatch_idx]}")
            print(f"do_move_to_state_flexible results: {states_numpy[mismatch_idx]}")
            #ac.state = copy.deepcopy(states_numpy[mismatch_idx])
            #ac.simplify_state()
            print(f"finger_ix_fast_vec_torch results: {states_torch_numpy[mismatch_idx]}")
            #print(f"     try simplifying finger_ix_fast_vec results: {ac.simplify_state_vec_torch(torch.tensor(states_torch_numpy))[0]}")
            exit()
    print("All moves matched between numpy and torch versions for nearly-full states!")
    print("Both finger_ix_fast and finger_ix_fast_vec implementations match!")
    print("\nAll moves matched between numpy and torch versions!")
test_torch_vs_numpy_finger_ix()

# Benchmark torch vs numpy implementations
def benchmark_torch_vs_numpy(batch_size=1000, num_trials=2):
    simplify=True
    import torch
    import time
    ac = AC_presentation(max_relator_length=100)
    
    # Create random test states
    test_states = create_valid_states(batch_size,max_relator_length=ac.max_relator_length)
    test_states_torch = torch.from_numpy(test_states).clone()
    
    # Benchmark numpy version
    numpy_times = []
    for _ in range(num_trials):
        states_numpy = test_states.copy()
        start = time.time()
        for move_idx in range(len(ac.moves)):
            ac.finger_ix_fast_vec(states_numpy, move_idx,simplify=simplify)
        numpy_times.append(time.time() - start)
    
    # Benchmark torch version
    torch_times = []
    for _ in range(num_trials):
        states_torch = test_states_torch.clone()
        start = time.time()
        for move_idx in range(len(ac.moves)):
            #finger_ix_fast_vec_torch(states_torch, move_idx,simplify=simplify)
            finger_ix_fast_vec_torch(states_torch, move_idx)
        torch_times.append(time.time() - start)
        
    numpy_avg = sum(numpy_times) / len(numpy_times)
    torch_avg = sum(torch_times) / len(torch_times)
    
    print(f"\nBenchmark results (batch_size={batch_size}, trials={num_trials}):")
    print(f"Average numpy time: {numpy_avg:.4f}s")
    print(f"Average torch time: {torch_avg:.4f}s")
    print(f"Speedup: {numpy_avg/torch_avg:.2f}x")
    
print("\nRunning benchmarks...")
benchmark_torch_vs_numpy()

# Test that moves and their inverses cancel out
def test_moves_and_inverses(num_states=1000,max_relator_length=5):
    print("\nTesting moves and their inverses...")
    ac = AC_presentation(max_relator_length=max_relator_length)
    
    # Create 3 test states
    test_states = create_valid_states(num_states, max_relator_length=ac.max_relator_length)
    test_states = ac.simplify_state_vec(test_states)
    move_not_admissible = 0
    move_needs_larger_max_rel = 0
    
    for state_idx in range(num_states):
        #print(f"\nTesting state {state_idx}:")
        original_state = test_states[state_idx:state_idx+1].copy()
        #print(f"Original state: {original_state[0]}")
        
        # Test each move and its inverse
        for move in range(len(ac.moves)):
            test_state = original_state.copy()
            
            # Apply move then its inverse
            test_state1=ac.finger_ix_fast_vec(test_state, move)
            test_state2=ac.finger_ix_fast_vec(test_state1, ac.inverse_moves[move])
            if np.array_equal(test_state1, test_state):
                ac.state = original_state[0]

                may_need_larger_max_rel = np.count_nonzero(ac.state) > ac.max_relator_length or \
                    np.count_nonzero(ac.state[ac.max_relator_length:])+2 > max_relator_length or\
                    np.count_nonzero(ac.state[:ac.max_relator_length])+2 > max_relator_length


                aclarge = AC_presentation(max_relator_length=max_relator_length*2)
                large_state = np.zeros(max_relator_length*2*2,dtype=aclarge.state.dtype)
                large_state[:max_relator_length] = original_state[0,:max_relator_length]
                large_state[2*max_relator_length:3*max_relator_length] = original_state[0,max_relator_length:]
                aclarge.state = large_state.copy()
                largecheck = aclarge.check_if_move_is_admissible(move)


                aclarge.finger_ix_fast(move)
                large_moved_state = aclarge.state

                move_is_not_admissible = not ac.check_if_move_is_admissible(move)
                if move_is_not_admissible:
                    #all fine, it's not admissible
                    #print("Move is not admissible")
                    move_not_admissible+=1
                elif largecheck and may_need_larger_max_rel:
                    #print("Explained by needing a larger max_rel to do make move admissible")
                    move_needs_larger_max_rel+=1
                elif np.array_equal(large_moved_state,large_state):
                    aclarge.state = large_state.copy()
                    aclarge.do_move_to_state_flexible(move)
                    if not np.array_equal(aclarge.state,large_moved_state):
                        print(aclarge.state)
                        print(large_moved_state)
                        #print("Larger state agrees after double checking - nothing happens. Ok?")
                        print("Larger state does not agree - something is wrong")
                        exit()
                else:
                    print(large_state)
                    print(large_moved_state)
                    print(f"test_state did not change after move {move}, which is admissible but didn't change the state")
                    print(f"test_state: {test_state}")
                    print(f"test_state1: {test_state1}")
                    print("move is admissible?: ", not move_is_not_admissible)
                    exit()
            elif not np.array_equal(test_state2, original_state):
                # Check if we got back to original state
                #print(f"Move {move} and inverse {ac.inverse_moves[move]} don't cancel!")
                #print(f"State after move+inverse: {test_state2[0]}")
                ac.state = original_state[0]
                #print('trying flexible')
                #print("Move: ",move)
                #print("Inverse: ",ac.inverse_moves[move])
                #print(ac.max_relator_length)
                #print("Possible moves: ",ac.check_conditions_return_possibilities())
                #print(ac.state)
                succeeded1 = ac.do_move_to_state_flexible(move)
                
                after_move = ac.state.copy()
                if not np.array_equal(after_move, test_state1[0]):
                    print(f"The state after move {move} is not the same as the state after the move {move} in finger_ix_fast_vec")

                    print(f"succeeded in acting with move {move}: {succeeded1}")
                    print(f"succeeded in acting with move {ac.inverse_moves[move]}: {succeeded2}")
                    if np.array_equal(test_state1[0],original_state[0]):
                        print("the finger_ix_fast_vec failed to change the state - makes sense?? maybe?")
                        exit()
                    else:
                        print("unexplained")
                        exit()

                succeeded2 = ac.do_move_to_state_flexible(ac.inverse_moves[move])
                after_inverse = ac.state.copy()
                fail1= np.array_equal(after_move, after_inverse)
                fail2= not np.array_equal(after_inverse, original_state[0])
                fail3 = not np.array_equal(after_inverse, test_state2[0])
                if fail1 or fail2:
                    print(f"succeeded in acting with move {move}: {succeeded1}")
                    print(f"succeeded in acting with move {ac.inverse_moves[move]}: {succeeded2}")
                    print("The problem is:")
                    if fail1:
                        print("   The inverse move didn't do anything")
                    if fail2:
                        print("   The state after the inverse move is not the same as the original state")
                    print(f"Move {move} and inverse {ac.inverse_moves[move]} don't cancel!")
                    print(f"State originally: {original_state[0]}")
                    print(f"State after move from fifv: {test_state1[0]}")
                    print(f"State after move+inverse from fifv: {test_state2[0]}")
                    print(f"State after move from do_move: {after_move}")
                    print(f"State after move+inverse from do_move: {after_inverse}")
                    exit()
                    break
            #elif np.array_equal(test_state1, test_state):
            #    print(f"FAILED: test_state did not change after m")
            #    exit()
            #elif  np.array_equal(test_state2, test_state1):
            #    print(f"FAILED: test_state did not change after m_inverse")
        #else:
        #    print("PASSED: All moves and their inverses cancel correctly")
    print(f"Tested {num_states} states successfully")   
    print(f"Move not admissible: {move_not_admissible}")
    print(f"Move needs larger max_rel: {move_needs_larger_max_rel}")
            
print("\nRunning move/inverse tests...")
test_moves_and_inverses(num_states=10000)



# Test finger_ix_fast_vec_torch_list_of_moves by comparing it to finger_ix_fast_vec_torch
def test_finger_ix_fast_vec_torch_list_of_moves(batch_size=100, max_relator_length=25):
    import torch
    ac = AC_presentation(max_relator_length=max_relator_length)
    
    # Create random valid test states
    test_states = create_valid_states(batch_size, max_relator_length=max_relator_length)
    test_states_torch = torch.from_numpy(test_states).clone()
    
    # Generate a random list of moves
    moves = torch.randint(0, len(ac.moves), (batch_size,), dtype=torch.long)
    
    # Apply moves using the list_of_moves function
    list_result = finger_ix_fast_vec_torch_list_of_moves(test_states_torch, moves)
    
    # Apply moves individually using the single move function
    individual_result = torch.zeros_like(test_states_torch)
    for i, move in enumerate(moves):
        individual_result[i] = finger_ix_fast_vec_torch(test_states_torch[i:i+1], move.item())
    
    # Verify that results from both methods match
    matches = torch.all(list_result == individual_result)
    print("\nTesting finger_ix_fast_vec_torch_list_of_moves:")
    print(f"Results match: {matches}")
    
    if not matches:
        mismatch = torch.where(list_result != individual_result)
        print(f"First mismatch at: batch {mismatch[0][0]}, position {mismatch[1][0]}")
        print(f"List of moves value: {list_result[mismatch[0][0], mismatch[1][0]]}")
        print(f"Individual moves value: {individual_result[mismatch[0][0], mismatch[1][0]]}")
        print(f"Move applied: {moves[mismatch[0][0]]}")

print("\nRunning list of moves test...")        
test_finger_ix_fast_vec_torch_list_of_moves()


# Test that all implementations of applying moves give consistent results and compare speeds
def test_all_move_implementations(batch_size=100, max_relator_length=25):
    import torch
    import time
    ac = AC_presentation(max_relator_length=max_relator_length)
    
    # Create random valid test states
    test_states = create_valid_states(batch_size, max_relator_length=max_relator_length)
    test_states_torch = torch.from_numpy(test_states).clone()
    
    # Generate random moves
    moves = torch.randint(0, len(ac.moves), (batch_size,), dtype=torch.long)
    
    print("\nTesting consistency and speed across all move application implementations:")
    
    # Test apply_all_moves implementations
    start = time.time()
    all_moves_result = OLD_apply_all_moves_to_all_states_torch(test_states_torch)
    orig_time = time.time() - start
    
    start = time.time()
    all_moves_new_result = apply_all_moves_to_all_states_torch(test_states_torch)
    new_time = time.time() - start
    
    moves_match = torch.allclose(all_moves_result, all_moves_new_result)
    print(f"apply_all_moves implementations match: {moves_match}")
    print(f"Original implementation time: {orig_time:.4f}s")
    print(f"New implementation time: {new_time:.4f}s")
    print(f"Speed difference: {orig_time/new_time:.2f}x {'faster' if new_time < orig_time else 'slower'}")
    
    if not moves_match:
        mismatch = torch.where(all_moves_result != all_moves_new_result)
        print(f"First mismatch at: batch {mismatch[0][0]}, move {mismatch[1][0]}, position {mismatch[2][0]}")
        print(f"Original value: {all_moves_result[mismatch[0][0], mismatch[1][0], mismatch[2][0]]}")
        print(f"New value: {all_moves_new_result[mismatch[0][0], mismatch[1][0], mismatch[2][0]]}")
    
    # Test list_of_moves implementations
    start = time.time()
    list_moves_result = finger_ix_fast_vec_torch_list_of_moves(test_states_torch, moves)
    new_time = time.time() - start
    
    start = time.time()
    all_pick_result = finger_ix_fast_vec_torch_list_of_moves_all_pick(test_states_torch, moves)
    pick_time = time.time() - start

    start = time.time()
    old_result = OLD_finger_ix_fast_vec_torch_list_of_moves(test_states_torch, moves)
    old_time = time.time() - start
    list_match = torch.allclose(list_moves_result, all_pick_result)
    list_match_old = torch.allclose(list_moves_result, old_result)
    print(f"\nlist_of_moves implementations match: {list_match}")
    print(f"list_of_moves implementations match: {list_match_old}")
    print(f"NEW implementation time: {new_time:.4f}s")
    print(f"all_pick implementation time: {pick_time:.4f}s") 
    print(f"Speed difference: {pick_time/new_time:.2f}x {'faster' if pick_time > new_time else 'slower'}")
    print(f"OLD implementation time: {old_time:.4f}s")
    print(f"Speed difference: {old_time/new_time:.2f}x {'faster' if old_time > new_time else 'slower'}")
    if not list_match:
        mismatch = torch.where(list_moves_result != all_pick_result)
        print(f"First mismatch at: batch {mismatch[0][0]}, position {mismatch[1][0]}")
        print(f"NEW value: {list_moves_result[mismatch[0][0], mismatch[1][0]]}")
        print(f"all_pick value: {all_pick_result[mismatch[0][0], mismatch[1][0]]}")
        print(f"Move applied: {moves[mismatch[0][0]]}")
        exit()
    if not list_match_old:
        mismatch_old = torch.where(list_moves_result != old_result)
        print(f"First mismatch at: batch {mismatch_old[0][0]}, position {mismatch_old[1][0]}")
        print(f"OLD value: {old_result[mismatch_old[0][0], mismatch_old[1][0]]}")
        print(f"Move applied: {moves[mismatch_old[0][0]]}")
        exit()

print("\nRunning implementation consistency tests...")
test_all_move_implementations()
