"""
Forward process for the AC presentation.
Must define the environment before using this file.
Must define TrainConfig before using this file.
"""

import torch
import torch.nn as nn
import numpy as np

import zlib
from typing import Union, List
from NN_models import *
from torch_AC import *
from AC_heuristic_search import *
import wandb
import time
import matplotlib.pyplot as plt
from tqdm import tqdm



#env = AC_presentation(max_relator_length=25)

# class ExperienceBuffer:
#     def __init__(self, capacity=100):
#         """
#         Stores full trajectories rather than individual tiransitions.
#         capacity: number of trajectories to store
#         """
#         self.capacity = capacity
#         self.trajectories = []  # Each trajectory is a tuple of (states, actions, returns, advantages, log_probs)

#     def add_trajectory(self, states, actions, returns, advantages, log_probs):
#         """Add a full trajectory to the buffer"""
#         if len(self.trajectories) >= self.capacity:
#             self.trajectories.pop(0)
            
#         self.trajectories.append((states, actions, returns, advantages, log_probs))

#     def sample(self, batch_size):
#         """
#         Sample batch_size transitions randomly from across all stored trajectories.
#         Returns tensors of shape [batch_size, ...]
#         """
#         # First randomly select which trajectories to sample from
#         traj_indices = torch.randint(0, len(self.trajectories), (batch_size,))
        
#         # Then randomly select timesteps from each chosen trajectory
#         sampled_states = []
#         sampled_actions = []
#         sampled_returns = []
#         sampled_advantages = []
#         sampled_log_probs = []
        
#         for idx in traj_indices:
#             traj = self.trajectories[idx]
#             timestep = torch.randint(0, len(traj[0]), (1,)).item()
            
#             sampled_states.append(traj[0][timestep])
#             sampled_actions.append(traj[1][timestep])
#             sampled_returns.append(traj[2][timestep])
#             sampled_advantages.append(traj[3][timestep])
#             sampled_log_probs.append(traj[4][timestep])
            
#         return (
#             torch.stack(sampled_states),
#             torch.stack(sampled_actions),
#             torch.stack(sampled_returns),
#             torch.stack(sampled_advantages),
#             torch.stack(sampled_log_probs)
#         )

class PPO_policy_net(nn.Module):
    def __init__(self, state_dim, num_moves=12, hidden_dim=128, depth= 5):
        super(PPO_policy_net, self).__init__()
        self.num_classes = 5  # For one-hot encoding
        self.input_dim = state_dim*self.num_classes
        self.dropout_rate = 0.0
        self.policy = nn.Sequential(
            LinearBlock(self.input_dim, hidden_dim, self.dropout_rate),
            LinearBlock(hidden_dim, hidden_dim, self.dropout_rate),
            *[ResidualBlock_no_time(hidden_dim, self.dropout_rate) for _ in range(depth)],
            LinearBlock(hidden_dim, num_moves,0),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs):
        x = nn.functional.one_hot((inputs + 2).long(), num_classes=self.num_classes).to(torch.float)
        x = x.reshape(-1, self.input_dim)
        return self.policy(x)


class PPO_value_net(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, depth= 5):
        super(PPO_value_net, self).__init__()
        self.num_classes = 5  # For one-hot encoding
        self.input_dim = state_dim*self.num_classes
        self.dropout_rate = 0.0 
        self.value = nn.Sequential(
            LinearBlock(self.input_dim, hidden_dim, self.dropout_rate),
            LinearBlock(hidden_dim, hidden_dim, self.dropout_rate),
            *[ResidualBlock_no_time(hidden_dim, self.dropout_rate) for _ in range(depth)],
            LinearBlock(hidden_dim, 1,0),
        )

    def forward(self, inputs):
        x = nn.functional.one_hot((inputs + 2).long(), num_classes=self.num_classes).to(torch.float)
        x = x.reshape(-1, self.input_dim)
        return self.value(x)

def compute_reward_vec(states,moves_mask,t):
    # Count number of nonzero entries in each state vector
    max_relator_length = states.shape[-1]//2
    penalty_length = max_relator_length//2
    r0s,r1s= states[:,0:max_relator_length],states[:,max_relator_length:]
    nonzero_counts_0   = (r0s != 0).sum(dim=1)
    nonzero_counts_1 = (r1s != 0).sum(dim=1)
    # Calculate penalty based on how much the count exceeds max_relator_length
    # Penalty is proportional to the excess length
    #excess_length_0 = torch.clamp(nonzero_counts_0 - penalty_length, min=0)
    #excess_length_1 = torch.clamp(nonzero_counts_1 - penalty_length, min=0)
    #penalty_0 = - 0.5*excess_length_0/penalty_length  # Scale penalty by -0.1 per excess element
    #penalty_1 = - 0.5*excess_length_1/penalty_length  # Scale penalty by -0.1 per excess element
    #penalty = (penalty_0 + penalty_1)
    can_do_0_4 = moves_mask[:,:4].float().mean(dim=-1)
    #reward_can_do_4_11 = moves_mask[:,4:].sum(dim=-1).float()
    reward = can_do_0_4# + reward_can_do_4_11
    
    
    reward_info = {
        'can_do_0_4': (can_do_0_4),
        #'penalty_1': (penalty_1),
        #'total_penalty': (penalty),
        'total_nonzero_counts': (nonzero_counts_0 + nonzero_counts_1).float()
    }
    
    return reward, reward_info

def batch_count_unique_if_indices(tensor):
    # tensor shape: (batch_size, scramble_length)
    return (torch.zeros((tensor.shape[0],tensor.max()+1),device=tensor.device).scatter_(-1, tensor, 1) > 0).sum(-1)
    

class PPOAgent:
    def __init__(self, env, lr=0.001, gamma=0.99,
                 epsilon=0.2, entropy_coef=1, value_coef=0.5, hidden_dim=128, depth= 5,use_experience_buffer=True):
        self.env = env
        self.state_dim = env.state_dim
        self.num_moves = env.num_moves

        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Policy network
        self.policy = PPO_policy_net(self.state_dim, self.num_moves, hidden_dim, depth)

        # Value network
        self.value = PPO_value_net(self.state_dim, hidden_dim,depth)

        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) +
                                          list(self.value.parameters()), lr=lr)
        self.reference_kolmogorov_complexity = approximate_kolmogorov_complexity_array(np.random.randint(-2,2,size=(len(env.goal))))

        self.min_proportion_of_0_4 = 0.1/4
        self.min_proportion_of_4_11 = 0.9/8
        self.at_least_hard_cutoff_0_4 = 0.01
        self.at_least_hard_cutoff_4_11 = 0.02

        #self.buffer = ExperienceBuffer()

    def get_action(self, states):
        """
        Vectorized method to get distribution over actions for a batch of states.
        Returns a sampled action, its log_prob, and state values for each state in the batch.
        """
        with torch.no_grad():
            action_probs = self.policy(states)
            mask_out_bad_moves,states_out = mask_impossible_moves(states,self.env,return_states_out=True)


            #states_out = apply_all_moves_to_all_states_torch(states.reshape(-1,env.state_dim)).reshape(states.shape[0],env.num_moves,env.state_dim)
            #mask_out_bad_moves = ~(states_out == states.unsqueeze(1).repeat(1,env.num_moves,1)).all(dim=-1)

            #print(mask_good_moves.shape)
            #print(action_probs.shape)
            action_probs = action_probs*mask_out_bad_moves
            action_probs = action_probs/action_probs.sum(dim=-1,keepdim=True)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            states_next = states_out[torch.arange(states.shape[0]),action]
            log_prob = dist.log_prob(action)
            value = self.value(states)
        return action, log_prob, value,states_next,mask_out_bad_moves,action_probs

    

    def compute_advantages(self, rewards, values):
        """
        Computes advantages in a simple reversed pass, vectorized for each environment in the batch.
        Assumes rewards and values are lists/tensors of shape [steps, batch_size].
        """
        advantages_list = []
        returns_list = []
        batch_size = rewards[0].size(0)
        running_return = torch.zeros(batch_size,device=rewards[0].device)

        # reverse pass
        for r, v in zip(reversed(rewards), reversed(values)):
            running_return = r + self.gamma * running_return
            advantage = running_return - v
            advantages_list.insert(0, advantage)
            returns_list.insert(0, running_return.clone())

        advantages = torch.stack(advantages_list)  # Now shape is [timesteps, batch_size]
        returns = torch.stack(returns_list)  # Now shape is [timesteps, batch_size]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def collect_experience(self, env, initial_states, scramble_length=100):
        """
        Collects a batch of trajectories using a vectorized stepping approach.
        Now stores full trajectories rather than flattened transitions.
        """
        current_states = initial_states
        batch_size = current_states.size(0)
        # Pre-allocate tensors for storing trajectory data
        all_states = torch.zeros((scramble_length, batch_size, current_states.size(-1)), device=current_states.device,dtype=torch.long)
        all_actions = torch.zeros((scramble_length, batch_size), device=current_states.device,dtype=torch.long)
        all_rewards = torch.zeros((scramble_length, batch_size), device=current_states.device,dtype=torch.float)
        all_values = torch.zeros((scramble_length, batch_size), device=current_states.device,dtype=torch.float)
        all_log_probs = torch.zeros((scramble_length, batch_size), device=current_states.device,dtype=torch.float)
        all_mask_good_moves = torch.zeros((scramble_length, batch_size,env.num_moves), device=current_states.device,dtype=torch.bool)
        times = torch.arange(scramble_length,device=current_states.device).unsqueeze(1).repeat(1,batch_size)
        all_action_probs = torch.zeros((scramble_length, batch_size,env.num_moves), device=current_states.device,dtype=torch.float)
        # Initialize reward component tracking
        reward_components = {
            'penalty_0': torch.zeros((scramble_length,batch_size), device=current_states.device),
            #'penalty_1': torch.zeros((scramble_length,batch_size), device=current_states.device),
            'total_penalty': torch.zeros((scramble_length,batch_size), device=current_states.device),
            #'diversity_bonus_entire_sample': torch.zeros((scramble_length, batch_size), device=current_states.device),
            'diversity_bonus_trajectory': torch.zeros((scramble_length,batch_size), device=current_states.device),
            'total_nonzero_counts': torch.zeros((scramble_length,batch_size), device=current_states.device),
            'kolmogorov_complexity': torch.zeros((scramble_length,batch_size), device=current_states.device)
        }
        
        for t in range(scramble_length):
            # Time get_action
            actions, log_probs, values,next_states,moves_mask,action_probs = self.get_action(current_states)

            #reward, reward_info = compute_reward_vec(next_states,moves_mask,t)
            all_states[t] = current_states
            all_actions[t] = actions
            #all_rewards[t] = reward
            all_values[t] = values.squeeze(-1)
            all_log_probs[t] = log_probs
            all_mask_good_moves[t] = moves_mask
            all_action_probs[t] = action_probs
            #moves_mask_all[t] = moves_mask
            
            # Store reward components
            #for key in reward_info:
            #    reward_components[key][t] = reward_info[key]

            current_states = next_states

        #compute rewards all at once
        _,rewards_info = compute_reward_vec(all_states.reshape(-1,all_states.shape[-1]),all_mask_good_moves.reshape(-1,all_mask_good_moves.shape[-1]),times.reshape(-1,times.shape[-1]))
        #reward_vecs=  reward_vecs.reshape(scramble_length,batch_size)
        for key in rewards_info:
            reward_components[key] = rewards_info[key].reshape(scramble_length,batch_size)

        # Calculate diversity bonuses using hashing - this is sufficient (hash collisions are rare)
        hash_states = hash_vectors_torch(all_states.reshape(-1, all_states.size(-1)),force_no_collision=False,monitor_collision=False)
        
        _,reverse_indices = torch.unique(hash_states,return_inverse=True)
        # Get number of unique states with hashing
        reverse_indices_reshaped= reverse_indices.reshape(scramble_length,batch_size).transpose(0,1)

        #unique_states_in_entire_sample= uniques.shape[0]
        unique_states_across_each_trajectory = batch_count_unique_if_indices(reverse_indices_reshaped)
        
        #total_states = scramble_length * batch_size
        #diversity_bonus_entire_sample = (unique_states_in_entire_sample / total_states) * scramble_length
        diversity_bonus_across_each_trajectory = (unique_states_across_each_trajectory / scramble_length)# number from 0 to 1
        diversity_bonus_across_each_trajectory = diversity_bonus_across_each_trajectory.unsqueeze(0).repeat(scramble_length,1)  
        # Add diversity bonuses to final step rewards and store in components
        #reward_components['diversity_bonus_entire_sample'] = diversity_bonus_entire_sample * scramble_length
        #all_rewards[-1] = all_rewards[-1] + diversity_bonus_entire_sample * scramble_length + diversity_bonus_across_each_trajectory * scramble_length

        #kolmogorov_complexity = 1/self.reference_kolmogorov_complexity* torch.tensor(
        #    list_approximate_kolmogorov_complexity_array(all_states.reshape(-1,all_states.shape[-1]).to('cpu')),
        #    device=all_states.device,dtype=torch.float).reshape(scramble_length,batch_size)
        kolmogorov_complexity = torch.zeros_like(all_rewards)

        #negative reward if 0-4 are not at least 10%
       

        #action_penalty_if_not_0_4 = -(1/min_proportion_of_0_4)*torch.relu(min_proportion_of_0_4 - all_action_probs[:,:,:4].sum(dim=-1))
        action_penalty_if_not_0_4 = -(1/4)*(1/self.min_proportion_of_0_4)*torch.abs((self.min_proportion_of_0_4 - all_action_probs[:,:,:4])).sum(dim=-1)
        action_penalty_if_not_4_11 = -(1/8)*(1/self.min_proportion_of_4_11)*torch.abs((self.min_proportion_of_4_11 - all_action_probs[:,:,4:])).sum(dim=-1)
        penalty_if_0_4_not_at_least_x_each = -(1/4)*(1/self.at_least_hard_cutoff_0_4)*torch.relu((self.at_least_hard_cutoff_0_4 - all_action_probs[:,:,:4])).sum(dim=-1)#penalise if 0-4 probs each <0.01
        penalty_if_4_11_not_at_least_y_each = -(1/8)*(1/self.at_least_hard_cutoff_4_11)*torch.relu((self.at_least_hard_cutoff_4_11 - all_action_probs[:,:,4:])).sum(dim=-1)#penalise if 4-11 probs each <0.02


        #total_nonzero_counts_close_to_correct_mean = (rewards_info['total_nonzero_counts'].reshape(scramble_length,batch_size).mean(dim=0,keepdim=True)/(2*env.max_relator_length))#0 if good, 0.5 if full, 0 if short
        #correct_mean_proportion = 0.3  
        penalty_keeping_mean_close_to_correct_mean = torch.zeros_like(all_rewards)#-1*(total_nonzero_counts_close_to_correct_mean-correct_mean_proportion)**2/((1-correct_mean_proportion)**2)

        all_rewards = 2*action_penalty_if_not_0_4 + 2*action_penalty_if_not_4_11 + 1*diversity_bonus_across_each_trajectory  + 0.5*kolmogorov_complexity +4*penalty_if_0_4_not_at_least_x_each + 4*penalty_if_4_11_not_at_least_y_each#+ 1*penalty_keeping_mean_close_to_correct_mean

        reward_components['diversity_bonus_trajectory'] = diversity_bonus_across_each_trajectory
        reward_components['kolmogorov_complexity'] = kolmogorov_complexity
        reward_components['total_rewards'] = all_rewards
        reward_components['action_penalty_if_not_0_4'] = action_penalty_if_not_0_4
        reward_components['action_penalty_if_not_4_11'] = action_penalty_if_not_4_11
        reward_components['penalty_keeping_mean_close_to_correct_mean'] = penalty_keeping_mean_close_to_correct_mean
        reward_components['penalty_if_0_4_not_at_least_x_each'] = penalty_if_0_4_not_at_least_x_each
        reward_components['penalty_if_4_11_not_at_least_y_each'] =penalty_if_4_11_not_at_least_y_each

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(all_rewards, all_values)
        dict_reward_components_means = {key: reward_components[key].mean(dim=0) for key in reward_components}

        return all_states, all_actions, returns, advantages, all_log_probs, all_mask_good_moves, dict_reward_components_means


    def update(self, states, actions, returns, advantages, old_log_probs, good_moves_mask, n_epochs_per_sample=1):
        """
        Update policy and value networks using collected experience.
        Now takes trajectory data directly rather than sampling from a buffer.
        Uses exclusive subsets of the data for each optimization step.
        """
        timesteps = states.shape[0]
        batch_size = states.shape[1] 
        state_dim = states.shape[-1]
        
        # Flatten trajectory data
        states_flat = states.reshape(-1, state_dim)
        actions_flat = actions.reshape(-1)
        returns_flat = returns.reshape(-1) 
        advantages_flat = advantages.reshape(-1)
        old_log_probs_flat = old_log_probs.reshape(-1)
        good_moves_mask_flat = good_moves_mask.reshape(-1,self.env.num_moves)

        # Calculate total number of samples and subset size
        total_samples = states_flat.shape[0]
        subset_size = total_samples // n_epochs_per_sample

        # Generate random permutation for the entire dataset
        indices = torch.randperm(total_samples, device=states_flat.device)

        losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(n_epochs_per_sample):
            # Get exclusive subset for this epoch
            start_idx = epoch * subset_size
            end_idx = start_idx + subset_size
            epoch_indices = indices[start_idx:end_idx]

            # Select data for this epoch
            epoch_states = states_flat[epoch_indices]
            epoch_actions = actions_flat[epoch_indices]
            epoch_returns = returns_flat[epoch_indices]
            epoch_advantages = advantages_flat[epoch_indices]
            epoch_old_log_probs = old_log_probs_flat[epoch_indices]
            epoch_good_moves_mask = good_moves_mask_flat[epoch_indices]

            # Forward pass
            action_probs = self.policy(epoch_states)
            action_probs = action_probs * epoch_good_moves_mask
            action_probs = action_probs / action_probs.sum(dim=-1, keepdims=True)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(epoch_actions)

            # Value prediction
            values = self.value(epoch_states).squeeze()

            # Compute losses
            ratios = torch.exp(new_log_probs - epoch_old_log_probs)
            surr1 = ratios * epoch_advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * epoch_advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(values, epoch_returns)
            entropy = dist.entropy().mean()

            # Combined loss
            loss = (policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy)

            # Update networks
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())

        return losses, policy_losses, value_losses, entropy_losses

    def train(self, env, initial_state, n_episodes=1000,
              scramble_length=100, batch_size=64, n_epochs_per_batch=10):
        """
        Trains the agent by collecting vectorized experience each iteration.
        """
        # Initialize wandb
        training_start_time = str(int(time.time()))
        wandb.init(project="andrews-curtis",
                  name=f"mrl_{env.max_relator_length}_sl{scramble_length}_bs{batch_size}_ts{training_start_time}",  # Add descriptive run name
                  config={
                      # Training hyperparameters
                      "n_episodes": n_episodes,
                      "scramble_length": scramble_length,
                      "batch_size": batch_size,
                      "n_epochs_per_batch": n_epochs_per_batch,
                      
                      # PPO hyperparameters
                      "value_coef": self.value_coef,  # Value loss coefficient
                      "entropy_coef": self.entropy_coef,  # Entropy bonus coefficient
                      "epsilon": self.epsilon,  # PPO clipping parameter
                      
                      # Model details
                      "policy_architecture": str(self.policy),  # Log model architectures
                      "value_architecture": str(self.value),
                      "optimizer": str(self.optimizer),  # Log optimizer details
                  },
                  tags=["PPO", "Andrews-Curtis"])  # Add relevant tags
        
        # Expand initial_state into a batch for vectorized collection
        init_states_batch = initial_state.unsqueeze(0).repeat(batch_size, 1)
        self.device = init_states_batch.device
        self.policy.to(self.device)
        self.value.to(self.device)
        
        # For plotting
        returns_history = []
        plt.ion() # Enable interactive mode
        fig, ax = plt.subplots()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Return')
        ax.set_title('Training Progress')
        
        try:
            pbar = tqdm(range(n_episodes))
            start_time = time.time()
            for episode in pbar:
                episode_start = time.time()
                with torch.no_grad():
                    states,actions,returns,advantages,old_log_probs,good_moves_mask,reward_components = self.collect_experience(env, init_states_batch, scramble_length)
                collection_time = time.time() - episode_start
                
                update_start = time.time()
                losses, policy_losses, value_losses, entropy_losses = self.update(states,actions,returns,advantages,old_log_probs,good_moves_mask,n_epochs_per_sample=n_epochs_per_batch)
                update_time = time.time() - update_start
                
                # Calculate mean return across all trajectories
                mean_return = returns.mean().item()
                returns_history.append(mean_return)
                
                # Calculate mean reward components
                #mean_reward_components = mean_reward_components
                
                # Log metrics to wandb for each optimization step
                for i in range(len(losses)):
                    log_dict = {
                        "mean_return": mean_return if i == 0 else None,  # Only log return once per episode
                        "total_loss": losses[i],
                        "policy_loss": policy_losses[i],
                        "value_loss": value_losses[i],
                        "entropy": entropy_losses[i],
                    }
                    # Add reward components to logging
                    if i == 0:  # Only log components once per episode
                        for key, value in reward_components.items():
                            #print(key,value.shape)
                            log_dict[f"reward_{key}"] = value.mean()
                            
                    wandb.log(log_dict, step=episode * n_epochs_per_batch + i)
                
                pbar.set_description(f"Final Loss: {losses[-1]:.4f},avg return: {mean_return:.4f},  Policy Loss: {policy_losses[-1]:.4f}, Value Loss: {value_losses[-1]:.4f}, Entropy: {entropy_losses[-1]:.4f}, collect: {collection_time:.2f}s, update: {update_time:.2f}s")
                    
            total_time = time.time() - start_time
            print(f"\nTotal training time: {total_time:.2f}s")
            wandb.log({"total_training_time": total_time})
            
            # Save models
            torch.save(self.policy.state_dict(), 'policy.pt')
            torch.save(self.value.state_dict(), 'value.pt')
            print("Models saved to policy.pt and value.pt")
                    
        except KeyboardInterrupt or Exception:
            print("\nTraining interrupted by user or exception")
            total_time = time.time() - start_time
            print(f"Training time before interruption: {total_time:.2f}s")
            wandb.log({"total_training_time": total_time})
            if episode>50:
                # Save models even if interrupted
                torch.save(self.policy.state_dict(), 'policy_interrupted.pt')
                torch.save(self.value.state_dict(), 'value_interrupted.pt')
            print("Models saved to policy_interrupted.pt and value_interrupted.pt")
            
        wandb.finish()

def approximate_kolmogorov_complexity_array(arr: np.ndarray,
                                          normalize: bool = True,baseline: float = 0) -> float:
    """
    Approximates Kolmogorov complexity for arrays with integers in [-2, 2]

    Args:
        arr: Input array or list with integers in range [-2, 2]
        normalize: Whether to normalize by log of array length

    Returns:
        float: Approximated Kolmogorov complexity
    """
    if isinstance(arr,torch.Tensor):
        arr = arr.cpu().numpy()
    # Convert to numpy array if list
    #if isinstance(arr, Union[list,np.ndarray,tf]):
    #    arr = np.array(arr)

    ## Validate input
    #if not np.all(np.logical_and(arr >= -2, arr <= 2)):
    #    raise ValueError("Array entries must be integers in range [-2, 2]")

    # Shift values to be non-negative (map [-2,2] to [0,4])
    arr_shifted = arr + 2
    #tensor_data = torch.tensor(arr_shifted, dtype=torch.int64)
    tensor_bytes = np.asarray(arr_shifted[arr_shifted!=2],dtype=np.int8).tobytes()
    #print(tensor_bytes)
    compressed = zlib.compress(tensor_bytes)
    #print(compressed)
    if normalize:
        complexity = len(compressed) / np.log(len(arr) + 1)
    else:
        complexity = len(compressed)

    return complexity-baseline


