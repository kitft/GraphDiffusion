import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from contextlib import nullcontext
from IPython.display import clear_output
import torch
import wandb
from torch_AC import *

apply_all_moves_to_all_states_torch_jit = torch.jit.script(apply_all_moves_to_all_states_torch)
from typing import Callable
# Define a custom loss function for our model f_theta(x,g,t)
def custom_loss_discrete(f_theta_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor, t: torch.Tensor, all_next_moves: torch.Tensor or None, goal_state: torch.Tensor) -> torch.Tensor:
    device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
    # Ensure inputs are on the correct device
    x = x.to(device)
    #g = g.to(device)
    t = t.to(device)

    batch_size, num_states_in_sequence, _ = x.shape
    num_steps = num_states_in_sequence-1
    
    # Initialize total loss
    total_loss = 0.0
    # Vectorize the computation for all steps
    #current_x = x#[:, :, :]  # (batch_size, num_steps, state_dim)
    current_y = x[:,0:-1,:]
    



    current_x = x[:,1:,:]

    current_t = t#[:]  # (batch_size, num_steps)
    
    state_dim = current_x.shape[-1]
    moves_dim = 12#env.num_moves
    assert current_x.shape == (batch_size, num_steps, state_dim), f"Expected shape (batch_size, num_steps, state_dim), but got {current_x.shape}"
    assert current_t.shape == (batch_size, num_steps), f"Expected shape (batch_size, num_steps), but got {current_t.shape}"
    # Add a random amount between 0 and 1/num_steps to each value of current_t
    #random_addition = torch.rand(batch_size, num_steps, device=device) / num_steps
    current_t = current_t# + random_addition
    current_t_for_m1 = current_t# - 1/num_steps

    #solved_state = torch.tensor(goal_state).unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1).to(device)
    #current_y = torch.cat((solved_state,current_x),dim=1)
    #current_y = current_y[:,0:-1,:]# remove the "init state" state, not needed for ys

    
    scores = f_theta_forward(current_x.reshape(batch_size*num_steps, state_dim), current_t.reshape(batch_size*num_steps)).reshape(batch_size, num_steps, moves_dim)

    # Compute next states for all steps
    #next_ys_from_all_g = env.apply_all_moves_to_all_states(current_y.reshape(batch_size*num_steps, state_dim).cpu().numpy()).reshape(batch_size*num_steps*moves_dim, state_dim)#.to(device)#currently (batch*num_steps, moves_dim, state_dim)
    #next_ys_from_all_g = torch.tensor(env.apply_all_moves_to_all_states(current_y.reshape(batch_size*num_steps, state_dim).cpu().numpy()).reshape(batch_size*num_steps*moves_dim, state_dim),dtype=torch.int8,device=device)#.to(device)#currently (batch*num_steps, moves_dim, state_dim)
    #next_ys_from_all_g = env.apply_all_moves_to_all_states_torch(current_y.reshape(batch_size*num_steps, state_dim)).reshape(batch_size*num_steps*moves_dim,state_dim)
    current_y = current_y.reshape(batch_size*num_steps, state_dim)
    ########next_ys_from_all_g = 
    
    if all_next_moves is not None:
        all_next_moves = all_next_moves[:,0:-1]#remove the current_x last step
    else:
        all_next_moves = apply_all_moves_to_all_states_torch_jit(current_y)
    #.reshape(batch_size*num_steps*moves_dim,state_dim)
    # Create mask that is 0 if next state equals current state
    # Reshape current_y to match next_ys_from_all_g for comparison
    # Create mask - 1 where states are different, 0 where they are the same
    # Compare each batch*num_steps element with its corresponding moves
    current_y_flat = current_y.unsqueeze(1)
    ######mask = (all_next_moves != current_y_flat.repeat(1,moves_dim,1)).any(dim=2).float()
    mask = (all_next_moves.reshape(batch_size*num_steps,moves_dim,state_dim) != current_y_flat.repeat(1,moves_dim,1)).any(dim=2).float()
    mask = mask.reshape(batch_size, num_steps, moves_dim)
    next_ys_from_all_g = all_next_moves.reshape(batch_size*num_steps*moves_dim, state_dim)
    
    #next_ys_from_all_g = apply_all_moves_to_all_states(current_y.reshape(batch_size*num_steps, state_dim),STICKER_SOURCE_IX,STICKER_TARGET_IX).reshape(batch_size*num_steps*moves_dim, state_dim)#currently (batch*num_steps, moves_dim, state_dim)
    #.reshape(batch_size, num_steps, moves_dim, state_dim)
    # Adjust current_t for the next states
    # Repeat current_t for each batch and move, but keep it constant within each step
    current_t_stretch = current_t_for_m1.unsqueeze(-1).repeat(1,1,moves_dim)# repeat moves_dim along last dimension, shape (batch_size, num_steps, num_moves)
    #print("current_t_stretch",current_t_stretch.shape)
    current_t_stretch = current_t_stretch.reshape(batch_size*num_steps*moves_dim)
    
    inverse_scores = f_theta_forward(next_ys_from_all_g,current_t_stretch).reshape(batch_size, num_steps, moves_dim, moves_dim)#second is moves choice, i.e. output choice.
 
    range_moves = torch.arange(moves_dim, device=inverse_scores.device)
    #inverse_moves = torch.tensor(env.inverse_moves, device=inverse_scores.device)
    inverse_moves = torch.tensor([2,3,0,1,8,9,10,11,4,5,6,7], device=inverse_scores.device)
    # Apply the permutation to the last two dimensions of inverse_scores
    inverse_scores = inverse_scores[..., range_moves, inverse_moves]
    #so sum_g f_theta(x)[g] - log(f_theta[g.x][g^{-1})

    # Calculate the loss for all steps
    #scores has indices [batch, steps, moves_g]
    step_losses = torch.sum((scores - torch.log(inverse_scores))*mask, dim=(2))#sum over the moves dimension
    # Compute total loss
    total_loss = step_losses.mean() #then mean over steps and batch!

    # Return average loss over all steps
    return total_loss #/ (num_steps)

from pytorch_optimizer import SOAP


def train(model, dataloader, val_dataloader,max_plot_val=30,warmup_frac=0.1):

    name_time = str(int(time.time()))
    name_for_saving_losses = TrainConfig.name+"training_losses"+name_time+ "_"+model.model_type
    name_for_saving_model = TrainConfig.name + "_"+ model.model_type + f"_{name_time}_"
    save_path = os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_losses)
    # Count number of parameters in millions and format as string
    num_params = sum(p.numel() for p in model.parameters())
    params_millions = round(num_params / 1_000_000)
    params_str = f"{params_millions}M"
    name_for_saving_model += f"_{params_str}"

    return_apply_all = dataloader.return_apply_all
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    wandb.init(
    # set the wandb project where this run will be logged
    project="DiscreteDiffusion",
    name=name_for_saving_model +" "+name_time+f"_{params_str}",
    # track hyperparameters and run metadata
    config={
    "architecture": model.model_type,
    "dataset": env.name,
    "max_depth": TrainConfig.max_depth,
    "gradient_accumulation_steps": TrainConfig.gradient_accumulation_steps,
    "learning_rate": TrainConfig.learning_rate,
    "epochs": TrainConfig.num_steps,
    "batch_size_per_depth": TrainConfig.batch_size_per_depth,
    "num_steps": TrainConfig.num_steps,
    "learning_rate": TrainConfig.learning_rate,
    "weight_decay": TrainConfig.weight_decay,
    "max_grad_norm": TrainConfig.max_grad_norm,
    "INTERVAL_VALIDATE": TrainConfig.INTERVAL_VALIDATE,
    "ENABLE_FP16": TrainConfig.ENABLE_FP16,
    "gradient_accumulation_steps": TrainConfig.gradient_accumulation_steps,
    "return_apply_all_at_generation_time": return_apply_all,
    "total_params": sum(p.numel() for p in model.parameters()),
    }
    )
    
    #writer = SummaryWriter(os.path.join(TrainConfig.SAVE_DIRECTORY, 'runs', TrainConfig.name + '_' + name_time))

    model.train()
    base_learning_rate = TrainConfig.learning_rate
    max_grad_norm = TrainConfig.max_grad_norm#1.0  # You can adjust this value as needed
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    #optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate, weight_decay=TrainConfig.weight_decay)
    optimizer = SOAP(model.parameters(), lr=base_learning_rate, weight_decay=TrainConfig.weight_decay)
    g = iter(dataloader)
    iter_val = iter(val_dataloader)

    train_losses = []
    val_losses = []
    train_losses_aux = []

    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if TrainConfig.ENABLE_FP16 else nullcontext()
    pbar = trange(TrainConfig.num_steps)
    warmup_steps = int(warmup_frac * TrainConfig.num_steps)  # 10% of total steps for warmup
    try:
        for i in pbar:
            # Linear warmup schedule
            if i <= warmup_steps:
                learning_rate = base_learning_rate * (i / warmup_steps)
            else:
                learning_rate = base_learning_rate
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            if return_apply_all:
                batch_x, _, all_next_moves = next(g)
                all_next_moves.to(device)
            else:
                batch_x, _,_ = next(g)
                all_next_moves = None
         
            
            batch_x = batch_x.to(device) # this has shape (batch, num_steps, state_dim)
            depth_trajectory = batch_x.shape[1]-1#REMOVE INITIAL STATE
            batch_t = torch.linspace(0+1/depth_trajectory,1, steps=depth_trajectory).to(device)
            batch_t = batch_t.unsqueeze(0).repeat(batch_x.shape[0], 1)

            if i % TrainConfig.INTERVAL_VALIDATE == 0:
                model.eval()
                with torch.no_grad():
                    if return_apply_all:
                        val_batch_x, _,val_all_next_moves = next(iter_val)
                        val_all_next_moves.to(device)
                    else:
                        val_batch_x, _ ,_= next(iter_val)
                        val_all_next_moves = None
                
                    val_batch_x = val_batch_x.to(device)

                    val_batch_t = torch.linspace(0+1/depth_trajectory,1, steps=depth_trajectory).to(device)
                    val_batch_t = val_batch_t.unsqueeze(0).repeat(val_batch_x.shape[0], 1)

                    val_loss = custom_loss_discrete(model, val_batch_x, val_batch_t,val_all_next_moves,GOAL_STATE)

                    val_losses.append(val_loss.item())
                    #writer.add_scalar('Loss/val', val_losses[-1], i)
                    #writer.add_scalar('Loss/val_accumulated_average', np.mean(val_losses[-TrainConfig.gradient_accumulation_steps:]), i)
                    wandb.log({"val_loss": val_losses[-1], "val_loss_accumulated_average": np.mean(val_losses[-TrainConfig.gradient_accumulation_steps:])}, step=i)
                model.train()

            with ctx:
                # Calculate the custom loss
                #custom_loss_value = custom_loss_concrete_matching(model, batch_x, batch_t)
                actual_loss = custom_loss_discrete(model, batch_x, batch_t,all_next_moves,GOAL_STATE)
                loss = actual_loss /TrainConfig.gradient_accumulation_steps
            
            if torch.isnan(loss):
                print("Loss became NaN. Reducing learning rate and resetting model parameters.")
                base_learning_rate *= 0.5
                model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                optimizer = SOAP(model.parameters(), lr=base_learning_rate, weight_decay=TrainConfig.weight_decay)
                g = iter(dataloader)
                train_losses = []
                val_losses = []
                train_losses_aux = []
                continue

            # Gradient accumulation logic
            loss.backward()
            
            if (i + 1) % TrainConfig.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * TrainConfig.gradient_accumulation_steps)  # Scale back up for logging
            train_losses_aux.append(actual_loss.item())
        
            #writer.add_scalar('Loss/train', train_losses[-1], i)
            #writer.add_scalar('Loss/train_accumulated_average', np.mean(train_losses[-TrainConfig.gradient_accumulation_steps:]), i)
            #writer.add_scalar('Learning_rate', learning_rate, i)
            wandb.log({"train_loss": train_losses[-1], "train_loss_accumulated_average": np.mean(train_losses[-TrainConfig.gradient_accumulation_steps:]), "learning_rate": learning_rate}, step=i)
            
            if TrainConfig.INTERVAL_PLOT and i % TrainConfig.INTERVAL_PLOT == 0:
                
                # Save losses to a file using numpy's savez which is more efficient and reliable
                losses_dict = {
                    'train_losses': np.array(train_losses),
                    'val_losses': np.array(val_losses),
                    'train_losses_aux': np.array(train_losses_aux)  # Note: Changed from train_losses_main to match variable name
                }

                clear_output(wait=True)
                plot_loss_curves(losses_dict,maxval=max_plot_val)
                pbar.refresh()  # Redraw the progress bar

                np.savez(os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_losses), **losses_dict)
            pbar.set_description(f"step: {i}, lr: {learning_rate:.10f} Loss: {actual_loss.item():.4f}, val_loss: {0 if len(val_losses)==0 else val_losses[-1]:.4f}")

            if TrainConfig.INTERVAL_SAVE and i % TrainConfig.INTERVAL_SAVE == 0 and i>0:
                torch.save(model.state_dict(), os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_model+f"{i}steps.pth"))
                #torch.save(model.state_dict(), os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_model+f"{i}steps.pth"))
                print("Model saved.")
            if TrainConfig.INTERVAL_BACKUP and i % TrainConfig.INTERVAL_BACKUP == 0 and i>0:
                torch.save(model.state_dict(), os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_model+f"_most_recent.pth"))
                #torch.save(model.state_dict(), os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_model+f"{i}steps.pth"))
                print("Model backed up.")
    except KeyboardInterrupt or Exception as e:
        print("Training interrupted. Returning current model and histories and saving losses.")
        print(e if isinstance(e,Exception) else "KeyboardInterrupt")
        # Save losses in a more structured format using numpy
        losses_dict = {
            'train_losses': np.array(train_losses),
            'val_losses': np.array(val_losses), 
            'train_losses_aux': np.array(train_losses_aux)
        }
        np.savez(save_path, **losses_dict)
        if len(train_losses) < 5:
            wandb.run.tags = wandb.run.tags + ("too_short",)
        else:   
            final_save_location = os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_model+f"_{TrainConfig.num_steps}steps_"+name_time)
            print("Saving final model to ", final_save_location+".pth")
            torch.save(model.state_dict(), final_save_location+".pth")
        wandb.finish()
        return model, losses_dict
    print(f"Trained on data equivalent to {TrainConfig.batch_size_per_depth * TrainConfig.num_steps} solves.")
    losses_dict = {
            'train_losses': np.array(train_losses),
            'val_losses': np.array(val_losses), 
            'train_losses_aux': np.array(train_losses_aux)
        }
    print("saving to ",save_path)
    np.savez(save_path, **losses_dict)
    final_save_location = os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_model+f"_{TrainConfig.num_steps}steps_"+name_time)
    print("Saving final model to ", final_save_location+".pth")
    torch.save(model.state_dict(), final_save_location+".pth")
    wandb.finish()
    return model, losses_dict


def plot_loss_curves(losses_dict,minval=None,maxval=None):
    train_losses = losses_dict['train_losses']
    train_losses_aux = losses_dict['train_losses_aux']
    val_losses = losses_dict['val_losses']
    
    # Create figure with one or two panels based on gradient accumulation
    if TrainConfig.gradient_accumulation_steps > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
    def format_axis(ax, title):
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss") 
        ax.set_title(title)
        ax.set_xscale("log")
        if minval is not None and maxval is not None:
            ax.set_ylim(minval,maxval)
        elif minval is not None:
            ax.set_ylim(minval,None)
        elif maxval is not None:
            ax.set_ylim(max(min(min(train_losses), min(val_losses),min(train_losses_aux)) - 1,-100),maxval)
        else:
            ax.set_ylim(max(min(min(train_losses), min(val_losses),min(train_losses_aux)) - 1,-10), 30)
        ax.legend()
    
    # Plot raw losses on left panel
    val_steps = list(range(0, len(train_losses), TrainConfig.INTERVAL_VALIDATE))
    val_steps[0] = 1  # move to 1
    ax1.plot(train_losses, label='Training Loss', color='green')
    ax1.plot(val_steps, val_losses, label='Validation Loss', color='red')
    format_axis(ax1, "Raw Training and Validation Loss")
    
    # Plot moving averages on right panel if gradient accumulation enabled
    if TrainConfig.gradient_accumulation_steps > 1 and len(train_losses) > TrainConfig.gradient_accumulation_steps *TrainConfig.INTERVAL_VALIDATE:
        window = TrainConfig.gradient_accumulation_steps
        train_avg = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_steps_avg = val_steps[window-1:]
        val_avg = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        
        ax2.plot(range(window-1, len(train_losses)), train_avg,
                label='Training Loss (Averaged)', color='green')
        ax2.plot(val_steps_avg, val_avg,
                label='Validation Loss (Averaged)', color='red')
        format_axis(ax2, "Moving Average Loss")
    
    plt.tight_layout()
    plt.show()