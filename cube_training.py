
import os
import time
import numpy as np
import torch
from tqdm import trange
from contextlib import nullcontext
from pytorch_optimizer import SOAP
import wandb
from envsAndScramble import *
from NN_models import *
import matplotlib.pyplot as plt
from IPython.display import clear_output



"""
Training functions for cube models.

Note: Requires TrainConfig to be defined with the following attributes:
    - gradient_accumulation_steps: int
    - INTERVAL_PLOT: int
    - INTERVAL_VALIDATE: int
    - INTERVAL_SAVE: int
    - INTERVAL_BACKUP: int
    - SAVE_DIRECTORY: str
    - num_steps: int
    - batch_size_per_depth: int
    - dropout: float
"""



# Define a custom loss function for our model f_theta(x,g,t)
def custom_loss_discrete(f_theta: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes the custom loss for discrete score matching on cube states.

    Args:
        f_theta: Neural network model that outputs scores for each possible move
        x: Tensor of shape (batch_size, num_steps, state_dim) containing sequences of cube states
        t: Tensor of shape (batch_size, num_steps) containing time values for each state

    Returns:
        total_loss: Scalar tensor containing the computed loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Ensure inputs are on the correct device
    x = x.to(device)
    #g = g.to(device)
    t = t.to(device)

    batch_size, num_steps, _ = x.shape
    
    # Initialize total loss
    total_loss = 0.0
    # Vectorize the computation for all steps
    current_x = x#[:, :, :]  # (batch_size, num_steps, state_dim)
    current_t = t#[:]  # (batch_size, num_steps)
    state_dim = current_x.shape[-1]
    moves_dim = env.num_moves
    assert current_x.shape == (batch_size, num_steps, state_dim), f"Expected shape (batch_size, num_steps, state_dim), but got {current_x.shape}"
    assert current_t.shape == (batch_size, num_steps), f"Expected shape (batch_size, num_steps), but got {current_t.shape}"
    # Add a random amount between 0 and 1/num_steps to each value of current_t
    #random_addition = torch.rand(batch_size, num_steps, device=device) / num_steps
    current_t = current_t.float().view(batch_size*num_steps)# + random_addition
    #current_t_for_m1 = current_t# - 1/num_steps

    solved_state = torch.tensor(env.goal).unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1).to(device)
    current_y = torch.cat((solved_state,current_x),dim=1)[:,:-1]
    #current_y = current_y[:,0:-1,:]# remove the "init state" state, not necessary


    scores = f_theta(current_x.reshape(batch_size*num_steps, state_dim), current_t)#.reshape(batch_size, num_steps, moves_dim)

    # Compute next states for all steps
    next_ys_from_all_g = apply_all_moves_to_all_states(current_y.reshape(batch_size*num_steps, state_dim),STICKER_SOURCE_IX,STICKER_TARGET_IX).view(batch_size*num_steps*moves_dim, state_dim)#currently (batch*num_steps, moves_dim, state_dim)
    #next_ys_from_all_g = torch.zeros(batch_size*num_steps*moves_dim, state_dim,device=device,dtype=current_x.dtype)
    #next_ys_from_all_g = apply_all_moves_to_all_states_no_reshape(current_y.reshape(batch_size*num_steps, state_dim),next_ys_from_all_g,STICKER_SOURCE_IX,STICKER_TARGET_IX)

    #.reshape(batch_size, num_steps, moves_dim, state_dim)
    # Adjust current_t for the next states
    # Repeat current_t for each batch and move, but keep it constant within each step
    #current_t_stretch = current_t_for_m1.unsqueeze(-1).repeat(1,1,moves_dim)# repeat moves_dim along last dimension, shape (batch_size, num_steps, num_moves)
    #current_t_stretch = current_t_stretch.reshape(batch_size*num_steps*moves_dim)
    current_t_stretch = current_t.repeat_interleave(moves_dim)

    inverse_scores = f_theta(next_ys_from_all_g,current_t_stretch).view(batch_size*num_steps, moves_dim, moves_dim)#second is moves choice, i.e. output choice.
    #print("isc: ",inverse_scores[0])

    # This should invert each move for an arbitrary moves_dim
    #permutation = torch.arange(moves_dim, device=inverse_scores.device).repeat_interleave(2).reshape(-1, 4)
    #permutation[:, 1:3] = permutation[:, 1:3].flip(1)
    #permutation[:,2:4]= permutation[:,2:4].flip(1)
    #permutation = permutation.reshape(-1,2)
    #permutation = torch.tensor([[1,0],[0,1],[3,2],[2,3],[5,4],[4,5],[7,6],[6,7],[9,8],[8,9],[11,10],[10,11]])
    permutation = torch.tensor([[0,1],[1,0],[2,3],[3,2],[4,5],[5,4],[6,7],[7,6],[8,9],[9,8],[10,11],[11,10]])
    # Apply to the last two dimensions of inverse_scores
    inverse_scores = inverse_scores[..., permutation[:, 0], permutation[:, 1]]
    #so sum_g f_theta(x)[g] - log(f_theta[g.x][g^{-1})

    # Calculate the loss for all steps
    #scores has indices [batch, steps, moves_g]
    #inverse_scores
    step_losses = torch.sum(scores - torch.log(inverse_scores), dim=(-1))#sum over the moves dimension
    #print("isc",inverse_scores[0])
    #print("sc",scores[0])
    # Compute total loss
    total_loss = step_losses.mean() #then mean over steps and batch!

    # Return average loss over all steps
    return total_loss #/ (num_steps)





def train(model, dataloader, val_dataloader, max_plot_val=30, warmup_frac=0.1, resume_id=None, init_step=0, plot_fn=plot_loss_curves):
    """
    Train a model using the provided dataloaders.

    Args:
        model: The neural network model to train
        dataloader: DataLoader containing training data
        val_dataloader: DataLoader containing validation data 
        max_plot_val: Maximum number of points to plot in loss curves (default: 30)
        warmup_frac: Fraction of total steps to use for learning rate warmup (default: 0.1)
        resume_id: Optional wandb run ID to resume training from (default: None)
        init_step: Initial step number when resuming training (default: 0)
        plot_fn: Function to plot training curves (default: plot_loss_curves)

    Returns:
        None. Model is trained in-place and metrics are logged to wandb.
    """
    if plot_fn is None:
        plot_fn = lambda *args, **kwargs: None  # Do nothing
    name_time = str(int(time.time()))
    name_for_saving_losses = TrainConfig.name+"training_losses"+name_time+ "_"+model.model_type
    name_for_saving_model = TrainConfig.name + "_"+ model.model_type
    save_path = os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_losses)
    # Count number of parameters in millions and format as string
    num_params = sum(p.numel() for p in model.parameters())
    params_millions = round(num_params / 1_000_000)
    params_str = f"{params_millions}M"
    name_for_saving_model += f"_{params_str}"
    wandb.init(
    # set the wandb project where this run will be logged
    project="DiscreteDiffusion",
    name=None if resume_id else name_for_saving_model +" "+name_time+f"_{params_str}",
    id=resume_id if resume_id else None,
    resume="must" if resume_id else None,
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
    #"use_torch_dataloader": use_torch_dataloader,
    "total_params": sum(p.numel() for p in model.parameters()),
    }
    )
    
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    pbar = trange(init_step,TrainConfig.num_steps)
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

            batch_x, _ = next(g)
            depth_trajectory = batch_x.shape[1]
            batch_x = batch_x.to(device) # this has shape (batch, num_steps, state_dim)
            
            batch_t = torch.linspace(0+1/depth_trajectory,1, steps=depth_trajectory).to(device)
            batch_t = batch_t.unsqueeze(0).repeat(batch_x.shape[0], 1)

            if i % TrainConfig.INTERVAL_VALIDATE == 0:
                model.eval()
                with torch.no_grad():
                    val_batch_x, _ = next(iter_val)
                    val_batch_x = val_batch_x.to(device)

                    val_batch_t = torch.linspace(0+1/depth_trajectory,1, steps=depth_trajectory).to(device)
                    val_batch_t = val_batch_t.unsqueeze(0).repeat(val_batch_x.shape[0], 1)

                    val_loss = custom_loss_discrete(model, val_batch_x, val_batch_t)

                    val_losses.append(val_loss.item())
                    #writer.add_scalar('Loss/val', val_losses[-1], i)
                    #writer.add_scalar('Loss/val_accumulated_average', np.mean(val_losses[-TrainConfig.gradient_accumulation_steps:]), i)
                    wandb.log({"val_loss": val_losses[-1], "val_loss_accumulated_average": np.mean(val_losses[-TrainConfig.gradient_accumulation_steps:])}, step=i)
                model.train()

            with ctx:
                # Calculate the custom loss
                #custom_loss_value = custom_loss_concrete_matching(model, batch_x, batch_t)
                actual_loss = custom_loss_discrete(model, batch_x, batch_t)
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
                plot_fn(losses_dict,maxval=max_plot_val)
                pbar.refresh()  # Redraw the progress bar

                np.savez(os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_losses), **losses_dict)
            pbar.set_description(f"step: {i}, lr: {learning_rate:.10f} Loss: {actual_loss.item():.4f}, val_loss: {0 if len(val_losses)==0 else val_losses[-1]:.4f}")

            if TrainConfig.INTERVAL_SAVE and i % TrainConfig.INTERVAL_SAVE == 0:
                torch.save(model.state_dict(), os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_model+f"{i}steps.pth"))
                print("Model saved.")
            if TrainConfig.INTERVAL_BACKUP and i % TrainConfig.INTERVAL_BACKUP == 0:
                torch.save(model.state_dict(), os.path.join(TrainConfig.SAVE_DIRECTORY, name_for_saving_model+f"{name_time}_backup.pth"))
                print("Model saved.")
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

def plot_loss_curves(losses_dict, minval=None, maxval=None):
    """
    Plot training and validation loss curves.

    Args:
        losses_dict: Dictionary containing training and validation losses with keys:
            - 'train_losses': Array of training losses
            - 'train_losses_aux': Array of auxiliary training losses 
            - 'val_losses': Array of validation losses
        minval: Optional minimum y-axis value for loss plots
        maxval: Optional maximum y-axis value for loss plots

    Returns:
        None. Displays loss curve plots using matplotlib.
    """
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