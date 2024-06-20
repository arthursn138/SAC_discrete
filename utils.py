import torch
from params import *

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def save_adapted(save_dir, model, wandb, ep=None):
    import os
    if not os.path.exists(os.path.join(save_dir, 'SAC_discrete')):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), os.path.join(save_dir, 'SAC_discrete') + str(ep) + ".pth")
        wandb.save(os.path.join(save_dir, 'SAC_discrete') + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, 'SAC_discrete') + ".pth")
        wandb.save(save_dir + os.path.join(save_dir, 'SAC_discrete') + ".pth")

def collect_random(env, dataset, num_samples=200):
    state, *_ = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, *_ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
