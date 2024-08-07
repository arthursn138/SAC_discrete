

import gym
# import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from SAC_discrete.buffer import ReplayBuffer
import glob
from SAC_discrete.utils import save_adapted, collect_random
import random
from SAC_discrete.agent import SAC
from params import *

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="SAC", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    
    args = parser.parse_args()
    return args

# def train(config):
    # # Seed setting occurs in main.py, environment is already set
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # env = gym.make(config.env)
    #
    # env.seed(config.seed)
    # env.action_space.seed(config.seed)
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(env, run_name, buffer_size=100_000, batch_size=256, episodes=EPISODES, save_every=1000, agent=None, start_step=0):
    
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    
    with wandb.init(project="SAC_Discrete", name=run_name):
        if agent is None:
            agent = SAC(state_size=env.observation_space.shape[0],
                        action_size=env.action_space.n,
                        device=device)

        wandb.watch(agent, log="gradients", log_freq=10)

        buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=device)
        
        collect_random(env=env, dataset=buffer, num_samples=10000)
        
        # if config.log_video:
        #     env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        for i in range(1, episodes+1):
            state, *_ = env.reset()
            episode_steps = 0
            rewards = 0
            hidden = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))   # matches hidden_size in agent.py

            while True:
                if not USE_LSTM:
                    action = agent.get_action(state)
                else:
                    action, hidden = agent.get_action_lstm(state, hidden)
                steps += 1
                # print('action:', action)
                next_state, reward, done, *_ = env.step(action)
                # print('')
                # print(np.shape(state))
                # print(np.shape(action))
                # print(np.shape(next_state))
                # print('')
                buffer.add(state, action, reward, next_state, done)
                if not USE_LSTM:
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(
                        steps, buffer.sample(), gamma=0.99)
                else:
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha, *_ = agent.learn_lstm(
                        steps, buffer.sample(), gamma=0.99, hidden=hidden)

                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    break

            

            average10.append(rewards)
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
            
            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Steps": total_steps,
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Bellmann error 1": bellmann_error1,
                       "Bellmann error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Steps": steps,
                       "Episode": i,
                       "Buffer size": buffer.__len__()})

            # if (i %10 == 0) and config.log_video:
            #     mp4list = glob.glob('video/*.mp4')
            #     if len(mp4list) > 1:
            #         mp4 = mp4list[-2]
            #         wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            # ## From SB3's SAC: # To implement
            # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            # self.logger.record("train/ent_coef", np.mean(ent_coefs))
            # self.logger.record("train/actor_loss", np.mean(actor_losses))
            # self.logger.record("train/critic_loss", np.mean(critic_losses))
            # if len(ent_coef_losses) > 0:
            #     self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

            if i % save_every == 0:
                save_adapted(save_dir=run_name, model=agent.actor_local, wandb=wandb, ep=i)

        save_adapted(save_dir=run_name, model=agent.actor_local, wandb=wandb, ep=None)

# if __name__ == "__main__":
#     config = get_config()
#     train(config)
