import argparse
import os
import random
import time
from distutils.util import strtobool
import math
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import envpool
import wandb

# ============================================================================
# [新功能] WandB 查重工具
# ============================================================================
def get_run_info(args):
    """
    生成 Run Name
    """
    project_name = f"Atari_{args.game_name}_v5_G2"

    if args.algo == "ANO":
        run_name = f"{args.algo}_{args.epsilons}_{args.seed}"
        group_name = f"{args.algo}_{args.epsilons}"
    elif args.algo == "TRPO":
        run_name = f"{args.algo}_{args.trpo_max_kl}_{args.seed}"
        group_name = f"{args.algo}_{args.trpo_max_kl}"
    elif args.algo == "PAPO":
        run_name = f"{args.algo}_{args.papo_omega1}_{args.papo_omega2}_{args.seed}"
        group_name = f"{args.algo}_{args.papo_omega1}_{args.papo_omega2}"
    else: # PPO / SPO
        run_name = f"{args.algo}_{args.clip_coef}_{args.seed}"
        group_name = f"{args.algo}_{args.clip_coef}"
        
    return project_name, group_name, run_name

def check_wandb_run_exists(entity, project, group, name):
    try:
        api = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, filters={"group": group, "display_name": name})
        if len(runs) > 0:
            print(f"⚠️  [Skip] Found existing run on WandB: {project}/{group}/{name}")
            return True
        return False
    except Exception as e:
        print(f"⚠️  [WandB Check Error] {e} -> Proceeding...")
        return False

# ============================================================================
# [Wrapper 1] 原版 Wrapper (用于除 Atlantis 外的所有游戏)
# ============================================================================
class AtariScoreWrapper_Original(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
        elif hasattr(env, "config") and "num_envs" in env.config:
            self.num_envs = env.config["num_envs"]
        else:
            raise AttributeError("Cannot find 'num_envs' in environment.")
            
        self.buffers = {"return": np.zeros(self.num_envs), "length": np.zeros(self.num_envs)}

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        done = term | trunc
        
        self.buffers["return"] += reward
        self.buffers["length"] += 1

        info["episode"] = {"r": [], "l": []}

        for i in range(self.num_envs):
            if done[i]:
                lives = info["lives"][i] if "lives" in info else 0
                if lives == 0:
                    info["episode"]["r"].append(self.buffers["return"][i])
                    info["episode"]["l"].append(self.buffers["length"][i])
                    self.buffers["return"][i] = 0
                    self.buffers["length"][i] = 0
                else:
                    pass

        return obs, reward, term, trunc, info

# ============================================================================
# [Wrapper 2] 修正版 Wrapper (仅用于 Atlantis)
# ============================================================================
class AtariScoreWrapper_Fixed(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
        elif hasattr(env, "config") and "num_envs" in env.config:
            self.num_envs = env.config["num_envs"]
        else:
            raise AttributeError("Cannot find 'num_envs' in environment.")
            
        self.buffers = {"return": np.zeros(self.num_envs), "length": np.zeros(self.num_envs)}

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        done = term | trunc
        
        self.buffers["return"] += reward
        self.buffers["length"] += 1

        info["episode"] = {"r": [], "l": []}
        
        has_lives = "lives" in info

        for i in range(self.num_envs):
            if done[i]:
                is_game_over = False
                if has_lives:
                    lives = info["lives"][i]
                    if lives == 0:
                        is_game_over = True
                else:
                    is_game_over = True
                
                if trunc[i]:
                    is_game_over = True

                if is_game_over:
                    info["episode"]["r"].append(self.buffers["return"][i])
                    info["episode"]["l"].append(self.buffers["length"][i])
                    self.buffers["return"][i] = 0
                    self.buffers["length"][i] = 0
                else:
                    pass

        return obs, reward, term, trunc, info

# ============================================================================
# ANO / TRPO / PAPO 核心数学工具
# ============================================================================
def f0(x):
    return 45/16 * (0.5 * torch.nn.functional.logsigmoid(2*x) - 2 * torch.sigmoid(x))

def f_func(x, d, device, epsilons):
    ks = [epsilons[0]/math.log(2), epsilons[1]/math.log(2)]
    bs = [1+epsilons[0], 1+epsilons[1]]
    val_1 = torch.tensor(1.0, device=device)
    term_d = 1/ks[d] * (val_1 - bs[d])
    ds_d = ks[d] * f0(term_d)
    term_x = (x - bs[d]) / ks[d]
    f1_x = ks[d] * f0(term_x)
    return f1_x + 1 - ds_d

# [关键修复] 使用 .reshape(-1) 替代 .view(-1) 以支持非连续 Tensor
def flat_grad(grads, params):
    grad_flatten = []
    for grad in grads:
        if grad is None:
            continue
        # Fix: view size is not compatible with input tensor's size and stride
        grad_flatten.append(grad.reshape(-1)) 
    return torch.cat(grad_flatten)

# [关键修复] 同理，flat_params 也建议用 reshape
def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.reshape(-1))
    return torch.cat(params)

def set_params(model, new_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            new_params[prev_ind:prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size

def get_kl_discrete(model, x, old_logits):
    """Calculate Analytical KL Divergence for Categorical Distribution"""
    new_logits = model.actor(model.network(x / 255.0))
    new_probs = torch.softmax(new_logits, dim=-1)
    old_probs = torch.softmax(old_logits, dim=-1)
    kl = (old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10))).sum(dim=-1, keepdim=True)
    return kl

def conjugate_gradient(fvp_func, b, cg_iters=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
        if rdotr < residual_tol: break
        z = fvp_func(p)
        alpha = rdotr / (torch.dot(p, z) + 1e-8)
        x += alpha * p
        r -= alpha * z
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

# ============================================================================
# 网络定义 (Standard Atari CNN)
# ============================================================================
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(512, envs.single_action_space.n),
        )
        self.critic = nn.Sequential(
            nn.Linear(512, 1),
        )

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def get_logits(self, x):
        return self.actor(self.network(x / 255.0))

# ============================================================================
# 单次训练流程 (Training Loop)
# ============================================================================
def train_one_game(args):
    project_name, group_name, run_name = get_run_info(args)
        
    print(f"--- Starting: {run_name} on {args.env_id} ---")

    run = wandb.init(
        project=project_name,
        group=group_name,
        name=run_name,
        config=vars(args),
        monitor_gym=False,
        save_code=True,
        reinit=True,
        sync_tensorboard=True,
    )
    writer = SummaryWriter(f"runs/{args.env_id}/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ==============================================================
    # [关键分支] 仅针对 Atlantis 启用特殊修复逻辑
    # ==============================================================
    if "Atlantis" in args.game_name:
        print(">>> Using FIXED Atlantis Logic (Max Steps + Fixed Wrapper) <<<")
        envs = envpool.make(
            args.env_id,
            env_type="gymnasium",
            num_envs=args.num_envs,
            noop_max=30,
            frame_skip=4,
            img_height=84,
            img_width=84,
            stack_num=4,
            gray_scale=True,
            episodic_life=True,
            reward_clip=False, 
            max_episode_steps=27000, 
            seed=args.seed,
        )
        envs = AtariScoreWrapper_Fixed(envs) 
    else:
        # print(">>> Using ORIGINAL Logic (Standard Wrapper) <<<")
        envs = envpool.make(
            args.env_id,
            env_type="gymnasium",
            num_envs=args.num_envs,
            noop_max=30,
            frame_skip=4,
            img_height=84,
            img_width=84,
            stack_num=4,
            gray_scale=True,
            episodic_life=True,
            reward_clip=False, 
            seed=args.seed,
        )
        envs = AtariScoreWrapper_Original(envs) 
    # ==============================================================

    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    raw_advantages = torch.zeros((args.num_steps, args.num_envs)).to(device) 

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            
            if "episode" in info and len(info["episode"]["r"]) > 0:
                avg_ret = np.mean(info["episode"]["r"])
                avg_len = np.mean(info["episode"]["l"])
                writer.add_scalar("charts/episodic_return", avg_ret, global_step)
                writer.add_scalar("charts/episodic_length", avg_len, global_step)
                wandb.log({
                    "rollout/ep_rew_mean": avg_ret, 
                    "rollout/ep_len_mean": avg_len,
                    "global_step": global_step
                })

            clipped_reward = np.sign(reward)
            rewards[step] = torch.tensor(clipped_reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                raw_advantages[t] = delta
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_raw_advantages = raw_advantages.reshape(-1)

        # ====================================================================
        # [Branch] TRPO Logic
        # ====================================================================
        if args.algo == "TRPO":
            with torch.no_grad():
                old_logits = agent.get_logits(b_obs)
            
            policy_params = list(agent.network.parameters()) + list(agent.actor.parameters())
            
            logits = agent.get_logits(b_obs)
            probs = Categorical(logits=logits)
            new_log_probs = probs.log_prob(b_actions)
            ratio = torch.exp(new_log_probs - b_logprobs)
            surrogate_loss = (ratio * b_advantages).mean()

            grads = torch.autograd.grad(surrogate_loss, policy_params)
            g = flat_grad(grads, policy_params)

            def fvp_func(v):
                kl = get_kl_discrete(agent, b_obs, old_logits).mean()
                grads = torch.autograd.grad(kl, policy_params, create_graph=True)
                flat_grad_kl = flat_grad(grads, policy_params)
                kl_v = (flat_grad_kl * v).sum()
                grads_v = torch.autograd.grad(kl_v, policy_params, retain_graph=False)
                return flat_grad(grads_v, policy_params) + args.trpo_damping * v

            step_dir = conjugate_gradient(fvp_func, g, cg_iters=args.trpo_cg_iters)
            shs = 0.5 * (step_dir * fvp_func(step_dir)).sum(0, keepdim=True)
            lm = torch.sqrt(shs / args.trpo_max_kl)
            full_step = step_dir / lm[0]
            
            if torch.isnan(full_step).any():
                print("TRPO Warning: NaN in full_step")
                full_step = torch.zeros_like(full_step)

            current_params = torch.cat([flat_params(agent.network), flat_params(agent.actor)])
            
            success = False
            for i in range(args.trpo_ls_iters):
                step_size = args.trpo_ls_backtrack_ratio ** i
                proposed_step = full_step * step_size
                new_params = current_params + proposed_step
                
                net_size = sum(p.numel() for p in agent.network.parameters())
                set_params(agent.network, new_params[:net_size])
                set_params(agent.actor, new_params[net_size:])
                
                with torch.no_grad():
                    new_logits_eval = agent.get_logits(b_obs)
                    new_probs_eval = Categorical(logits=new_logits_eval)
                    new_log_probs_eval = new_probs_eval.log_prob(b_actions)
                    ratio_eval = torch.exp(new_log_probs_eval - b_logprobs)
                    new_surrogate_loss = (ratio_eval * b_advantages).mean()
                    kl_val = get_kl_discrete(agent, b_obs, old_logits).mean()

                # 在 Policy Update 的 Line Search 循环中：
                if new_surrogate_loss > surrogate_loss and kl_val <= args.trpo_max_kl:
                    success = True
                    break
            
            if not success:
                set_params(agent.network, current_params[:net_size])
                set_params(agent.actor, current_params[net_size:])

            # Value Function Update
            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                    if args.clip_vloss:
                         v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                         v_loss = 0.5 * torch.max((newvalue - b_returns[mb_inds]) ** 2, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                    else:
                         v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # === [修复开始] ===
                    optimizer.zero_grad()
                    (v_loss * args.vf_coef).backward()
                    
                    # 裁剪 Critic 的梯度 (可选，但推荐)
                    nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                    
                    # [删除] 绝对不要在这里调用 agent.network.zero_grad() !!!
                    # [删除] agent.network.zero_grad() 
                    # [删除] agent.actor.zero_grad()
                    
                    optimizer.step()
                    # === [修复结束] ===
            
            pg_loss, v_loss, entropy_loss, approx_kl, clipfracs = surrogate_loss, v_loss, torch.tensor(0.0), kl_val, [0.0]

        # ====================================================================
        # [Branch] PPO / ANO / SPO / PAPO Logic
        # ====================================================================
        else:
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            continue_training = True

            for epoch in range(args.update_epochs):
                if not continue_training: break

                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    if args.algo == "ANO":
                        x = ratio
                        term_p = f_func(x, 0, device, args.epsilons)
                        term_n = 2.0 - f_func(2.0 - x, 1, device, args.epsilons)
                        pg_loss = -torch.min(mb_advantages * term_p, mb_advantages * term_n).mean()
                    
                    elif args.algo == "SPO":
                        pg_loss = -(mb_advantages * ratio - torch.abs(mb_advantages) * torch.pow(ratio - 1, 2) / (2 * args.clip_coef)).mean()
                    
                    elif args.algo == "PAPO":
                        clipped_ratio = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        mean_surr = torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                        
                        raw_adv = b_raw_advantages[mb_inds]
                        tmp_1 = (ratio - 1) * raw_adv**2
                        tmp_2 = 2 * ratio * raw_adv
                        clip_tmp_1 = (clipped_ratio - 1) * raw_adv**2
                        clip_tmp_2 = 2 * clipped_ratio * raw_adv
                        
                        mean_var_surr = args.papo_omega1 * torch.min(tmp_1 + tmp_2 * args.papo_omega2, clip_tmp_1 + clip_tmp_2 * args.papo_omega2).mean()
                        batch_val = b_values[mb_inds]
                        
                        if args.papo_detailed:
                            kl_div = approx_kl
                            epsilon_adv = torch.max(mb_advantages)
                            bias = 4 * args.gamma * kl_div * epsilon_adv / (1 - args.gamma)**2
                            term_check = mean_surr + batch_val.mean() - bias
                            min_J_square = mean_surr**2 + 2 * batch_val.mean() * mean_surr
                            if term_check < 0: min_J_square = min_J_square * 0.0
                        else:
                            min_J_square = mean_surr**2 + 2 * batch_val.mean() * mean_surr
                        
                        factor = args.papo_omega1 * (1 - args.gamma**2) / args.papo_k
                        L_ = torch.abs(mb_advantages)
                        var_mean_surr = factor * (L_**2 + 2 * L_ * batch_val).mean() - min_J_square
                        
                        pg_loss = -(mean_surr - args.papo_k * (mean_var_surr + var_mean_surr))

                    else: # PPO
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.algo == "PAPO":
                     if approx_kl > args.target_kl: continue_training = False
            
            if args.algo != "PAPO" and args.target_kl is not None and approx_kl > args.target_kl: break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        wandb.log({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "global_step": global_step
        })

        if global_step % 100000 == 0:
            print(f"Game: {args.env_id} | Step: {global_step} | SPS: {int(global_step / (time.time() - start_time))}")
            wandb.log({"charts/SPS": int(global_step / (time.time() - start_time))}, commit=False)

    envs.close()
    writer.close()
    run.finish()

# ============================================================================
# Main 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", help="Options: PPO, ANO, SPO, TRPO, PAPO")
    
    # [PAPO]
    parser.add_argument("--papo-k", type=float, default=7.0)
    parser.add_argument("--papo-omega1", type=float, default=0.005) # Atari Tuned
    parser.add_argument("--papo-omega2", type=float, default=0.005) # Atari Tuned
    parser.add_argument("--papo-detailed", type=lambda x: bool(strtobool(x)), default=True)

    # [TRPO]
    parser.add_argument("--trpo-max-kl", type=float, default=0.02)
    parser.add_argument("--trpo-damping", type=float, default=0.1)
    parser.add_argument("--trpo-cg-iters", type=int, default=10)
    parser.add_argument("--trpo-ls-iters", type=int, default=10)
    parser.add_argument("--trpo-ls-backtrack-ratio", type=float, default=0.5)

    # [Common]
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB User/Team Name")
    parser.add_argument("--total-timesteps", type=int, default=6000000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--epsilons", type=float, nargs='+', default=[0.2, 0.2])
    parser.add_argument("--anneal-lr", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", type=bool, default=True)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--clip-vloss", type=bool, default=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None) 

    args = parser.parse_args()
    
    # Auto-set Target KL for PAPO
    if args.algo == "PAPO" and args.target_kl is None:
        args.target_kl = 0.02

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    seeds = [1, 2, 3, 4, 5] 
    atari_ale_games = [
        "Pong", "Breakout", "Freeway", "Boxing", "Seaquest", "BeamRider", 
        "SpaceInvaders", "Riverraid", "DemonAttack", "Centipede", "VideoPinball", 
        "DoubleDunk", "Asteroids", "Atlantis", "Gopher", "RoadRunner", 
        "TimePilot", "CrazyClimber", "Tutankham", "Robotank", "StarGunner", 
        "UpNDown", "Phoenix", "Bowling", "IceHockey", "Skiing", "Kangaroo", 
        "Zaxxon", "ElevatorAction", "Assault", "AirRaid", "Alien", "Amidar", 
        "Asterix", "BankHeist", "BattleZone", "Berzerk", "Carnival", 
        "ChopperCommand", "Enduro"
    ]

    for seed in seeds:
        args.seed = seed
        for game_name in atari_ale_games:
            args.game_name = game_name
            args.env_id = f"{game_name}-v5"
            
            # [Parallel Check]
            project_name, group_name, run_name = get_run_info(args)
            if check_wandb_run_exists(args.wandb_entity, project_name, group_name, run_name):
                continue
            
            time.sleep(random.uniform(1, 10))
            if check_wandb_run_exists(args.wandb_entity, project_name, group_name, run_name):
                continue 

            train_one_game(args)
    
    print(f"All Experiments Finished!")
