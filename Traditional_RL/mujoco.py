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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import envpool
import wandb

# ============================================================================
# [Optimization] ANO Math Kernel (JIT Compiled)
# ============================================================================
@torch.jit.script
def _ano_math_kernel(x: torch.Tensor, k: float, b: float, const_term: float) -> torch.Tensor:
    """
    JIT compiled kernel for ANO math logic.
    f0(x) = 45/16 * (0.5 * logsigmoid(2x) - 2 * sigmoid(x))
    f_func = k * f0((x - b) / k) + const_term
    """
    term_x = (x - b) / k
    # 45/16 = 2.8125
    f0_val = 2.8125 * (0.5 * torch.nn.functional.logsigmoid(2 * term_x) - 2 * torch.sigmoid(term_x))
    return k * f0_val + const_term

@torch.jit.script
def _compute_ano_loss(
    mb_advantage: torch.Tensor, 
    ratio: torch.Tensor, 
    k_pos: float, b_pos: float, c_pos: float,
    k_neg: float, b_neg: float, c_neg: float
) -> torch.Tensor:
    """
    Computes the ANO loss efficiently using branch selection.
    """
    # Branch A: Positive Advantage Case -> f(r)
    f_val_pos = _ano_math_kernel(ratio, k_pos, b_pos, c_pos)
    
    # Branch B: Negative Advantage Case -> 2 - f(2-r)
    # Note: mujoco5 passes positive epsilon magnitudes for both sides, so we handle the reflection here
    f_val_neg = 2.0 - _ano_math_kernel(2.0 - ratio, k_neg, b_neg, c_neg)
    
    # Selection: If Adv >= 0 use Pos branch, else use Neg branch
    target_f_val = torch.where(mb_advantage >= 0, f_val_pos, f_val_neg)
    
    loss = -mb_advantage * target_f_val
    return loss.mean()

# ============================================================================
# [New Feature] WandB Check
# ============================================================================
def get_run_info(args):
    project_name = f"MuJoCo_{args.env_id}_G4" 

    if args.algo == "ANO":
        run_name = f"{args.algo}_{args.epsilons}_{args.seed}"
        group_name = f"{args.algo}_{args.epsilons}_JIT"
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
# [Math Tool] Running Mean Std
# ============================================================================
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

# ============================================================================
# [Wrapper] EnvPoolNormalizeRewardWrapper
# ============================================================================
class EnvPoolNormalizeRewardWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
        elif hasattr(env, "config") and "num_envs" in env.config:
            self.num_envs = env.config["num_envs"]
        else:
            raise AttributeError("Cannot find 'num_envs'")
            
        self.rms = RunningMeanStd(shape=())
        self.gamma = gamma
        self.returns = np.zeros(self.num_envs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        dones = np.logical_or(term, trunc)
        
        self.returns = self.returns * self.gamma + reward
        self.rms.update(self.returns)
        
        # 1. Normalize
        reward = reward / np.sqrt(self.rms.var + 1e-8)
        
        # 2. Clip Reward
        reward = np.clip(reward, -10.0, 10.0)
        
        self.returns[dones] = 0.0
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        return self.env.reset()

# ============================================================================
# [Wrapper] EnvPoolNormalizeObsWrapper
# ============================================================================
class EnvPoolNormalizeObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
        elif hasattr(env, "config") and "num_envs" in env.config:
            self.num_envs = env.config["num_envs"]
        else:
            raise AttributeError("Cannot find 'num_envs'")
            
        self.obs_shape = env.observation_space.shape
        self.rms = RunningMeanStd(shape=self.obs_shape)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.rms.update(obs)
        obs = np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -10.0, 10.0)
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        ret = self.env.reset()
        if isinstance(ret, tuple):
            obs, info = ret
        else:
            obs = ret
            info = {}
        obs = np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -10.0, 10.0)
        if isinstance(ret, tuple):
            return obs, info
        return obs

# ============================================================================
# [Wrapper] RecordEpisodeStatistics
# ============================================================================
class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
        elif hasattr(env, "config") and "num_envs" in env.config:
            self.num_envs = env.config["num_envs"]
        else:
            raise AttributeError("Cannot find 'num_envs'")
            
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.episode_returns += reward
        self.episode_lengths += 1
        
        dones = np.logical_or(term, trunc)
        if np.sum(dones) > 0:
            if "episode" not in info:
                info["episode"] = {"r": [], "l": []}
            
            for i in range(self.num_envs):
                if dones[i]:
                    info["episode"]["r"].append(self.episode_returns[i])
                    info["episode"]["l"].append(self.episode_lengths[i])
                    self.episode_returns[i] = 0
                    self.episode_lengths[i] = 0
        
        return obs, reward, term, trunc, info

# ============================================================================
# [Math Tool] TRPO Utils & Helpers
# ============================================================================
def flat_grad(grads, params):
    grad_flatten = []
    for grad in grads:
        if grad is None:
            continue
        grad_flatten.append(grad.view(-1))
    return torch.cat(grad_flatten)

def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)

def set_params(model, new_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            new_params[prev_ind:prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size

def get_kl(model, x, old_mean, old_logstd):
    """Calculate Analytical KL Divergence"""
    new_mean = model.actor_mean(x)
    new_logstd = model.actor_logstd.expand_as(new_mean)
    new_std = torch.exp(new_logstd)
    old_std = torch.exp(old_logstd)
    
    kl = new_logstd - old_logstd + (old_std.pow(2) + (old_mean - new_mean).pow(2)) / (2.0 * new_std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)

def conjugate_gradient(fvp_func, b, cg_iters=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    
    for _ in range(cg_iters):
        if rdotr < residual_tol:
            break
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
# [Agent]
# ============================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.obs_dim = np.array(envs.single_observation_space.shape).prod()
        self.act_dim = np.array(envs.single_action_space.shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, self.act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def train_one_game(args):
    if "Humanoid" in args.env_id:
        real_total_timesteps = max(args.total_timesteps, 10_000_000)
    else:
        real_total_timesteps = args.total_timesteps

    project_name, group_name, run_name = get_run_info(args)
    
    print(f"--- Starting: {run_name} on {args.env_id} ---")
    run = wandb.init(project=project_name, group=group_name, name=run_name, config=vars(args), monitor_gym=False, reinit=True)
    writer = SummaryWriter(f"runs/{args.env_id}/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ====================================================================
    # [Optimization] Pre-compute ANO Constants
    # ====================================================================
    # We calculate these once on CPU to avoid doing it every step in the loop
    if args.algo == "ANO":
        # Positive side (using args.epsilons[0])
        _eps_pos = args.epsilons[0]
        _k_pos = _eps_pos / math.log(2)
        _b_pos = 1.0 + _eps_pos
        _term_d_pos = (1.0 - _b_pos) / _k_pos
        # Manually compute f0 constant for term_d
        _f0_val_pos = 2.8125 * (0.5 * math.log(1 / (1 + math.exp(-2 * _term_d_pos))) - 2 * (1 / (1 + math.exp(-_term_d_pos))))
        _c_pos = 1.0 - (_k_pos * _f0_val_pos)

        # Negative side (using args.epsilons[1])
        _eps_neg = args.epsilons[1]
        _k_neg = _eps_neg / math.log(2)
        _b_neg = 1.0 + _eps_neg
        _term_d_neg = (1.0 - _b_neg) / _k_neg
        _f0_val_neg = 2.8125 * (0.5 * math.log(1 / (1 + math.exp(-2 * _term_d_neg))) - 2 * (1 / (1 + math.exp(-_term_d_neg))))
        _c_neg = 1.0 - (_k_neg * _f0_val_neg)
    else:
        # Dummy values to prevent UnboundLocalError if algo is not ANO
        _k_pos = _b_pos = _c_pos = _k_neg = _b_neg = _c_neg = 0.0

    # --- Env Setup ---
    num_cpus = os.cpu_count() or 4
    envs = envpool.make(
        args.env_id,
        env_type="gymnasium",
        num_envs=args.num_envs,
        seed=args.seed,
        num_threads = min(args.num_envs, num_cpus),
    )
    envs = RecordEpisodeStatistics(envs)
    envs = EnvPoolNormalizeObsWrapper(envs) 
    envs = EnvPoolNormalizeRewardWrapper(envs, gamma=args.gamma)
    
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    raw_advantages = torch.zeros((args.num_steps, args.num_envs)).to(device) 

    global_step = 0
    start_time = time.time()
    
    initial_reset = envs.reset()
    if isinstance(initial_reset, tuple): initial_obs = initial_reset[0]
    else: initial_obs = initial_reset
    next_obs = torch.Tensor(initial_obs).to(device)
    next_term = torch.zeros(args.num_envs).to(device)

    num_updates = real_total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            terminations[step] = next_term

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_term = torch.Tensor(term).to(device)

            if "episode" in info and len(info["episode"]["r"]) > 0:
                avg_ret = np.mean(info["episode"]["r"])
                writer.add_scalar("charts/episodic_return", avg_ret, global_step)
                wandb.log({"rollout/ep_rew_mean": avg_ret, "global_step": global_step})

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_term
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - terminations[t+1]
                    nextvalues = values[t + 1]
                
                # Delta (Raw Advantage)
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                raw_advantages[t] = delta 
                
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_raw_advantages = raw_advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ====================================================================
        # [Branch] TRPO Logic
        # ====================================================================
        if args.algo == "TRPO":
            with torch.no_grad():
                old_action_mean = agent.actor_mean(b_obs)
                old_action_logstd = agent.actor_logstd.expand_as(old_action_mean)

            action_mean = agent.actor_mean(b_obs)
            action_logstd = agent.actor_logstd.expand_as(action_mean)
            dist = Normal(action_mean, torch.exp(action_logstd))
            new_log_probs = dist.log_prob(b_actions).sum(1)
            ratio = torch.exp(new_log_probs - b_logprobs)
            surrogate_loss = (ratio * b_advantages).mean()
            
            grads = torch.autograd.grad(surrogate_loss, list(agent.actor_mean.parameters()) + [agent.actor_logstd])
            g = flat_grad(grads, list(agent.actor_mean.parameters()) + [agent.actor_logstd])

            def fvp_func(v):
                kl = get_kl(agent, b_obs, old_action_mean, old_action_logstd).mean()
                grads = torch.autograd.grad(kl, list(agent.actor_mean.parameters()) + [agent.actor_logstd], create_graph=True)
                flat_grad_kl = flat_grad(grads, list(agent.actor_mean.parameters()) + [agent.actor_logstd])
                kl_v = (flat_grad_kl * v).sum()
                grads_v = torch.autograd.grad(kl_v, list(agent.actor_mean.parameters()) + [agent.actor_logstd], retain_graph=False)
                return flat_grad(grads_v, list(agent.actor_mean.parameters()) + [agent.actor_logstd]) + args.trpo_damping * v

            step_dir = conjugate_gradient(fvp_func, g, cg_iters=args.trpo_cg_iters)

            shs = 0.5 * (step_dir * fvp_func(step_dir)).sum(0, keepdim=True)
            lm = torch.sqrt(shs / args.trpo_max_kl)
            full_step = step_dir / lm[0]
            
            if torch.isnan(full_step).any():
                print("TRPO Warning: NaN in full_step, skipping update.")
                full_step = torch.zeros_like(full_step)

            current_actor_params = flat_params(agent.actor_mean)
            current_logstd = agent.actor_logstd.data.view(-1)
            all_current_params = torch.cat([current_actor_params, current_logstd])
            
            success = False
            for i in range(args.trpo_ls_iters):
                step_size = args.trpo_ls_backtrack_ratio ** i
                proposed_step = full_step * step_size
                new_params = all_current_params + proposed_step
                
                split_idx = sum(p.numel() for p in agent.actor_mean.parameters())
                set_params(agent.actor_mean, new_params[:split_idx])
                agent.actor_logstd.data.copy_(new_params[split_idx:].view(agent.actor_logstd.shape))
                
                with torch.no_grad():
                    new_mean_eval = agent.actor_mean(b_obs)
                    new_dist_eval = Normal(new_mean_eval, torch.exp(agent.actor_logstd.expand_as(new_mean_eval)))
                    new_log_probs_eval = new_dist_eval.log_prob(b_actions).sum(1)
                    ratio_eval = torch.exp(new_log_probs_eval - b_logprobs)
                    new_surrogate_loss = (ratio_eval * b_advantages).mean()
                    kl_val = get_kl(agent, b_obs, old_action_mean, old_action_logstd).mean()

                if new_surrogate_loss > surrogate_loss and kl_val <= args.trpo_max_kl:
                    success = True
                    break
            
            if not success:
                set_params(agent.actor_mean, all_current_params[:split_idx])
                agent.actor_logstd.data.copy_(all_current_params[split_idx:].view(agent.actor_logstd.shape))

            # --- 2. Value Function Update ---
            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                    if args.clip_vloss:
                         v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                         v_clipped = b_values[mb_inds] + torch.clamp(
                             newvalue - b_values[mb_inds],
                             -args.clip_coef,
                             args.clip_coef,
                         )
                         v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                         v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                         v_loss = 0.5 * v_loss_max.mean()
                    else:
                         v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    optimizer.zero_grad()
                    (v_loss * args.vf_coef).backward()
                    nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                    agent.actor_mean.zero_grad()
                    if agent.actor_logstd.grad is not None: agent.actor_logstd.grad.zero_()
                    optimizer.step()

        # ====================================================================
        # [Branch] PPO / ANO / SPO / PAPO Logic
        # ====================================================================
        else:
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            
            # [PAPO Early Stopping Flag]
            continue_training = True

            for epoch in range(args.update_epochs):
                if not continue_training: break # Early Stop

                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # [ALGO SELECTOR]
                    if args.algo == "ANO":
                        # [Optimized] Call JIT compiled kernel directly
                        pg_loss = _compute_ano_loss(
                            mb_advantages, ratio,
                            _k_pos, _b_pos, _c_pos,
                            _k_neg, _b_neg, _c_neg
                        )
                    
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
                        
                        mean_var_surr = args.papo_omega1 * torch.min(
                            tmp_1 + tmp_2 * args.papo_omega2, 
                            clip_tmp_1 + clip_tmp_2 * args.papo_omega2
                        ).mean()
                        
                        batch_val = b_values[mb_inds]
                        
                        if args.papo_detailed:
                            kl_div = approx_kl 
                            epsilon_adv = torch.max(mb_advantages) 
                            bias = 4 * args.gamma * kl_div * epsilon_adv / (1 - args.gamma)**2
                            
                            term_check = mean_surr + batch_val.mean() - bias
                            
                            min_J_square = mean_surr**2 + 2 * batch_val.mean() * mean_surr
                            if term_check < 0:
                                min_J_square = min_J_square * 0.0 
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
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                # [PAPO Check] Early Stopping Condition
                if args.algo == "PAPO":
                     if approx_kl > args.target_kl: 
                         continue_training = False
                         
            # Early stop for PPO (Optional)
            if args.algo != "PAPO" and args.target_kl is not None and approx_kl > args.target_kl:
                break

    envs.close()
    writer.close()
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", help="Options: PPO, ANO, SPO, TRPO, PAPO")
    
    # [PAPO Hyperparameters]
    parser.add_argument("--papo-k", type=float, default=7.0, help="Probability factor k")
    parser.add_argument("--papo-omega1", type=float, default=0.005, help="Weight for mean variance (Default: 0.005 from ASCPO grid search)")
    parser.add_argument("--papo-omega2", type=float, default=0.005, help="Hyperparameter for H_max (Default: 0.005 from ASCPO grid search)")
    parser.add_argument("--papo-detailed", type=lambda x: bool(strtobool(x)), default=True, help="Use detailed bias correction in PAPO")

    # [TRPO Hyperparameters]
    parser.add_argument("--trpo-max-kl", type=float, default=0.01)
    parser.add_argument("--trpo-damping", type=float, default=0.1)
    parser.add_argument("--trpo-cg-iters", type=int, default=10)
    parser.add_argument("--trpo-ls-iters", type=int, default=10)
    parser.add_argument("--trpo-ls-backtrack-ratio", type=float, default=0.5)
    
    # [Common Hyperparameters]
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--total-timesteps", type=int, default=int(1e7)) # 10M
    parser.add_argument("--num-envs", type=int, default=128) 
    parser.add_argument("--num-steps", type=int, default=32) 
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=10) 
    
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--epsilons", type=float, nargs='+', default=[0.2, 0.2])
    parser.add_argument("--anneal-lr", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--norm-adv", type=bool, default=True)
    parser.add_argument("--clip-coef", type=float, default=0.3)
    parser.add_argument("--clip-vloss", type=bool, default=True)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    # [Target KL] 
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    
    args = parser.parse_args()

    if args.algo == "PAPO" and args.target_kl is None:
        args.target_kl = 0.02
        
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    # Game Loop
    seeds = [1, 2, 3, 4, 5]
    mujoco_games = [
        "HalfCheetah-v4",
        "Ant-v4",
        "Hopper-v4",
        "Walker2d-v4",
        "Humanoid-v4",
        "Swimmer-v4",
    ]
    for seed in seeds:
        args.seed = seed
        for env_id in mujoco_games:
            args.env_id = env_id
            
            # [Parallel Check]
            project_name, group_name, run_name = get_run_info(args)
            if check_wandb_run_exists(args.wandb_entity, project_name, group_name, run_name):
                continue
                
            time.sleep(random.uniform(1, 10))
            if check_wandb_run_exists(args.wandb_entity, project_name, group_name, run_name):
                continue
                
            train_one_game(args)
