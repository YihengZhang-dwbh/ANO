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

import gymnasium as gym
import envpool
import wandb

# ============================================================================
# [新功能] WandB 查重工具
# ============================================================================
def get_run_info(args):
    """
    生成 Run Name

    group = tag （同组实验共享一个 group，跨 seed 聚合曲线）
    run   = tag + seed （唯一标识单次运行，用于 WandB 查重）
    """
    project_name = f"Atari_{args.game_name}_v5_G2"

    if args.algo == "ANO":
        # 超参顺序: eps(epsilons[0]) / y1 / b —— 三者唯一确定 G(x) 形状
        # :g 去掉无意义尾零（3.0→3、-1.0→-1），保持紧凑且不与分隔符 _ 混淆
        tag = f"TANO_{args.epsilons[0]:g}_{args.ano_y1:g}_{args.ano_b:g}"
    elif args.algo == "TRPO":
        tag = f"TTRPO_{args.trpo_max_kl}"
    elif args.algo == "PAPO":
        tag = f"TPAPO_{args.papo_omega1}_{args.papo_omega2}"
    else: # PPO / SPO
        tag = f"T{args.algo}_{args.clip_coef}"

    group_name = tag
    run_name = f"{tag}_{args.seed}"
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
# ============================================================================
# [ANO] G(x) shaping kernel.
#
# 数学来源见 C:\Code\ICLR_ANO\show_rate.py 与 construction.tex（与
# RLHF/experimental/ano/g_shaping.py、Traditional_RL/mujoco.py 里的实现完全
# 一致，只是内联到这个独立脚本、并保留本文件原有的 f_func(x, side, ...) 调用
# 形式，方便下面 term_p/term_n 的调用点不用大改）。
#
# 旧核 f0 = 45/16*(0.5*logsigmoid(2x)-2*sigmoid(x)) 只有一个自由度（复用
# epsilon）；塑形函数的角色与旧核一致：f(1)=1、f'(1)=1（切于 y=x）、且在
# x=1+eps 处取局部极大值。这些边界条件对应的是 G(x) 本身，**不是** G'(x)：
#
#     G(x) = 1 + (y1/a) * [ bracket(a(x-x0)) - bracket(a(1-x0)) ]
#     bracket(z) = z + log(1-u) + ((r+1)/(r-1)) * log(u+r(1-u))   (r != 1)
#     bracket(z) = z + log(1-u) + 2*(1-u)                         (r == 1)
#     u = sigmoid(z)
#
# 三个独立超参 (eps, y1, b) 里 y1（最大推力/G'(-inf)）与 b（最大拉力/G' 的
# 全局最小值）完全解耦。
# ============================================================================

def _g_shaping_r_of_b(y1: float, b: float) -> float:
    """由 (y1, b) 闭式反解形状参数 r。要求 -y1 < b < 0（构造的精确可达范围）。"""
    if not (y1 > 1.0):
        raise ValueError(f"ano_y1 must be > 1, got {y1}")
    if not (-y1 < b < 0.0):
        raise ValueError(f"ano_b must satisfy -ano_y1 < ano_b < 0 (got ano_b={b}, ano_y1={y1})")
    d = -b / y1
    s = (2.0 * d + math.sqrt(2.0 * (d + 1.0))) / (1.0 - d)
    return 0.5 * (s * s - 2.0)


def _g_shaping_a_of_scale(eps: float, y1: float, r: float) -> float:
    """由 (eps, y1, r) 闭式解标度 a（二次方程的正根，共轭形式避免相消）。"""
    if not (eps > 0.0):
        raise ValueError(f"epsilon must be > 0, got {eps}")
    B = r * (y1 + 1.0) + 1.0
    C = r * (y1 - 1.0)
    q = 2.0 * C / (B + math.sqrt(B * B + 4.0 * C))
    return -math.log(q) / eps


def solve_g_shaping_constants(eps: float, y1: float, b: float):
    """训练开始前调用一次：由 (eps, y1, b) 解出 (r, a, x0)。"""
    r = _g_shaping_r_of_b(y1, b)
    a = _g_shaping_a_of_scale(eps, y1, r)
    return r, a, 1.0 + eps


def g_shaping_kernel(x, a, x0, y1, r):
    """G(x)，塑形函数本身。**不是** G'(x)（早前版本在这里写错了）。

    用 u=sigmoid(z) 参数化避免 E=e^z 溢出；见上面模块头的公式，与
    RLHF/experimental/ano/g_shaping.py 的 g_shaping_kernel 完全同构，已用
    sympy 符号验证过对 z 求导恒等于 show_rate.py 的 G'，并数值网格对照过
    show_rate.G()（见该文件 _selftest，100 组参数、最大相对误差 6.7e-13）。
    """
    z = a * (x - x0)
    z1 = a * (1.0 - x0)
    u = torch.sigmoid(z)
    u1 = 1.0 / (1.0 + math.exp(-z1))

    log_1mu = -torch.nn.functional.softplus(z)
    log_1mu1 = -math.log1p(math.exp(z1)) if z1 < 0 else -(z1 + math.log1p(math.exp(-z1)))

    if abs(r - 1.0) < 1e-6:
        bracket = z + log_1mu + 2.0 * (1.0 - u)
        bracket1 = z1 + log_1mu1 + 2.0 * (1.0 - u1)
    else:
        c3 = (r + 1.0) / (r - 1.0)
        denom = u + r * (1.0 - u)
        denom1 = u1 + r * (1.0 - u1)
        bracket = z + log_1mu + c3 * torch.log(denom)
        bracket1 = z1 + log_1mu1 + c3 * math.log(denom1)

    return 1.0 + (y1 / a) * (bracket - bracket1)


def f_func(x, d, device, g_consts):
    """保留原有调用形式：f_func(x, side, device, ...)，side=0/1 对应正/负分支。

    g_consts = ((r_pos, a_pos, x0_pos, y1_pos), (r_neg, a_neg, x0_neg, y1_neg))；
    两侧目前用同一组 (eps, y1, b)（与旧代码用同一个 epsilons 但两侧含义相同的
    习惯一致），调用点里的 "2 - f_func(2-x, 1, ...)" 已经做了点反射，所以这里
    不需要像旧 f_func 那样自己再翻一次符号。
    """
    r, a, x0, y1 = g_consts[d]
    return g_shaping_kernel(x, a, x0, y1, r)

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
    new_logits = model.get_logits(x)
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
# 网络定义 (TRPO 使用独立的 actor / critic 特征提取器)
# ============================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # TRPO 的信赖域只约束策略。将 actor 与 critic 完全分离，避免 value
        # update 改变共享 CNN 后破坏已验收的 KL 约束。
        self.actor_network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.critic_network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(self.critic_network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        actor_hidden = self.actor_network(x / 255.0)
        logits = self.actor(actor_hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.critic(self.critic_network(x / 255.0))
        return action, probs.log_prob(action), probs.entropy(), value

    def get_logits(self, x):
        return self.actor(self.actor_network(x / 255.0))

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
        sync_tensorboard=False,  # TB sync disabled: wandb 0.13.11's TB watcher
        # adopts stale same-slot event files and truncates the history stream at
        # the previous attempt's last step (poisoned ~20 runs on 2026-09-20).
        # rollout/ep_rew_mean (wandb.log, below) carries the identical values.
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ==============================================================
    # [ANO] Pre-compute G(x) shaping constants (closed form, solved once)
    # ==============================================================
    if args.algo == "ANO":
        _g_r, _g_a, _g_x0 = solve_g_shaping_constants(args.epsilons[0], args.ano_y1, args.ano_b)
        _g_consts = ((_g_r, _g_a, _g_x0, args.ano_y1), (_g_r, _g_a, _g_x0, args.ano_y1))
    else:
        _g_consts = None

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
                wandb.log({
                    "rollout/ep_rew_mean": avg_ret,
                    "rollout/ep_len_mean": avg_len,
                    "global_step": global_step
                })

            clipped_reward = np.sign(reward)
            rewards[step] = torch.tensor(clipped_reward).to(device).reshape(-1)
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
            # [fix] honor norm_adv like every other algorithm (TRPO branch used raw
            # advantages while config sets norm_adv=True; standard TRPO also normalizes).
            adv_t = b_advantages
            if args.norm_adv:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            with torch.no_grad():
                old_logits = agent.get_logits(b_obs)

            diag_ratios = []  # [诊断] line search 评估点的 ratio
            
            policy_params = list(agent.actor_network.parameters()) + list(agent.actor.parameters())
            
            logits = agent.get_logits(b_obs)
            probs = Categorical(logits=logits)
            new_log_probs = probs.log_prob(b_actions)
            ratio = torch.exp(new_log_probs - b_logprobs)
            surrogate_loss = (ratio * adv_t).mean()

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
            # [fix] guard the degenerate natural gradient: when the policy gradient is
            # ~0 (e.g., zero-variance returns), CG exits at iteration 0 with step_dir=0,
            # giving shs=0 -> lm=0 -> full_step=0/0=NaN, which permanently froze the
            # policy (this corrupted all historical Freeway/Enduro/Tutankham/ElevatorAction
            # TRPO runs). Skip the policy update cleanly instead of producing NaN.
            if (not torch.isfinite(shs).all()) or shs.item() <= 1e-12:
                full_step = torch.zeros_like(step_dir)
            else:
                lm = torch.sqrt(shs / args.trpo_max_kl)
                full_step = step_dir / lm[0]
                if torch.isnan(full_step).any():
                    print("TRPO Warning: NaN in full_step")
                    full_step = torch.zeros_like(full_step)

            current_params = torch.cat([flat_params(agent.actor_network), flat_params(agent.actor)])
            
            success = False
            for i in range(args.trpo_ls_iters):
                step_size = args.trpo_ls_backtrack_ratio ** i
                proposed_step = full_step * step_size
                new_params = current_params + proposed_step
                
                actor_network_size = sum(p.numel() for p in agent.actor_network.parameters())
                set_params(agent.actor_network, new_params[:actor_network_size])
                set_params(agent.actor, new_params[actor_network_size:])
                
                with torch.no_grad():
                    new_logits_eval = agent.get_logits(b_obs)
                    new_probs_eval = Categorical(logits=new_logits_eval)
                    new_log_probs_eval = new_probs_eval.log_prob(b_actions)
                    ratio_eval = torch.exp(new_log_probs_eval - b_logprobs)
                    diag_ratios.append(ratio_eval.detach())
                    new_surrogate_loss = (ratio_eval * b_advantages).mean()
                    kl_val = get_kl_discrete(agent, b_obs, old_logits).mean()

                # 在 Policy Update 的 Line Search 循环中：
                if new_surrogate_loss > surrogate_loss and kl_val <= args.trpo_max_kl:
                    success = True
                    break
            
            if not success:
                set_params(agent.actor_network, current_params[:actor_network_size])
                set_params(agent.actor, current_params[actor_network_size:])

            # Value Function Update
            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    newvalue = agent.get_value(b_obs[mb_inds]).reshape(-1)
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
                    
                    # 不应清零 actor 的梯度：critic_network 与 actor_network 已分离，
                    # 此处反向传播只会产生 critic 侧梯度。
                    
                    optimizer.step()
                    # === [修复结束] ===
            
            pg_loss, v_loss, entropy_loss, approx_kl, clipfracs = surrogate_loss, v_loss, torch.tensor(0.0), kl_val, [0.0]

        # ====================================================================
        # [Branch] PPO / ANO / SPO / PAPO Logic
        # ====================================================================
        else:
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            diag_ratios = []  # [诊断] 收集本 update 所有 minibatch 的 ratio，用于分位数统计
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
                    diag_ratios.append(ratio.detach())

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        # ANO 的"边界"由 epsilons[0] 定义（零交叉点），而非 clip_coef
                        _out_thresh = args.epsilons[0] if args.algo == "ANO" else args.clip_coef
                        clipfracs += [((ratio - 1.0).abs() > _out_thresh).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    if args.algo == "ANO":
                        x = ratio
                        term_p = f_func(x, 0, device, _g_consts)
                        term_n = 2.0 - f_func(2.0 - x, 1, device, _g_consts)
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

                    newvalue = newvalue.reshape(-1)
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

        with torch.no_grad():
            if len(diag_ratios) > 0:
                r_all = torch.cat(diag_ratios).float()
                _q = torch.quantile(r_all, torch.tensor([0.5, 0.9, 0.99, 0.999], device=r_all.device))
                _q50, _q90, _q99, _q999 = (v.item() for v in _q)
            else:
                _q50 = _q90 = _q99 = _q999 = float("nan")

        wandb.log({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "diagnostics/out_of_boundary": np.mean(clipfracs) if len(clipfracs) > 0 else float("nan"),
            "diagnostics/ratio_q50": _q50,
            "diagnostics/ratio_q90": _q90,
            "diagnostics/ratio_q99": _q99,
            "diagnostics/ratio_q999": _q999,
            "global_step": global_step
        })

        if global_step % 100000 == 0:
            print(f"Game: {args.env_id} | Step: {global_step} | SPS: {int(global_step / (time.time() - start_time))}")
            wandb.log({"charts/SPS": int(global_step / (time.time() - start_time))}, commit=False)

    envs.close()
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
    parser.add_argument("--ano-y1", type=float, default=3.0,
                         help="G(x) shaping function saturation level as x->-inf ('maximal push'); must be > 1.")
    parser.add_argument("--ano-b", type=float, default=-1.0,
                         help="G(x) shaping function's global min of G' ('maximal pull'); must satisfy -ano_y1 < ano_b < 0.")
    parser.add_argument("--anneal-lr", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", type=bool, default=True)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--clip-vloss", type=bool, default=False)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None) 
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="Optional subset of Atari games to run (default: all 40).")

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

    if args.games:
        unknown = set(args.games) - set(atari_ale_games)
        if unknown:
            raise ValueError(f"Unknown game names: {sorted(unknown)}")
        atari_ale_games = [g for g in atari_ale_games if g in set(args.games)]
        print(f"[games filter] running {len(atari_ale_games)} games: {atari_ale_games}")

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
