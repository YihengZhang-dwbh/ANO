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
# [ANO] G(x) shaping kernel (JIT compiled)
#
# 数学来源见 C:\Code\ICLR_ANO\show_rate.py 与 construction.tex（同一构造，这
# 里是它的 torch 版，与 RLHF/experimental/ano/g_shaping.py 完全一致，只是内联
# 到这个独立脚本里，因为它不是一个可 import 的包）。
#
# 旧核 f0 = 45/16*(0.5*logsigmoid(2x)-2*sigmoid(x)) 只有一个自由度（复用
# epsilon）；塑形函数扮演的角色与旧核一致：直接替换 PPO 里 clip(ratio) 的
# 位置，f(1)=1、f'(1)=1（切于 y=x）、且在 x=1+eps 处取**局部极大值**。这些
# 边界条件对应的是 G(x) 本身（G(1)=1, G'(1)=1, G 在 1+eps 处局部极大），
# **不是** G'(x)（早前版本在这里写错了，把 G' 当塑形函数塞了进去）。
#
#     G(x) = 1 + (y1/a) * [ bracket(a(x-x0)) - bracket(a(1-x0)) ]
#     bracket(z) = z + log(1-u) + ((r+1)/(r-1)) * log(u+r(1-u))   (r != 1)
#     bracket(z) = z + log(1-u) + 2*(1-u)                         (r == 1，极限)
#     u = sigmoid(z)
#
# 三个独立超参 (eps, y1, b)：eps 是零点偏移（与旧核相同语义），y1 是
# G'(-inf)（"最大推力"），b 是 G' 的全局最小值（"最大拉力"），且 b 与 y1
# 完全解耦（不像旧核只有一个自由度）。(r, a, x0) 在训练开始前用闭式（二次
# 方程正根 + 显式深度反解）解一次，训练循环里只做逐元素初等运算，不含任何
# 求根，用 torch.jit.script 保持与旧核一致的性能特征。
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


@torch.jit.script
def g_shaping_kernel(x: torch.Tensor, r: float, a: float, x0: float, y1: float) -> torch.Tensor:
    """G(x)，塑形函数本身（替代旧 _ano_math_kernel）。**不是** G'(x)。

    见上面模块头的公式；数值上用 u=sigmoid(z) 代替 E=e^z 避免溢出，
    log(1-u) 用稳定的 -softplus(z) 求。已用 sympy 符号验证 + 数值网格
    对照 show_rate.py（见 RLHF/experimental/ano/g_shaping.py 的 _selftest，
    完全同构，这里为了跑在独立脚本里改成了内联版）。
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


@torch.jit.script
def _compute_ano_loss(
    mb_advantage: torch.Tensor,
    ratio: torch.Tensor,
    r: float, a: float, x0: float, y1: float,
) -> torch.Tensor:
    """
    Computes the ANO loss efficiently using branch selection.

    与旧版完全一致的正负 Adv 分支组合方式（核心函数换成了 G，不是 G'）：
        Adv >= 0: loss = -Adv * G(r)
        Adv <  0: loss = -Adv * [2 - G(2-r)]

    这里的常数 "2" 不需要按 y1 缩放：G 与旧核共享同一个锚点 G(1)=1，
    对偶 g(x)=2-G(2-x) 只需 g(1)=2-G(1)=2-1=1，用常数 2 就精确满足。
    """
    # Branch A: Positive Advantage Case -> G(r)
    f_val_pos = g_shaping_kernel(ratio, r, a, x0, y1)

    # Branch B: Negative Advantage Case -> 2 - G(2-r)
    f_val_neg = 2.0 - g_shaping_kernel(2.0 - ratio, r, a, x0, y1)

    # Selection: If Adv >= 0 use Pos branch, else use Neg branch
    target_f_val = torch.where(mb_advantage >= 0, f_val_pos, f_val_neg)

    loss = -mb_advantage * target_f_val
    return loss.mean()

# ============================================================================
# [New Feature] WandB Check
# ============================================================================
def get_run_info(args):
    """
    生成 Run Name

    group = tag （同组实验共享一个 group，跨 seed 聚合曲线）
    run   = tag + seed （唯一标识单次运行，用于 WandB 查重）
    """
    project_name = f"MuJoCo_{args.env_id}_G4"

    if args.algo == "ANO":
        # 超参顺序: eps(epsilons[0]) / y1 / b —— 三者唯一确定 G(x) 形状
        # :g 去掉无意义尾零（3.0→3、-1.0→-1），保持紧凑且不与分隔符 _ 混淆
        tag = f"TANO_{args.epsilons[0]:g}_{args.ano_y1:g}_{args.ano_b:g}"
    elif args.algo == "TRPO":
        tag = f"TTRPO_{args.trpo_max_kl}"
    elif args.algo == "PAPO":
        tag = f"TPAPO_{args.papo_omega1}_{args.papo_omega2}"
    elif args.algo == "TrulyPPO":
        tag = f"TTrulyPPO_{args.trulyppo_klrange}_{args.trulyppo_slope_rollback}_{args.trulyppo_slope_likelihood}"
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
            print(f"??  [Skip] Found existing run on WandB: {project}/{group}/{name}")
            return True
        return False
    except Exception as e:
        print(f"??  [WandB Check Error] {e} -> Proceeding...")
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
    # [ANO] Pre-compute G(x) shaping constants (closed form, solved once)
    # ====================================================================
    if args.algo == "ANO":
        _g_r, _g_a, _g_x0 = solve_g_shaping_constants(args.epsilons[0], args.ano_y1, args.ano_b)
        _g_y1 = args.ano_y1
    else:
        # Dummy values to prevent UnboundLocalError if algo is not ANO
        _g_r = _g_a = _g_x0 = _g_y1 = 0.0

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

        # [诊断] 收集本 update 所有 minibatch 的 ratio，用于分位数统计
        diag_ratios = []

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
            # [fix] guard the degenerate natural gradient (same bug as atari.py):
            # g ~ 0 -> CG exits at iter 0 -> step_dir = 0 -> shs = 0 -> lm = 0
            # -> full_step = 0/0 = NaN -> policy frozen. Skip cleanly instead.
            if (not torch.isfinite(shs).all()) or shs.item() <= 1e-12:
                full_step = torch.zeros_like(step_dir)
            else:
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
                    diag_ratios.append(ratio_eval.detach())
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
        # [Branch] PPO / ANO / SPO / PAPO / TrulyPPO Logic
        # ====================================================================
        else:
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            
            # [PAPO Early Stopping Flag]
            continue_training = True

            if args.algo == "TrulyPPO":
                with torch.no_grad():
                    b_old_action_mean = agent.actor_mean(b_obs).clone().detach()
                    b_old_action_logstd = agent.actor_logstd.expand_as(b_old_action_mean).clone().detach()

            for epoch in range(args.update_epochs):
                if not continue_training: break # Early Stop

                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    diag_ratios.append(ratio.detach())

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # [ALGO SELECTOR]
                    if args.algo == "ANO":
                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            # ANO 的"边界"由 epsilons[0] 定义（零交叉点），而非 clip_coef
                            clipfracs += [((ratio - 1.0).abs() > args.epsilons[0]).float().mean().item()]
                        # [Optimized] Call JIT compiled G(x) shaping kernel directly
                        pg_loss = _compute_ano_loss(
                            mb_advantages, ratio,
                            _g_r, _g_a, _g_x0, _g_y1,
                        )
                    
                    elif args.algo == "SPO":
                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                        pg_loss = -(mb_advantages * ratio - torch.abs(mb_advantages) * torch.pow(ratio - 1, 2) / (2 * args.clip_coef)).mean()
                    
                    elif args.algo == "PAPO":
                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
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

                    elif args.algo == "TrulyPPO":
                        new_action_mean = agent.actor_mean(b_obs[mb_inds])
                        new_action_logstd = agent.actor_logstd.expand_as(new_action_mean)
                        new_std = torch.exp(new_action_logstd)
                        old_std = torch.exp(b_old_action_logstd[mb_inds])
                        old_mean = b_old_action_mean[mb_inds]

                        kl = new_action_logstd - b_old_action_logstd[mb_inds] + (old_std.pow(2) + (old_mean - new_action_mean).pow(2)) / (2.0 * new_std.pow(2)) - 0.5
                        kl = kl.sum(1) # shape: [minibatch_size]

                        with torch.no_grad():
                            approx_kl = kl.mean()
                            old_approx_kl = (-logratio).mean()
                            clipfracs += [((kl >= args.trulyppo_klrange) & (ratio * mb_advantages > mb_advantages)).float().mean().item()]

                        pg_targets = torch.where(
                            (kl >= args.trulyppo_klrange) & (ratio * mb_advantages > mb_advantages),
                            args.trulyppo_slope_likelihood * ratio * mb_advantages + args.trulyppo_slope_rollback * kl,
                            ratio * mb_advantages
                        )
                        pg_loss = -pg_targets.mean()

                    else: # PPO
                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
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

        # ====================================================================
        # [Logging] Per-update metrics (losses + gain-field diagnostics)
        # ====================================================================
        with torch.no_grad():
            if len(diag_ratios) > 0:
                r_all = torch.cat(diag_ratios).float()
                _q = torch.quantile(r_all, torch.tensor([0.5, 0.9, 0.99, 0.999], device=r_all.device))
                _q50, _q90, _q99, _q999 = (v.item() for v in _q)
            else:
                _q50 = _q90 = _q99 = _q999 = float("nan")

            _payload = {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "diagnostics/ratio_q50": _q50,
                "diagnostics/ratio_q90": _q90,
                "diagnostics/ratio_q99": _q99,
                "diagnostics/ratio_q999": _q999,
                "global_step": global_step,
            }
            if args.algo == "TRPO":
                _payload["losses/approx_kl"] = kl_val.item()
                _payload["diagnostics/trpo_step_accepted"] = float(success)
            else:
                _payload["losses/value_loss"] = v_loss.item()
                _payload["losses/policy_loss"] = pg_loss.item()
                _payload["losses/entropy"] = entropy_loss.item()
                _payload["losses/approx_kl"] = approx_kl.item()
                _payload["diagnostics/out_of_boundary"] = (
                    float(np.mean(clipfracs)) if len(clipfracs) > 0 else float("nan")
                )
            wandb.log(_payload)

        if global_step % 100000 == 0:
            print(f"Env: {args.env_id} | Step: {global_step} | SPS: {int(global_step / (time.time() - start_time))}")

    envs.close()
    writer.close()
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", help="Options: PPO, ANO, SPO, TRPO, PAPO, TrulyPPO")
    
    # [TrulyPPO Hyperparameters]
    parser.add_argument("--trulyppo-klrange", type=float, default=0.03, help="TrulyPPO KLRANGE")
    parser.add_argument("--trulyppo-slope-rollback", type=float, default=-5.0, help="TrulyPPO slope_rollback")
    parser.add_argument("--trulyppo-slope-likelihood", type=float, default=1.0, help="TrulyPPO slope_likelihood")
    
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
    parser.add_argument("--ano-y1", type=float, default=3.0,
                         help="G(x) shaping function saturation level as x->-inf ('maximal push'); must be > 1.")
    parser.add_argument("--ano-b", type=float, default=-1.0,
                         help="G(x) shaping function's global min of G' ('maximal pull'); must satisfy -ano_y1 < ano_b < 0.")
    parser.add_argument("--anneal-lr", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--norm-adv", type=bool, default=True)
    parser.add_argument("--clip-coef", type=float, default=0.3)
    parser.add_argument("--clip-vloss", type=bool, default=False)
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
