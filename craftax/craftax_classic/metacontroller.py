import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import jax
from typing import cast
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from torch.distributions.categorical import Categorical

from envs.craftax_state import EnvParams, StaticEnvParams

### Copied from CleanRL


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CraftaxAgent(nn.Module):
    def __init__(self, observation_shape, action_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space), std=0.01),
        )

    def get_value(self, x) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
            self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class ClassicMetaController:
    def __init__(
        self,
        env_params: EnvParams = EnvParams(),
        static_parameters: StaticEnvParams = StaticEnvParams(),
        device: str = "cpu",
        steps_each_time: int = 300,
        num_iterations: int = 100,
        learning_rate: float = 2.5e-4,
        anneal_lr: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_minibatches: int = 4,
        update_epochs: int = 4,
        clip_coef: float = 0.2,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
    ):
        """
        Params:
        - env_params: non-static parameters
        - static_parameters: static environment parameters
        - device: device to use
        - steps_each_time: Number of steps to take for each batch. Note that this can be less than
            the actual number of steps used for training since the agent can be dead for some steps
        - learning_rate: learning rate
        - anneal_lr: whether to anneal learning rate
        - gamma: decay rate
        - gae_lambda: lambda for generalized advantage estimation 
        - num_minibatches: number of minibatches to use for each batch
        - update_epochs: number of epochs to perform in update step
        - clip_coef: surrogate clipping coefficient
        - norm_adv: whether to use advantage normalization
        - clip_vloss: Whether to use clipped loss for the value function
        - ent_coef: Entropy coefficient
        - vf_coef: Value function coefficient
        - max_grad_norm: The maximum norm for gradient clipping
        - target_kl: target KL divergence threshold
        """
        self.static_params = static_parameters
        self.env = CraftaxClassicSymbolicEnv(self.static_params)
        self.env_params = env_params
        self.steps_each_time = steps_each_time
        self.device = torch.device(device)
        self.rng = jax.random.PRNGKey(np.random.randint(2**31))
        # Learning params
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.num_iterations = num_iterations
        # algo storage
        self.obs = torch.zeros((self.steps_each_time, self.static_params.num_players) + self.env.observation_space(env_params).shape).to(self.device)
        self.actions = torch.zeros((self.steps_each_time, self.static_params.num_players) + self.env.action_space(env_params).shape).to(self.device)
        self.logprobs = torch.zeros((self.steps_each_time, self.static_params.num_players)).to(self.device)
        self.rewards = torch.zeros((self.steps_each_time, self.static_params.num_players)).to(self.device)
        self.dones = torch.zeros((self.steps_each_time, self.static_params.num_players)).to(self.device)
        self.values = torch.zeros((self.steps_each_time, self.static_params.num_players)).to(self.device)

        self.agents = [
            CraftaxAgent(
                self.env.observation_space(env_params).shape,
                self.env.action_space(env_params).n,
            )
            for _ in range(self.static_params.num_players)
        ]
        self.optimizers = [
            optim.Adam(agent.parameters(), lr=self.learning_rate, eps=1e-5) for agent in self.agents  # pyright: ignore
        ]

        self.iteration = 0


    def run_some_episodes(self) -> None:
        """
        Run some episodes, and perform backpropagation on the agents
        """

        self.rng, _rng = jax.random.split(self.rng)
        next_obs, env_state = self.env.reset(_rng, self.env_params)
        next_obs = torch.from_numpy(np.asarray(next_obs)).to(self.device)
        next_done = torch.zeros(self.static_params.num_players).to(self.device)

        self.iteration += 1

        if self.anneal_lr:
            frac = 1.0 - (self.iteration - 1.0) / self.num_iterations
            lrnow = frac * self.learning_rate
            for optimizer in self.optimizers:
                optimizer.param_groups[0]["lr"] = lrnow

        for step in range(self.steps_each_time):
            self.obs[step] = next_obs
            self.dones[step] = next_done
            with torch.no_grad():
                for i, agent in enumerate(self.agents):
                    action, logprob, _, value = agent.get_action_and_value(next_obs[i])
                    self.values[step, i] = value.flatten()
                    self.actions[step, i] = action
                    self.logprobs[step, i] = logprob
            self.rng, _rng = jax.random.split(self.rng)
            next_obs, env_state, reward, next_done, _info = self.env.step(_rng, env_state, self.actions[step].int().cpu().numpy(), self.env_params)
            self.rewards[step] = torch.from_numpy(np.asarray(reward)).to(self.device).view(-1)
            next_obs, next_done = torch.from_numpy(np.asarray(next_obs)).to(self.device), torch.from_numpy(np.asarray(next_done)).to(self.device)
        # bootstrap value if not done
        with torch.no_grad():
            next_values = torch.concat([agent.get_value(obs).reshape(1,) for agent, obs in zip(self.agents, next_obs)]).to(self.device)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.steps_each_time)):
                if t == self.steps_each_time - 1:
                    nextnonterminal = ~next_done
                    nextvalues = next_values
                else:
                    nextnonterminal = 1 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        # Optimize the policy and value network
        for agent_idx, agent in enumerate(self.agents):
            non_dead = self.dones[:, agent_idx] == 0.0
            b_obs = self.obs[non_dead, agent_idx]
            b_logprobs = self.logprobs[non_dead, agent_idx]
            b_actions = self.actions[non_dead, agent_idx]
            b_advantages = advantages[non_dead, agent_idx]
            b_returns = returns[non_dead, agent_idx]
            b_values = self.values[non_dead, agent_idx]

            optimizer = self.optimizers[agent_idx]

            # effectively an int
            batch_size = cast(int, torch.sum(non_dead, dtype=torch.int32).item())
            minibatch_size = batch_size // self.num_minibatches
            b_inds = np.arange(batch_size)
            clipfracs: list[float] = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                approx_kl = 0
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        if len(mb_advantages) == 1:
                            mb_advantages = torch.tensor(0.0)
                        else:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                    optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            print("Agent", agent_idx, "loss:", loss.item())  # pyright: ignore
            print("Agent", agent_idx, "explained variance:", explained_var)
    
    def train(self):
        for episode in range(self.num_iterations):
            print("Episode", episode)
            self.run_some_episodes()


if __name__ == "__main__":
    metacontroller = ClassicMetaController(
        static_parameters=StaticEnvParams(num_players=4),
        steps_each_time=1000,
        num_iterations=10,
    )
    metacontroller.train()
