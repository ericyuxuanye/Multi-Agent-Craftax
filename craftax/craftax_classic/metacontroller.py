from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from craftax.craftax_classic.constants import Action
from craftax.craftax_classic.envs.craftax_symbolic_env import \
    CraftaxClassicSymbolicEnv
from craftax.craftax_classic.game_logic import are_players_alive
from craftax.craftax_classic.renderer import render_craftax_pixels
from numpy.typing import NDArray
from torch import Tensor
from torch.distributions.categorical import Categorical

from envs.craftax_state import EnvParams, EnvState, StaticEnvParams

### Copied from CleanRL


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CraftaxAgent(nn.Module):
    def __init__(self, observation_shape, action_space):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(32, 32)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)
        self.actor = layer_init(nn.Linear(32, action_space))
        self.critic = layer_init(nn.Linear(32, 1), std=1)

    def get_states(
        self, x: Tensor, lstm_state: tuple[Tensor, Tensor], done: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hidden: Tensor = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h)
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(
        self, x: Tensor, lstm_state: tuple[Tensor, Tensor], done: Tensor
    ) -> Tensor:
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(
        self,
        x: Tensor,
        lstm_state: tuple[Tensor, Tensor],
        done: Tensor,
    ) -> tuple[Categorical, Tensor, tuple[Tensor, Tensor]]:
        """
        Returns the action, log probs, entropy, value, and lstm_state
        """
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return (
            probs,
            self.critic(hidden),
            lstm_state,
        )


class ClassicMetaController:
    def __init__(
        self,
        env_params: EnvParams = EnvParams(),
        static_parameters: StaticEnvParams = StaticEnvParams(),
        device: str = "cpu",
        num_envs: int = 8,
        num_steps: int = 300,
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
        - num_envs: number of environments to run in parallel during rollout
        - num_steps: Number of steps to take for each batch. Note that this can be less than
            the actual number of steps used for training since the agent can be dead for some steps
        - num_iterations: Number of rollout/training iterations
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
        self.num_envs = num_envs
        self.env = CraftaxClassicSymbolicEnv(self.static_params)
        self.env_params = env_params
        self.step_fn = jax.jit(jax.vmap(self.env.step, in_axes=(0, 0, 1, None), out_axes=(1, 0, 1, 1, 0)))
        self.reset_fn = jax.jit(jax.vmap(self.env.reset, in_axes=(0, None), out_axes=(1, 0)))
        self.player_alive_check = jax.jit(jax.vmap(are_players_alive, out_axes=1))
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.rng = jax.random.PRNGKey(np.random.randint(2**31))
        self.observation_space = self.env.observation_space(env_params)
        self.action_space = self.env.action_space(env_params)
        # Learning params
        self.batch_size = int(self.num_envs * self.num_steps)
        self.learning_rate = learning_rate
        self.num_envs = num_envs
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
        self.obs = torch.zeros(
            (self.num_steps, self.static_params.num_players, self.num_envs)
            + self.observation_space.shape
        ).to(self.device)
        self.actions = torch.zeros(
            (self.num_steps, self.static_params.num_players, self.num_envs)
            + self.action_space.shape
        ).to(self.device)
        self.logprobs = torch.zeros(
            (self.num_steps, self.static_params.num_players, self.num_envs)
        ).to(self.device)
        self.rewards = torch.zeros(
            (self.num_steps, self.static_params.num_players, self.num_envs)
        ).to(self.device)
        self.dones = torch.zeros(
            (self.num_steps, self.static_params.num_players, self.num_envs)
        ).to(self.device)
        self.values = torch.zeros(
            (self.num_steps, self.static_params.num_players, self.num_envs)
        ).to(self.device)

        self.agents = [
            CraftaxAgent(
                self.observation_space.shape,
                self.action_space.n,
            ).to(device)
            for _ in range(self.static_params.num_players)
        ]
        self.optimizers = [
            optim.Adam(  # pyright: ignore
                agent.parameters(), lr=self.learning_rate, eps=1e-5
            )
            for agent in self.agents
        ]

        self.iteration = 0

    def train_some_episodes(self) -> None:
        """
        Run some episodes, and perform backpropagation on the agents
        """
        self.rng, _rng = jax.random.split(self.rng)
        next_obs, env_state = self.reset_fn(jax.random.split(_rng, self.num_envs), self.env_params)
        next_obs = torch.from_numpy(np.asarray(next_obs)).to(self.device)
        next_done = torch.zeros(self.static_params.num_players, self.num_envs).to(self.device)
        next_lstm_states = [
            (
                torch.zeros(
                    agent.lstm.num_layers, self.num_envs, agent.lstm.hidden_size
                ).to(self.device),
                torch.zeros(
                    agent.lstm.num_layers, self.num_envs, agent.lstm.hidden_size
                ).to(self.device),
            )
            for agent in self.agents
        ]

        initial_lstm_states = [
            (next_lstm_state[0].clone(), next_lstm_state[1].clone())
            for next_lstm_state in next_lstm_states
        ]

        self.iteration += 1

        if self.anneal_lr:
            frac = 1.0 - (self.iteration - 1.0) / self.num_iterations
            lrnow = frac * self.learning_rate
            for optimizer in self.optimizers:
                optimizer.param_groups[0]["lr"] = lrnow

        for step in range(self.num_steps):
            self.obs[step] = next_obs
            self.dones[step] = next_done
            with torch.no_grad():
                agents_alive = self.player_alive_check(env_state)
                for agent_idx, agent in enumerate(self.agents):
                    probs, value, next_lstm_state = (
                        agent.get_action_and_value(
                            next_obs[agent_idx], next_lstm_states[agent_idx], next_done[agent_idx]
                        )
                    )
                    action = probs.sample()
                    action[np.asarray(~agents_alive[agent_idx] | env_state.is_sleeping[:, agent_idx])] = Action.NOOP.value
                    logprob = probs.log_prob(action)
                    self.values[step, agent_idx] = value.flatten()
                    self.actions[step, agent_idx] = action
                    self.logprobs[step, agent_idx] = logprob
                    next_lstm_states[agent_idx] = next_lstm_state
            self.rng, _rng = jax.random.split(self.rng)
            next_obs, env_state, reward, next_done, _info = self.step_fn(
                jax.random.split(_rng, self.num_envs), env_state, self.actions[step].int().cpu().numpy(), self.env_params
            )
            self.rewards[step] = (
                torch.from_numpy(np.asarray(reward)).to(self.device)
            )
            next_obs, next_done = (
                torch.from_numpy(np.asarray(next_obs)).to(self.device),
                torch.from_numpy(np.asarray(next_done)).to(self.device, dtype=torch.float32),
            )
        # bootstrap value if not done
        with torch.no_grad():
            next_values = torch.cat(
                [
                    agent.get_value(obs, next_lstm_state, done).flatten()
                    for agent, obs, next_lstm_state, done in zip(
                        self.agents, next_obs, next_lstm_states, next_done
                    )
                ]
            ).reshape(self.static_params.num_players, self.num_envs).to(self.device)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1 - next_done
                    nextvalues = next_values
                else:
                    nextnonterminal = 1 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = (
                    self.rewards[t]
                    + self.gamma * nextvalues * nextnonterminal
                    - self.values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + self.values

        # Optimize the policy and value network
        envsperbatch = self.num_envs // self.num_minibatches
        # environment indices
        envinds = np.arange(self.num_envs)
        flatinds = np.arange(self.batch_size).reshape(self.num_steps, self.num_envs)
        for agent_idx, agent in enumerate(self.agents):
            b_obs = self.obs[:, agent_idx].reshape((-1,) + self.observation_space.shape)
            b_logprobs = self.logprobs[:, agent_idx].reshape(-1)
            b_actions = self.actions[:, agent_idx].reshape((-1,) + self.action_space.shape)
            b_dones = self.dones[:, agent_idx].reshape(-1)
            b_advantages = advantages[:, agent_idx].reshape(-1)
            b_returns = returns[:, agent_idx].reshape(-1)
            b_values = self.values[:, agent_idx].reshape(-1)

            optimizer = self.optimizers[agent_idx]

            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(envinds)
                approx_kl = torch.tensor(0)
                for start in range(0, self.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                    probs, newvalue, _ = agent.get_action_and_value(
                        b_obs[mb_inds],
                        (
                            initial_lstm_states[agent_idx][0][:, mbenvinds],
                            initial_lstm_states[agent_idx][1][:, mbenvinds],
                        ),
                        b_dones[mb_inds],
                    )
                    action = b_actions.long()[mb_inds]
                    newlogprob = probs.log_prob(action)
                    entropy = probs.entropy()
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        )

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        if len(mb_advantages) == 1:
                            mb_advantages = torch.tensor(0.0)
                        else:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std() + 1e-8
                            )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                    optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )
            print("Agent", agent_idx, "loss:", loss.item())  # pyright: ignore
            print("Agent", agent_idx, "explained variance:", explained_var)
            print("Agent", agent_idx, "KL divergence:", approx_kl.item())  # pyright: ignore
            print("Agent", agent_idx, "old KL divergence:", old_approx_kl.item())  # pyright: ignore
            print("Agent", agent_idx, "average reward:", self.rewards[:, agent_idx].mean().item())  # pyright: ignore

    def train(self):
        for iteration in range(self.num_iterations):
            print("Iteration", iteration)
            self.train_some_episodes()

    def run_one_episode(
        self,
    ) -> tuple[list[EnvState], list[NDArray[np.int32]], list[NDArray[np.float32]]]:
        """
        Runs a single episode, and returns a tuple containing
        the list of states, the list of actions, and the list of rewards
        """
        self.rng, _rng = jax.random.split(self.rng)
        next_obs, env_state = self.env.reset(_rng, self.env_params)
        next_done: NDArray[np.bool] = np.zeros((self.static_params.num_players, 1), dtype=np.bool)
        states: list[EnvState] = [env_state]
        actions: list[NDArray[np.int32]] = []
        rewards: list[NDArray[np.float32]] = []
        next_lstm_states = [
            (
                torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(
                    self.device
                ),
                torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(
                    self.device
                ),
            )
            for agent in self.agents
        ]
        while not jnp.all(next_done):
            agent_actions = np.zeros(self.static_params.num_players, dtype=int)
            with torch.no_grad():
                for i, agent in enumerate(self.agents):
                    probs, value, next_lstm_state = (
                        agent.get_action_and_value(
                            torch.from_numpy(np.asarray(next_obs[i])).to(self.device),
                            next_lstm_states[i],
                            torch.from_numpy(np.asarray(next_done[i], dtype=np.float32)).to(self.device),
                        )
                    )
                    action = probs.sample()
                    agent_actions[i] = action
                    next_lstm_states[i] = next_lstm_state
            self.rng, _rng = jax.random.split(self.rng)
            next_obs, env_state, reward, next_done, _info = self.env.step(
                _rng, env_state, agent_actions, self.env_params
            )
            states.append(env_state)
            actions.append(agent_actions)
            rewards.append(reward)
        return states, actions, rewards


def replay_episode(states: list[EnvState], actions: list[NDArray[np.int32]], num_players: int, player: int = 0):
    import pygame

    pygame.init()
    pygame.key.set_repeat(250, 75)
    screen_surface = pygame.display.set_mode((576, 576))
    render = render_craftax_pixels
    state_idx = 0
    done = False
    clock = pygame.time.Clock()
    while not done:
        # Render
        screen_surface.fill((0, 0, 0))

        pixels = render(
            states[state_idx],
            block_pixel_size=64,
            num_players=num_players,
            player=player,
        )

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        screen_surface.blit(surface, (0, 0))

        pygame.display.flip()

        pygame_events = pygame.event.get()
        for event in pygame_events:
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if state_idx < len(actions):
                    print([Action(x).name for x in actions[state_idx]])
                state_idx += 1
                if state_idx == len(states):
                    done = True

        clock.tick(10)


if __name__ == "__main__":
    metacontroller = ClassicMetaController(
        static_parameters=StaticEnvParams(num_players=4),
        num_envs=128,
        num_minibatches=8,
        num_steps=100,
        num_iterations=20,
        update_epochs=5,
        device="cpu"
    )
    metacontroller.train()
    states, actions, rewards = metacontroller.run_one_episode()
    replay_episode(states, actions, 4)
