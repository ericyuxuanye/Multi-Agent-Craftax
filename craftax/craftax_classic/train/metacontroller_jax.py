from functools import partial
from random import randrange

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.linen import initializers

from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.craftax_classic.game_logic import are_players_alive
from craftax.craftax_classic.train.logger import TrainLogger


class LSTM(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, terminations, last_state):
        features = self.features
        batch_size = last_state[0].shape[0]

        reshaped_x = x.reshape(-1, batch_size, features)
        reshaped_terminations = terminations.reshape(-1, batch_size)

        class LSTMOut(nn.Module):
            @nn.compact
            def __call__(self, carry, inputs):
                inputs, terminate = inputs
                # Reset the hidden state on termination
                carry = jax.tree.map(
                    lambda carry_component: jax.lax.select(
                        jnp.repeat(
                            jnp.reshape(terminate != 0.0, (-1, 1)),
                            carry_component.shape[1],
                            axis=1,
                        ),
                        jnp.zeros_like(carry_component),
                        carry_component,
                    ),
                    carry,
                )
                (new_c, new_h), new_h = nn.OptimizedLSTMCell(
                    features,
                    kernel_init=initializers.orthogonal(jnp.sqrt(2)),
                    bias_init=initializers.constant(0.0),
                )(carry, inputs)
                return (new_c, new_h), ((new_c, new_h), new_h)

        model = nn.scan(
            LSTMOut, variable_broadcast="params", split_rngs={"params": False}
        )
        final_state, (new_states, y_t) = model()(
            last_state, (reshaped_x, reshaped_terminations)
        )

        if reshaped_x.shape[0] == 1:
            y_t = jnp.squeeze(y_t, axis=0)

        return y_t, final_state


class CraftaxAgent(nn.Module):
    action_space: int

    def setup(self):
        self.network = nn.Sequential(
            [
                nn.Dense(64, kernel_init=initializers.orthogonal(jnp.sqrt(2))),
                nn.relu,
                nn.Dense(32, kernel_init=initializers.orthogonal(jnp.sqrt(2))),
                nn.relu,
            ]
        )
        self.actor_1 = nn.Dense(64, kernel_init=initializers.orthogonal(2))
        self.actor_out = nn.Dense(
            self.action_space, kernel_init=initializers.orthogonal(0.01)
        )
        self.critic_1 = nn.Dense(64, kernel_init=initializers.orthogonal(2))
        self.critic_out = nn.Dense(1, kernel_init=initializers.orthogonal(1.0))
        self.lstm = LSTM(32)

    def get_states(self, x, lstm_state, done):
        x = self.network(x)
        return self.lstm(x, done, lstm_state)

    def actor(self, x):
        x = self.actor_1(x)
        x = nn.relu(x)
        x = self.actor_out(x)
        return x

    def critic(self, x):
        x = self.critic_1(x)
        x = nn.relu(x)
        x = self.critic_out(x)
        return x

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def __call__(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value, lstm_state


class ClassicMetaController:
    def __init__(
        self,
        env_params: EnvParams = EnvParams(),
        static_parameters: StaticEnvParams = StaticEnvParams(),
        num_envs: int = 8,
        num_steps: int = 300,
        num_iterations: int = 100,
        learning_rate: float = 2.5e-3,
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
        # target_kl: float | None = None,
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
        self.step_fn = jax.vmap(
            self.env.step, in_axes=(0, 0, 1, None), out_axes=(1, 0, 1, 1, 0)
        )
        self.reset_fn = jax.vmap(self.env.reset, in_axes=(0, None), out_axes=(1, 0))
        self.player_alive_check = jax.vmap(are_players_alive, out_axes=1)
        self.num_steps = num_steps
        # self.rng = jax.random.PRNGKey(randrange(2**31))
        self.rng = jax.random.PRNGKey(56)
        self.observation_space = self.env.observation_space(env_params)
        self.action_space = self.env.action_space(env_params)
        # Learning params
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
        # self.target_kl = target_kl
        self.num_iterations = num_iterations

        self.agent = CraftaxAgent(self.action_space.n)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=self.learning_rate),
        )
        self.lr_schedule = optax.linear_schedule(
            self.learning_rate, 0.0, self.num_iterations
        )

    @partial(jax.jit, static_argnums=(0,))
    def train_some_episodes(
        self, rng, tick, model_params, opt_states, next_lstm_states, next_obs, env_state
    ):
        rng, _rng = jax.random.split(rng)
        next_done = jnp.zeros((self.static_params.num_players, self.num_envs))
        init_lstm_states = next_lstm_states
        if self.anneal_lr:
            # update learning rate
            opt_states[1].hyperparams["learning_rate"] = jnp.full(  # pyright: ignore
                self.static_params.num_players, self.lr_schedule(tick)
            )

        def rollout_step(carry, step):
            (
                next_obs,
                next_done,
                env_state,
                next_lstm_states,
                rng,
            ) = carry
            init_obs = next_obs
            init_done = next_done
            # agents_alive = self.player_alive_check(env_state)

            def eval_agent(rng, agent_idx, model_param, next_lstm_state):
                rng, _rng = jax.random.split(rng)
                logits, value, next_lstm_state = self.agent.apply(  # pyright: ignore
                    model_param,
                    next_obs[agent_idx],
                    next_lstm_state,
                    next_done[agent_idx],
                )
                # sets the action to NOOP if player is dead or sleeping
                # I think this does more harm than good
                # action = jax.lax.select(
                #     ~agents_alive[agent_idx] | env_state.is_sleeping[:, agent_idx],
                #     jnp.full(self.num_envs, Action.NOOP.value),
                #     jax.random.categorical(_rng, logits),
                # )
                action = jax.random.categorical(_rng, logits)
                return (
                    value.flatten(),  # pyright: ignore
                    action,
                    jax.nn.log_softmax(logits)[jnp.arange(self.num_envs), action],
                    next_lstm_state,
                )

            rng, _rng = jax.random.split(rng)
            value, action, logprob, next_lstm_states = jax.vmap(eval_agent)(
                jax.random.split(_rng, self.static_params.num_players),
                jnp.arange(self.static_params.num_players),
                model_params,
                next_lstm_states,
            )
            rng, _rng = jax.random.split(rng)
            next_obs, env_state, reward, next_done, _info = self.step_fn(
                jax.random.split(_rng, self.num_envs),
                env_state,
                action.astype(int),
                self.env_params,
            )
            next_done = next_done.astype(float)
            return (
                next_obs,
                next_done,
                env_state,
                next_lstm_states,
                rng,
            ), (init_obs, init_done, value, action, logprob, reward)

        # ugly code
        (
            (
                next_obs,
                next_done,
                env_state,
                next_lstm_states,
                rng,
            ),
            (obs, dones, values, actions, logprobs, rewards),
        ) = jax.lax.scan(
            rollout_step,
            (
                next_obs,
                next_done,
                env_state,
                next_lstm_states,
                rng,
            ),
            jnp.arange(self.num_steps),
        )

        # bootstrap value if not done
        def produce_value(model_param, next_obs, next_lstm_state, next_done):
            return self.agent.apply(
                model_param,
                next_obs,
                next_lstm_state,
                next_done,
                method=CraftaxAgent.get_value,
            ).flatten()  # pyright: ignore

        next_values = jax.vmap(produce_value)(
            model_params, next_obs, next_lstm_states, next_done
        ).reshape(self.static_params.num_players, self.num_envs)

        def compute_advantages(carry, transition):
            lastgaelam, next_value, next_done = carry
            done, value, reward = transition
            delta = reward + self.gamma * next_value * (1 - next_done) - value
            lastgaelam = (
                delta + self.gamma * self.gae_lambda * (1 - next_done) * lastgaelam
            )
            return (lastgaelam, value, done), lastgaelam

        _, advantages = jax.lax.scan(
            compute_advantages,
            (jnp.zeros_like(next_done), next_values, next_done),
            (dones, values, rewards),
            reverse=True,
        )
        returns = advantages + values

        # this function calculates the ppo loss function
        def ppo_loss(
            model_params,
            mb_obs,
            mb_logprobs,
            mb_actions,
            mb_dones,
            mb_advantages,
            mb_returns,
            mb_values,
            init_lstm_state,
        ):
            logits, newvalue, _ = self.agent.apply(  # pyright: ignore
                model_params, mb_obs, init_lstm_state, mb_dones
            )
            probs = jax.nn.softmax(logits)
            newlogprobs = jnp.log(probs)
            entropy = -jnp.sum(probs * newlogprobs, axis=-1)
            # basically newlogprobs[action], where action tells us what to take in the last dimension
            newlogprob = jnp.take_along_axis(
                newlogprobs, jnp.expand_dims(mb_actions.astype(int), axis=-1), axis=-1
            ).squeeze(-1)
            logratio = newlogprob - mb_logprobs
            ratio = jnp.exp(logratio)

            # calculate approx kl
            # old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()

            if self.norm_adv:
                if len(mb_advantages) == 1:
                    mb_advantages = 0.0
                else:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jax.lax.clamp(
                1 - self.clip_coef, ratio, 1 + self.clip_coef
            )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = jnp.squeeze(newvalue, axis=-1)  # pyright: ignore

            if self.clip_vloss:
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + jax.lax.clamp(
                    -self.clip_coef, newvalue - mb_values, self.clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
            return loss, (
                pg_loss,
                v_loss,
                entropy_loss,
                jax.lax.stop_gradient(approx_kl),
            )

        grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

        # Optimize the policy and value network

        envsperbatch = self.num_envs // self.num_minibatches
        # environment indices
        envinds = jnp.arange(self.num_envs)

        def process_agent(agent_idx, model_param, opt_state, rng):
            b_obs = obs[:, agent_idx]
            b_logprobs = logprobs[:, agent_idx]
            b_actions = actions[:, agent_idx]
            b_dones = dones[:, agent_idx]
            b_advantages = advantages[:, agent_idx]
            b_returns = returns[:, agent_idx]
            b_values = values[:, agent_idx]

            def do_epoch(carry, epoch):
                rng, model_param, optimizer_state = carry
                rng, _rng = jax.random.split(rng)
                shuffled_envinds = jax.random.permutation(_rng, envinds)

                def do_minibatch(carry, start):
                    model_param, optimizer_state = carry

                    mbenvinds = jax.lax.dynamic_slice(
                        shuffled_envinds, (start,), (envsperbatch,)
                    )

                    mb_obs = b_obs[:, mbenvinds]
                    mb_logprobs = b_logprobs[:, mbenvinds]
                    mb_actions = b_actions[:, mbenvinds]
                    mb_dones = b_dones[:, mbenvinds]
                    mb_advantages = b_advantages[:, mbenvinds]
                    mb_returns = b_returns[:, mbenvinds]
                    mb_values = b_values[:, mbenvinds]
                    init_lstm_state = jax.tree.map(
                        lambda state: state[agent_idx, mbenvinds], init_lstm_states
                    )

                    (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = grad_fn(
                        model_param,
                        mb_obs,
                        mb_logprobs,
                        mb_actions,
                        mb_dones,
                        mb_advantages,
                        mb_returns,
                        mb_values,
                        init_lstm_state,
                    )
                    updates, optimizer_state = self.optimizer.update(
                        grads, optimizer_state, model_param
                    )
                    model_param = optax.apply_updates(model_param, updates)
                    return (model_param, optimizer_state), (
                        loss,
                        pg_loss,
                        v_loss,
                        entropy_loss,
                        approx_kl,
                    )

                (
                    (model_param, optimizer_state),
                    (losses, pg_losses, v_losses, entropy_losses, approx_kl),
                ) = jax.lax.scan(
                    do_minibatch,
                    (model_param, optimizer_state),
                    jnp.arange(0, self.num_envs, envsperbatch),
                )

                return (rng, model_param, optimizer_state), (
                    losses,
                    pg_losses,
                    v_losses,
                    entropy_losses,
                    approx_kl,
                )

            (
                (_rng, model_param, opt_state),
                (losses, pg_losses, v_losses, entropy_losses, approx_kl),
            ) = jax.lax.scan(
                do_epoch, (rng, model_param, opt_state), jnp.arange(self.update_epochs)
            )
            last_epoch_loss = losses[-1].mean()
            last_approx_kl = approx_kl[-1].mean()
            last_pg_loss = pg_losses[-1].mean()
            last_v_loss = v_losses[-1].mean()
            last_entropy_loss = entropy_losses[-1].mean()
            return (
                model_param,
                opt_state,
                last_epoch_loss,
                last_pg_loss,
                last_v_loss,
                last_entropy_loss,
                last_approx_kl,
            )

        (
            model_params,
            opt_states,
            agent_loss,
            agent_pg_loss,
            agent_v_loss,
            agent_entropy_loss,
            last_approx_kl,
        ) = jax.vmap(process_agent)(
            jnp.arange(self.static_params.num_players),
            model_params,
            opt_states,
            jax.random.split(rng, self.static_params.num_players)
        )

        return (
            model_params,
            opt_states,
            next_lstm_states,
            next_obs,
            env_state,
            agent_loss,
            agent_pg_loss,
            agent_v_loss,
            agent_entropy_loss,
            rewards.mean(axis=(0, 2)),
            last_approx_kl,
        )

    def train(self, model_params=None):
        if model_params is None:
            dummy_obs = jnp.ones(
                (self.num_steps, self.num_envs) + self.observation_space.shape
            )
            dummy_lstm_state = (
                jnp.ones((self.num_envs, 32)),
                jnp.ones((self.num_envs, 32)),
            )
            dummy_done = jnp.ones((self.num_steps, self.num_envs))
            rng, _rng = jax.random.split(self.rng)
            model_params = jax.vmap(self.agent.init, in_axes=(0, None, None, None))(
                jax.random.split(_rng, self.static_params.num_players),
                dummy_obs,
                dummy_lstm_state,
                dummy_done,
            )
        else:
            rng = self.rng
        opt_states = jax.vmap(self.optimizer.init)(model_params)

        # Logger
        log = TrainLogger(self.env_params, self.static_params)

        # initialize environment
        rng, _rng = jax.random.split(rng)
        next_obs, env_state = self.reset_fn(
            jax.random.split(_rng, self.num_envs), self.env_params
        )
        next_lstm_states = (
            jnp.zeros((self.static_params.num_players, self.num_envs, 32)),
            jnp.zeros((self.static_params.num_players, self.num_envs, 32)),
        )
        for iteration in range(self.num_iterations):
            print("Iteration", iteration)
            rng, _rng = jax.random.split(rng)
            (
                model_params,
                opt_states,
                next_lstm_states,
                next_obs,
                env_state,
                agent_loss,
                agent_pg_loss,
                agent_v_loss,
                agent_entropy_loss,
                rewards,
                last_approx_kl,
            ) = self.train_some_episodes(
                _rng,
                iteration,
                model_params,
                opt_states,
                next_lstm_states,
                next_obs,
                env_state,
            )
            log.insert_stat(iteration, "loss", agent_loss)
            log.insert_stat(iteration, "reward", rewards)
            log.insert_stat(iteration, "kl", last_approx_kl)
            log.insert_stat(iteration, "pg_loss", agent_pg_loss)
            log.insert_stat(iteration, "v_loss", agent_v_loss)
            log.insert_stat(iteration, "entropy_loss", agent_entropy_loss)
            if iteration % 20 == 0:
                log.insert_model_snapshot(iteration, model_params)
            for agent in range(self.static_params.num_players):
                print("Agent", agent, "loss:", agent_loss[agent])
                print("Agent", agent, "PG loss:", agent_pg_loss[agent])
                print("Agent", agent, "value loss:", agent_v_loss[agent])
                print("Agent", agent, "entropy:", agent_entropy_loss[agent])
                print("Agent", agent, "reward:", rewards[agent])
                print("Agent", agent, "approx KL:", last_approx_kl[agent])
        return model_params, opt_states, log

    def run_one_episode(self, model_params):
        rng, _rng = jax.random.split(self.rng)
        next_obs, env_state = self.env.reset(_rng, self.env_params)
        next_done = jnp.zeros((self.static_params.num_players, 1), dtype=bool)
        states = []
        actions = []
        logits = []
        rewards = []
        next_lstm_states = (
            jnp.zeros((self.static_params.num_players, 1, 32)),
            jnp.zeros((self.static_params.num_players, 1, 32)),
        )

        def eval_agent(model_param, next_lstm_state, next_obs, next_done, rng):
            logits, value, next_lstm_state = self.agent.apply(  # pyright: ignore
                model_param, next_obs, next_lstm_state, next_done
            )
            action = jax.random.categorical(rng, logits).squeeze()
            return action, logits, next_lstm_state

        eval_fn = jax.jit(jax.vmap(eval_agent))
        while not jnp.all(next_done):
            rng, _rng = jax.random.split(rng)
            states.append(env_state)
            agent_actions, agent_logits, next_lstm_states = eval_fn(
                model_params,
                next_lstm_states,
                next_obs,
                next_done,
                jax.random.split(_rng, self.static_params.num_players),
            )
            next_obs, env_state, reward, next_done, _info = self.env.step(
                _rng, env_state, agent_actions, self.env_params
            )
            actions.append(agent_actions)
            logits.append(agent_logits)
            rewards.append(reward)
        return states, actions, logits, rewards


if __name__ == "__main__":
    metacontroller = ClassicMetaController(
        static_parameters=StaticEnvParams(num_players=4),
        num_envs=128,
        num_minibatches=2,
        num_steps=200,
        num_iterations=25,
        update_epochs=5,
        anneal_lr=False,
        learning_rate=2.5e-4,
        max_grad_norm=1.0,
    )
    params, opt_states, log = metacontroller.train()
    # states, actions, logits, rewards = metacontroller.run_one_episode(params)
    # replay_episode(states, actions, 4, 0)
    # n_steps = 10
    # n_envs = 8
    # dummy_obs = jnp.ones((n_steps, n_envs, 1346))
    # dummy_lstm_state = (jnp.ones((n_envs, 32)), jnp.ones((n_envs, 32)))
    # dummy_done = jnp.ones((n_steps, n_envs))
    # print(
    #     CraftaxAgent(17).tabulate(
    #         jax.random.PRNGKey(randrange(2**31)),
    #         dummy_obs,
    #         dummy_lstm_state,
    #         dummy_done,
    #     )
    # )
    # model = CraftaxAgent(17)
    # rng_key = jax.random.PRNGKey(0)
    # params = model.init(rng_key, dummy_obs, dummy_lstm_state, dummy_done)
    # print(params)
    # breakpoint()
