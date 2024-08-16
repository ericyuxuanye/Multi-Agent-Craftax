import functools
from typing import Any, Optional, Tuple, Union, override

import chex
from craftax.craftax_classic.constants import *
from craftax.craftax_classic.envs.common import compute_score
from craftax.craftax_classic.envs.craftax_state import (EnvParams, EnvState,
                                                        StaticEnvParams)
from craftax.craftax_classic.game_logic import craftax_step, is_game_over
from craftax.craftax_classic.renderer import render_craftax_symbolic
from craftax.craftax_classic.world_gen import generate_world
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from gymnax.environments import environment, spaces
from jax import lax


def get_map_obs_shape():
    num_mobs = 4
    num_blocks = len(BlockType)

    return OBS_DIM[0], OBS_DIM[1], num_blocks + num_mobs


def get_flat_map_obs_shape():
    map_obs_shape = get_map_obs_shape()
    return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]


def get_inventory_obs_shape():
    inv_size = 12
    num_intrinsics = 4
    light_level = 1
    is_sleeping = 1
    is_alive = 1
    direction = 4

    return inv_size + num_intrinsics + light_level + is_sleeping + direction + is_alive


class CraftaxClassicSymbolicEnvNoAutoReset(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: StaticEnvParams | None = None):
        super().__init__()

        if static_env_params is None:
            self.static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        state, reward = craftax_step(rng, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = compute_score(state, done)
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        pixels = render_craftax_symbolic(state)
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return 17

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(17)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )


class CraftaxClassicSymbolicEnv(environment.Environment):
    def __init__(self, static_env_params: StaticEnvParams | None = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: jnp.ndarray, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        state, reward = craftax_step(rng, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = compute_score(state, done)
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    @override
    def discount(self, state, params) -> jax.Array:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(
            self.is_terminal(state, params),
            jnp.zeros(len(state.player_position), dtype=float),
            jnp.ones(len(state.player_position), dtype=float)
        )

    # This is copied from gymnax Environment because the done needs to be jnp.all
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvState] = None,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        all_done = jnp.all(done)
        state = jax.tree_map(
            lambda x, y: jax.lax.select(all_done, x, y), state_re, state_st
        )
        obs = jax.lax.select(all_done, obs_re, obs_st)
        return obs, state, reward, done, info

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        def _render_player(player):
            return render_craftax_symbolic(state, player)
        all_pixels = jax.vmap(_render_player)(jnp.arange(self.static_env_params.num_players))
        # pixels = render_craftax_symbolic(state)
        return all_pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-v1"

    @property
    def num_actions(self) -> int:
        return 17

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(17)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )
