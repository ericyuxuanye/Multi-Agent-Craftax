import functools
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import chex
from gymnax.environments import environment, spaces
from jax import lax

from craftax.craftax_classic.constants import *
from craftax.craftax_classic.envs.common import compute_score
from craftax.craftax_classic.envs.craftax_state import (
    EnvParams,
    EnvState,
    StaticEnvParams,
)
from craftax.craftax_classic.game_logic import (
    are_players_alive,
    craftax_step,
    is_game_over,
)
from craftax.craftax_classic.renderer import (
    render_craftax_symbolic,
    render_others_inventories,
)
from craftax.craftax_classic.world_gen import generate_world
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset


def get_map_obs_shape(observe_others: bool = False, num_players: None | int = None):
    """
    Returns the map shape of the symbolic environment.
    If observe_others is true, num_players is needed.
    """
    if observe_others and num_players is None:
        raise Exception("num_players must not be None if observe_others")
    num_mobs = 4 if observe_others else 5  # pyright: ignore
    num_blocks = len(BlockType)

    num_dropped_item_types = 6  # one channel per droppable tool type
    return OBS_DIM[0], OBS_DIM[1], num_blocks + num_mobs + num_dropped_item_types


def get_flat_map_obs_shape(observe_others: bool = False, num_players: None | int = None):
    """
    Returns the flattened map observation shape. If observe_others is true,
    num_players is needed.
    """
    map_obs_shape = get_map_obs_shape(observe_others, num_players)
    return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]


def get_inventory_obs_shape():
    inv_size = 12
    num_intrinsics = 4
    light_level = 1
    is_sleeping = 1
    is_alive = 1
    direction = 4

    return inv_size + num_intrinsics + light_level + is_sleeping + direction + is_alive

def get_player_data_obs_shape(static_params):
    """
    Returns the observation shape of observing other players
    """
    map_size = OBS_DIM[0] * OBS_DIM[1]
    inventory = 12
    intrinsics = 5
    directions = 4

    return (static_params.num_players - 1, map_size + inventory + intrinsics + directions)



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

    def discount(self, state, params) -> jax.Array:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(
            self.is_terminal(state, params),
            jnp.zeros(len(state.player_position), dtype=float),
            jnp.ones(len(state.player_position), dtype=float),
        )

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        def _render_player(player):
            return render_craftax_symbolic(state, player)

        all_pixels = jax.vmap(_render_player)(
            jnp.arange(self.static_env_params.num_players)
        )
        # pixels = render_craftax_symbolic(state)
        return all_pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return 23

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(23)

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

    def discount(self, state, params) -> jax.Array:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(
            self.is_terminal(state, params),
            jnp.zeros(len(state.player_position), dtype=float),
            jnp.ones(len(state.player_position), dtype=float),
        )

    # This is copied from gymnax Environment because the done needs to be jnp.all
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams] = None,
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
        state = jax.tree_util.tree_map(
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
        all_pixels = jax.vmap(render_craftax_symbolic, in_axes=(None, 0))(
            state, jnp.arange(self.static_env_params.num_players)
        )
        # pixels = render_craftax_symbolic(state)
        return all_pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-v1"

    @property
    def num_actions(self) -> int:
        return 23

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(23)

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


class CraftaxClassicSymbolicEnvShareStats(CraftaxClassicSymbolicEnv):
    """
    Note: This is not compatible with CraftaxClassicSymbolicEnv.
    Needs special handling.

    The observation returns a tuple containing player observations,
    and inventory and health of other players
    """

    def __init__(self, static_env_params: StaticEnvParams | None = None):
        super().__init__(static_env_params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-ShareStats-v1"

    def observation_space(self, params: EnvParams) -> spaces.Tuple:
        flat_map_obs_shape = get_flat_map_obs_shape(True, self.static_env_params.num_players)
        inventory_obs_shape = get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape

        player_observations = spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )

        other_player_status = spaces.Box(
            0, 1, get_player_data_obs_shape(self.static_env_params), dtype=jnp.float16
        )

        return spaces.Tuple([player_observations, other_player_status])

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> Tuple[
        Tuple[chex.Array, chex.Array],
        EnvState,
        jnp.ndarray,
        jnp.ndarray,
        dict[Any, Any],
    ]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        all_done = jnp.all(done)
        state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(all_done, x, y), state_re, state_st
        )
        obs = jax.tree_util.tree_map(
            lambda re, st: jax.lax.select(all_done, re, st), obs_re, obs_st
        )
        return obs, state, reward, done, info

    def get_obs(self, state: EnvState) -> Tuple[chex.Array, chex.Array]:
        """
        Returns a tuple
        """
        self_obs, player_obs = jax.vmap(render_craftax_symbolic, in_axes=(None, 0, None))(
            state, jnp.arange(self.static_env_params.num_players), True
        )
        return (self_obs, player_obs)


class CraftaxClassicSymbolicEnvShareStatsNoAutoReset(
    CraftaxClassicSymbolicEnvNoAutoReset
):
    """
    Note: This is not compatible with CraftaxClassicSymbolicEnv.
    Needs special handling.

    The observation returns a tuple containing player observations,
    and inventory and health of other players
    """

    def __init__(self, static_env_params: StaticEnvParams | None = None):
        super().__init__(static_env_params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-ShareStats-NoAutoReset-v1"

    def observation_space(self, params: EnvParams) -> spaces.Tuple:
        flat_map_obs_shape = get_flat_map_obs_shape(True, self.static_env_params.num_players)
        inventory_obs_shape = get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape

        player_observations = spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )

        other_player_status = spaces.Box(
            0, 1, get_player_data_obs_shape(self.static_env_params), dtype=jnp.float16
        )

        return spaces.Tuple([player_observations, other_player_status])

    def get_obs(self, state: EnvState) -> Tuple[chex.Array, chex.Array]:
        """
        Returns a tuple
        """
        self_obs, player_obs = jax.vmap(render_craftax_symbolic, in_axes=(None, 0, None))(
            state, jnp.arange(self.static_env_params.num_players), True
        )
        return (self_obs, player_obs)
