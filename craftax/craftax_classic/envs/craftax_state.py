from dataclasses import dataclass
from typing import Tuple, Any

import jax.random
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Inventory:
    """All items in inventory are int arrays"""

    @staticmethod
    def generate_inventory(num_agents: int) -> "Inventory":
        # we can do this since arrays are immutable
        zero_array = jnp.zeros(num_agents, dtype=int)
        return Inventory(*(zero_array for _ in range(12)))

    wood: jnp.ndarray
    stone: jnp.ndarray
    coal: jnp.ndarray
    iron: jnp.ndarray
    diamond: jnp.ndarray
    sapling: jnp.ndarray
    wood_pickaxe: jnp.ndarray
    stone_pickaxe: jnp.ndarray
    iron_pickaxe: jnp.ndarray
    wood_sword: jnp.ndarray
    stone_sword: jnp.ndarray
    iron_sword: jnp.ndarray


@struct.dataclass
class Mobs:
    """Represents every occurrence of one type of Mob"""

    position: jnp.ndarray
    """Positions of each mob with shape (n,2)"""
    health: jnp.ndarray
    """An int array representing the mob healths"""
    mask: jnp.ndarray
    """A boolean array, representing who's still alive"""
    attack_cooldown: jnp.ndarray
    """Actually an int array"""


@struct.dataclass
class EnvState:
    map: jnp.ndarray
    """2D int array of tiles"""
    mob_map: jnp.ndarray
    """2D boolean array"""

    # After refactoring, needs to be (n, 2)
    player_position: jnp.ndarray
    """Integer (x, y) coordinates of players. Has shape (n, 2)"""
    player_direction: jnp.ndarray
    """Direction of players as ints, with shape (n,)"""

    # Intrinsics - player stats range from 0 to 9
    player_health: jnp.ndarray
    """Int array representing player healths"""
    player_food: jnp.ndarray
    """Int array representing player food"""
    player_drink: jnp.ndarray
    """Int array representing player water satisfaction level"""
    player_energy: jnp.ndarray
    """Int array representing player energy"""
    is_sleeping: jnp.ndarray
    """Boolean array representing whether the player is sleeping"""

    # Second order intrinsics
    player_recover: jnp.ndarray
    """Float array representing player recovery level"""
    player_hunger: jnp.ndarray
    """Float array representing player hunger level"""
    player_thirst: jnp.ndarray
    """Float array representing player thirst level"""
    player_fatigue: jnp.ndarray
    """Float array representing player fatigue level"""

    inventory: Inventory
    """Represents inventory of all players. Refer to `Inventory`"""

    # Mobs
    zombies: Mobs
    cows: Mobs
    skeletons: Mobs
    arrows: Mobs
    arrow_directions: jnp.ndarray

    growing_plants_positions: jnp.ndarray
    """(num_plants, 2) array"""
    growing_plants_age: jnp.ndarray
    """(num_plants,) array"""
    growing_plants_mask: jnp.ndarray
    """(num_plants,) boolean array"""

    light_level: jnp.ndarray

    achievements: jnp.ndarray
    """(n, n_unique_achievements(22)) bool array representing achievements. Refer to `Achievement` in constants"""

    state_rng: Any

    timestep: int

    fractal_noise_angles: tuple[int | None, int | None, int | None, int | None] = (
        None,
        None,
        None,
        None,
    )
    """Honestly idk why this is in state rather than params"""


@struct.dataclass
class EnvParams:
    max_timesteps: int = 10000
    day_length: int = 300

    always_diamond: bool = True

    zombie_health: int = 5
    cow_health: int = 3
    skeleton_health: int = 3

    mob_despawn_distance: int = 14

    spawn_cow_chance: float = 0.3
    spawn_zombie_base_chance: float = 0.02
    spawn_zombie_night_chance: float = 0.1
    spawn_skeleton_chance: float = 0.05

    fractal_noise_angles: tuple[int | None, int | None, int | None, int | None] = (
        None,
        None,
        None,
        None,
    )

    god_mode: bool = False
    """Turn this on to not die lol"""


@struct.dataclass
class StaticEnvParams:
    num_players: int = 3
    map_size: Tuple[int, int] = (64, 64)

    # Mobs
    max_zombies: int = 3
    max_cows: int = 10
    max_growing_plants: int = 10
    max_skeletons: int = 2
    max_arrows: int = 3
