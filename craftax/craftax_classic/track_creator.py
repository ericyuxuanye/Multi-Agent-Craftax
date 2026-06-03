from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from craftax.craftax_classic.game_logic import *


@struct.dataclass
class TrackedState:
    table_self_count: jnp.ndarray
    """Number of times used own table"""

    furnace_self_count: jnp.ndarray
    """Number of times used own furnace"""

    table_total_count: jnp.ndarray
    """Total number of times used any table"""

    furnace_total_count: jnp.ndarray
    """Total number of times used any any furnace"""

    table_map: jnp.ndarray
    """Keep track of last player that created the table"""

    furnace_map: jnp.ndarray
    """Keep track of last player that created the furnace"""


def initialize_tracked_state(static_params) -> TrackedState:
    """
    Initializes a TrackedState object that keeps track of which player placed the furnace
    or crafting table, and the count of each player and object
    """
    return TrackedState(
        table_self_count=jnp.zeros(static_params.num_players, dtype=jnp.int16),
        furnace_self_count=jnp.zeros(static_params.num_players, dtype=jnp.int16),
        table_total_count=jnp.zeros(static_params.num_players, dtype=jnp.int16),
        furnace_total_count=jnp.zeros(static_params.num_players, dtype=jnp.int16),
        table_map=jnp.full(static_params.map_size, -1, dtype=jnp.int16),
        furnace_map=jnp.full(static_params.map_size, -1, dtype=jnp.int16),
    )


def player_near_own_object(state, object_map):
    def _is_placed_by_player(loc_add):
        pos = state.player_position + loc_add
        in_bounds_x = jnp.logical_and(0 <= pos[:, 0], pos[:, 0] < state.map.shape[0])
        in_bounds_y = jnp.logical_and(0 <= pos[:, 1], pos[:, 1] < state.map.shape[1])
        is_in_bounds = jnp.logical_and(in_bounds_x, in_bounds_y)
        is_correct_player = object_map[pos[:, 0], pos[:, 1]] == jnp.arange(
            state.player_position.shape[0]
        )
        return jnp.logical_and(is_in_bounds, is_correct_player)

    return jnp.any(jax.vmap(_is_placed_by_player)(CLOSE_BLOCKS), axis=0)


def do_crafting_tracked(state, action, tracked_state: TrackedState):
    """
    Overrides do_crafting in game_logic where we track the player state
    """
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value)
    is_at_furnace = is_near_block(state, BlockType.FURNACE.value)

    # Check if player-made stuff is near
    player_crafting_table_near = player_near_own_object(state, tracked_state.table_map)
    player_furnace_near = player_near_own_object(state, tracked_state.furnace_map)

    new_achievements = state.achievements

    # Wood pickaxe
    can_craft_wood_pickaxe = state.inventory.wood >= 1

    is_crafting_wood_pickaxe = jnp.logical_and(
        action == Action.MAKE_WOOD_PICKAXE.value,
        jnp.logical_and(can_craft_wood_pickaxe, is_at_crafting_table),
    )

    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 1 * is_crafting_wood_pickaxe,
        wood_pickaxe=state.inventory.wood_pickaxe + 1 * is_crafting_wood_pickaxe,
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_WOOD_PICKAXE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_WOOD_PICKAXE.value],
            is_crafting_wood_pickaxe,
        )
    )

    # Update tracked state
    tracked_state = tracked_state.replace(  # pyright: ignore
        table_self_count=tracked_state.table_self_count
        + jnp.logical_and(player_crafting_table_near, is_crafting_wood_pickaxe),
        table_total_count=tracked_state.table_total_count + is_crafting_wood_pickaxe,
    )

    # Stone pickaxe
    can_craft_stone_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1, new_inventory.stone >= 1
    )
    is_crafting_stone_pickaxe = jnp.logical_and(
        action == Action.MAKE_STONE_PICKAXE.value,
        jnp.logical_and(can_craft_stone_pickaxe, is_at_crafting_table),
    )

    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_crafting_stone_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_stone_pickaxe,
        stone_pickaxe=new_inventory.stone_pickaxe + 1 * is_crafting_stone_pickaxe,
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_STONE_PICKAXE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_STONE_PICKAXE.value],
            is_crafting_stone_pickaxe,
        )
    )

    # Update tracked state
    tracked_state = tracked_state.replace(  # pyright: ignore
        table_self_count=tracked_state.table_self_count
        + jnp.logical_and(player_crafting_table_near, is_crafting_stone_pickaxe),
        table_total_count=tracked_state.table_total_count + is_crafting_stone_pickaxe,
    )

    # Iron pickaxe
    can_craft_iron_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1,
        jnp.logical_and(
            new_inventory.stone >= 1,
            jnp.logical_and(
                new_inventory.iron >= 1,
                new_inventory.coal >= 1,
            ),
        ),
    )
    is_crafting_iron_pickaxe = jnp.logical_and(
        action == Action.MAKE_IRON_PICKAXE.value,
        jnp.logical_and(
            can_craft_iron_pickaxe, jnp.logical_and(is_at_furnace, is_at_crafting_table)
        ),
    )

    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 1 * is_crafting_iron_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_iron_pickaxe,
        stone=new_inventory.stone - 1 * is_crafting_iron_pickaxe,
        coal=new_inventory.coal - 1 * is_crafting_iron_pickaxe,
        iron_pickaxe=new_inventory.iron_pickaxe + 1 * is_crafting_iron_pickaxe,
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_IRON_PICKAXE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_IRON_PICKAXE.value],
            is_crafting_iron_pickaxe,
        )
    )

    # Update tracked state
    tracked_state = tracked_state.replace(  # pyright: ignore
        table_self_count=tracked_state.table_self_count
        + jnp.logical_and(player_crafting_table_near, is_crafting_iron_pickaxe),
        table_total_count=tracked_state.table_total_count + is_crafting_iron_pickaxe,
        furnace_self_count=tracked_state.furnace_self_count
        + jnp.logical_and(player_furnace_near, is_crafting_iron_pickaxe),
        furnace_total_count=tracked_state.furnace_total_count
        + is_crafting_iron_pickaxe,
    )

    # Wood sword
    can_craft_wood_sword = new_inventory.wood >= 1
    is_crafting_wood_sword = jnp.logical_and(
        action == Action.MAKE_WOOD_SWORD.value,
        jnp.logical_and(can_craft_wood_sword, is_at_crafting_table),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_wood_sword,
        wood_sword=new_inventory.wood_sword + 1 * is_crafting_wood_sword,
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_WOOD_SWORD.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_WOOD_SWORD.value],
            is_crafting_wood_sword,
        )
    )

    # Update tracked state
    tracked_state = tracked_state.replace(  # pyright: ignore
        table_self_count=tracked_state.table_self_count + jnp.logical_and(player_crafting_table_near, is_crafting_wood_sword),
        table_total_count=tracked_state.table_total_count + is_crafting_wood_sword,
    )

    # Stone sword
    can_craft_stone_sword = jnp.logical_and(
        new_inventory.stone >= 1, new_inventory.wood >= 1
    )
    is_crafting_stone_sword = jnp.logical_and(
        action == Action.MAKE_STONE_SWORD.value,
        jnp.logical_and(can_craft_stone_sword, is_at_crafting_table),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_stone_sword,
        stone=new_inventory.stone - 1 * is_crafting_stone_sword,
        stone_sword=new_inventory.stone_sword + 1 * is_crafting_stone_sword,
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_STONE_SWORD.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_STONE_SWORD.value],
            is_crafting_stone_sword,
        )
    )

    # Update tracked state
    tracked_state = tracked_state.replace(  # pyright: ignore
        table_self_count=tracked_state.table_self_count + jnp.logical_and(player_crafting_table_near, is_crafting_stone_sword),
        table_total_count=tracked_state.table_total_count + is_crafting_stone_sword,
    )

    # Iron sword
    can_craft_iron_sword = jnp.logical_and(
        new_inventory.iron >= 1,
        jnp.logical_and(
            new_inventory.wood >= 1,
            jnp.logical_and(new_inventory.stone >= 1, new_inventory.coal >= 1),
        ),
    )
    is_crafting_iron_sword = jnp.logical_and(
        action == Action.MAKE_IRON_SWORD.value,
        jnp.logical_and(
            can_craft_iron_sword, jnp.logical_and(is_at_furnace, is_at_crafting_table)
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_iron_sword,
        iron=new_inventory.iron - 1 * is_crafting_iron_sword,
        stone=new_inventory.stone - 1 * is_crafting_iron_sword,
        coal=new_inventory.coal - 1 * is_crafting_iron_sword,
        iron_sword=new_inventory.iron_sword + 1 * is_crafting_iron_sword,
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_IRON_SWORD.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_IRON_SWORD.value],
            is_crafting_iron_sword,
        )
    )

    # Update tracked state
    tracked_state = tracked_state.replace(  # pyright: ignore
        table_self_count=tracked_state.table_self_count + jnp.logical_and(player_crafting_table_near, is_crafting_iron_sword),
        table_total_count=tracked_state.table_total_count + is_crafting_iron_sword,
        furnace_self_count=tracked_state.furnace_self_count + jnp.logical_and(player_furnace_near, is_crafting_iron_sword),
        furnace_total_count=tracked_state.furnace_total_count + is_crafting_iron_sword,
    )

    state = state.replace(
        inventory=new_inventory,
        achievements=new_achievements,
    )

    return state, tracked_state


def place_block_tracked(state, action, static_params, tracked_state: TrackedState):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]
    placing_block_in_bounds = in_bounds_vec(state, placing_block_position)
    placing_block_in_bounds = jnp.logical_and(
        placing_block_in_bounds,
        jnp.logical_not(is_in_mob_vec(state, placing_block_position)),
    )

    # Crafting table
    crafting_table_key_down = action == Action.PLACE_TABLE.value
    crafting_table_key_down_in_bounds = jnp.logical_and(
        crafting_table_key_down, placing_block_in_bounds
    )
    has_wood = state.inventory.wood >= 2
    is_placing_crafting_table = jnp.logical_and(
        crafting_table_key_down_in_bounds,
        jnp.logical_and(
            jnp.logical_not(is_in_wall_vec(state, placing_block_position)), has_wood
        ),
    )
    placed_crafting_table_block = jax.lax.select(
        is_placing_crafting_table,
        jnp.full((len(state.player_position),), BlockType.CRAFTING_TABLE.value),
        state.map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = state.map.at[
        placing_block_position[:, 0], placing_block_position[:, 1]
    ].set(placed_crafting_table_block)
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 2 * is_placing_crafting_table
    )
    new_achievements = state.achievements.at[:, Achievement.PLACE_TABLE.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.PLACE_TABLE.value],
            is_placing_crafting_table,
        )
    )

    # Update tracked state. Since we are adding new players, it's basically a noop otherwise
    new_players_table = jnp.where(
        is_placing_crafting_table, jnp.arange(static_params.num_players), 0
    )
    tracked_state = tracked_state.replace(  # pyright: ignore
        table_map=tracked_state.table_map.at[
            placing_block_position[:, 0], placing_block_position[:, 1]
        ].add(new_players_table)
    )

    # Furnace
    furnace_key_down = action == Action.PLACE_FURNACE.value
    furnace_key_down_in_bounds = jnp.logical_and(
        furnace_key_down, placing_block_in_bounds
    )
    has_stone = new_inventory.stone > 0
    is_placing_furnace = jnp.logical_and(
        furnace_key_down_in_bounds,
        jnp.logical_and(
            jnp.logical_not(is_in_wall_vec(state, placing_block_position)), has_stone
        ),
    )
    placed_furnace_block = jax.lax.select(
        is_placing_furnace,
        jnp.full((len(state.player_position),), BlockType.FURNACE.value),
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[
        placing_block_position[:, 0], placing_block_position[:, 1]
    ].set(placed_furnace_block)
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_placing_furnace
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_FURNACE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_FURNACE.value], is_placing_furnace
        )
    )

    # Update tracked state for furnace
    new_players_furnace = jnp.where(
        is_placing_furnace, jnp.arange(static_params.num_players), 0
    )
    tracked_state = tracked_state.replace(  # pyright: ignore
        furnace_map=tracked_state.furnace_map.at[
            placing_block_position[:, 0], placing_block_position[:, 1]
        ].add(new_players_furnace)
    )

    # Stone
    stone_key_down = action == Action.PLACE_STONE.value
    stone_key_down_in_bounds = jnp.logical_and(stone_key_down, placing_block_in_bounds)
    has_stone = new_inventory.stone > 0
    is_placing_on_valid_block = jnp.logical_or(
        state.map[placing_block_position[:, 0], placing_block_position[:, 1]]
        == BlockType.WATER.value,
        jnp.logical_not(is_in_wall_vec(state, placing_block_position)),
    )
    is_placing_stone = jnp.logical_and(
        stone_key_down_in_bounds,
        jnp.logical_and(is_placing_on_valid_block, has_stone),
    )
    placed_stone_block = jax.lax.select(
        is_placing_stone,
        jnp.full((len(state.player_position),), BlockType.STONE.value),
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[
        placing_block_position[:, 0], placing_block_position[:, 1]
    ].set(placed_stone_block)
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_placing_stone
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_STONE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_STONE.value], is_placing_stone
        )
    )

    # Plant
    sapling_key_down = action == Action.PLACE_PLANT.value
    sapling_key_down_in_bounds = jnp.logical_and(
        sapling_key_down, placing_block_in_bounds
    )
    has_sapling = new_inventory.sapling > 0
    is_placing_sapling = jnp.logical_and(
        sapling_key_down_in_bounds,
        jnp.logical_and(
            new_map[placing_block_position[:, 0], placing_block_position[:, 1]]
            == BlockType.GRASS.value,
            has_sapling,
        ),
    )
    placed_sapling_block = jax.lax.select(
        is_placing_sapling,
        jnp.full((len(state.player_position),), BlockType.PLANT.value),
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[
        placing_block_position[:, 0], placing_block_position[:, 1]
    ].set(placed_sapling_block)
    new_inventory = new_inventory.replace(
        sapling=new_inventory.sapling - 1 * is_placing_sapling
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_PLANT.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_PLANT.value], is_placing_sapling
        )
    )
    (
        new_growing_plants_positions,
        new_growing_plants_age,
        new_growing_plants_mask,
    ) = add_new_growing_plant(
        state, placing_block_position, is_placing_sapling, static_params
    )

    state = state.replace(
        map=new_map,
        inventory=new_inventory,
        achievements=new_achievements,
        growing_plants_positions=new_growing_plants_positions,
        growing_plants_age=new_growing_plants_age,
        growing_plants_mask=new_growing_plants_mask,
    )

    return state, tracked_state

def update_mobs_tracked(rng, state, params, static_params, tracked_state: TrackedState):
    old_crafting_table_map = state.map == BlockType.CRAFTING_TABLE
    old_furnace_map = state.map == BlockType.FURNACE
    state = update_mobs(rng, state, params, static_params)
    removed_crafting_table = jnp.logical_and(old_crafting_table_map, jnp.logical_not(state.map == BlockType.CRAFTING_TABLE))
    removed_furnace = jnp.logical_and(old_furnace_map, jnp.logical_not(state.map == BlockType.FURNACE))

    tracked_state = tracked_state.replace(  # pyright: ignore
        table_map = jnp.where(removed_crafting_table, -1, tracked_state.table_map),
        furnace_map = jnp.where(removed_furnace, -1, tracked_state.furnace_map),
    )

    return state, tracked_state


@partial(jax.jit, static_argnums=4)
def craftax_step_tracked(
    rng, state, actions, params, static_params, tracked_state: TrackedState
):
    init_achievements = state.achievements
    init_health = state.player_health
    init_food = state.player_food
    init_drink = state.player_drink
    init_energy = state.player_energy
    # dead players don't change their state
    was_dead = ~are_players_alive(state)

    # dead and sleeping players cannot do anything
    actions = jax.lax.select(
        was_dead | state.is_sleeping, jnp.full_like(actions, Action.NOOP.value), actions
    )

    # Two players cannot operate on same block
    rng, _rng = jax.random.split(rng)
    actions = break_ties(_rng, state, actions)

    # Crafting
    state, tracked_state = do_crafting_tracked(state, actions, tracked_state)

    # Interact (mining, attacking, eating plants, drinking water)
    rng, _rng = jax.random.split(rng)
    state = do_action(_rng, state, actions, static_params)

    # Placing
    state, tracked_state = place_block_tracked(
        state, actions, static_params, tracked_state
    )

    # Movement
    state = move_player(state, actions)

    # Mobs
    rng, _rng = jax.random.split(rng)
    state, tracked_state = update_mobs_tracked(_rng, state, params, static_params, tracked_state)

    rng, _rng = jax.random.split(rng)
    state = spawn_mobs(state, _rng, params, static_params)

    # Plants
    state = update_plants(state, static_params)

    # Intrinsics
    state = update_player_intrinsics(state, actions)

    # Cap inv
    state = cap_inventory(state)

    # Reward
    achievement_reward = (
        (state.achievements.astype(jnp.float32) - init_achievements.astype(jnp.float32))
        * params.achievement_weights
    ).sum(axis=1)
    health_reward = (state.player_health - init_health) * 0.1
    reward = achievement_reward + health_reward

    rng, _rng = jax.random.split(rng)

    state = state.replace(
        timestep=state.timestep + 1,
        light_level=calculate_light_level(state.timestep + 1, params),
        state_rng=_rng,
        player_health=jax.lax.select(
            params.god_mode | was_dead,
            jnp.full((static_params.num_players,), init_health),
            state.player_health,
        ),
        player_food=jax.lax.select(
            params.god_mode | was_dead,
            jnp.full((static_params.num_players,), init_food),
            state.player_food,
        ),
        player_drink=jax.lax.select(
            params.god_mode | was_dead,
            jnp.full((static_params.num_players,), init_drink),
            state.player_drink,
        ),
        player_energy=jax.lax.select(
            params.god_mode | was_dead,
            jnp.full((static_params.num_players,), init_energy),
            state.player_energy,
        ),
    )

    return state, tracked_state, reward
