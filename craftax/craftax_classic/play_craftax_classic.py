import argparse
import sys
from typing import cast

import pygame

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax.craftax_state import StaticEnvParams
from craftax.craftax_classic.constants import (
    OBS_DIM,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
    BLOCK_PIXEL_SIZE_HUMAN,
)
from craftax.craftax_classic.envs.craftax_state import EnvParams
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.craftax_classic.game_logic import is_player_alive
from craftax.craftax_classic.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name

KEY_MAPPING = {
    pygame.K_q: Action.NOOP,
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
    pygame.K_SPACE: Action.DO,
    pygame.K_t: Action.PLACE_TABLE,
    pygame.K_TAB: Action.SLEEP,
    pygame.K_r: Action.PLACE_STONE,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_WOOD_SWORD,
    pygame.K_5: Action.MAKE_STONE_SWORD,
    pygame.K_6: Action.MAKE_IRON_SWORD,
}


class CraftaxRenderer:
    def __init__(self, env: CraftaxClassicSymbolicEnv, env_params, num_players, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.num_players = num_players
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self._player_font = pygame.font.SysFont("Arial", 24)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = jax.jit(render_craftax_pixels, static_argnums=(1,2))

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()
        # time.sleep(0.01)

    def render(self, env_state, player=0):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN, num_players=self.num_players, player=player)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        player_text_img = self._player_font.render(f"Player {player+1}", True, (255, 255, 255))

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 0))
        self.screen_surface.blit(player_text_img, dest=(0,0))

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state, player=0):
        if state.is_sleeping[player] and not is_player_alive(state)[player]:
            return Action.NOOP.value
        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    return KEY_MAPPING[event.key].value

        return None


def print_new_achievements(old_achievements, new_achievements):
    for player in range(old_achievements.shape[0]):
        for i in range(old_achievements.shape[1]):
            if old_achievements[player][i] == 0 and new_achievements[player][i] == 1:
                print(f"Player {player+1} achieved {Achievement(i).name} ({new_achievements[player].sum()}/{22})")


def main(args):
    env = CraftaxClassicSymbolicEnv()
    env_params: EnvParams = cast(EnvParams, env.default_params)

    if args.god:
        env_params = env_params.replace(god_mode=True)

    if args.players:
        env.static_env_params = env.static_env_params.replace(num_players=args.players)

    num_players = env.static_env_params.num_players
    print("Controls")
    for k, v in KEY_MAPPING.items():
        print(f"{pygame.key.name(k)}: {v.name.lower()}")

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    _, env_state = env.reset(_rng, env_params)

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN

    renderer = CraftaxRenderer(env, env_params, num_players, pixel_render_size=pixel_render_size)
    renderer.render(env_state)

    current_player = 0
    actions = jnp.zeros(num_players, dtype=int)

    step_fn = jax.jit(env.step)

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state, current_player)

        if action is not None:
            actions = actions.at[current_player].set(action)
            current_player += 1
            if current_player == num_players:
                rng, _rng = jax.random.split(rng)
                old_achievements = env_state.achievements
                obs, env_state, rewards, done, info = step_fn(
                    _rng, env_state, actions, env_params
                )
                new_achievements = env_state.achievements
                print_new_achievements(old_achievements, new_achievements)

                for player, reward in enumerate(rewards):
                    if reward > 0.01 or reward < -0.01:
                        print(f"Player {player+1} got reward: {reward}\n")
                # reset actions
                actions = jnp.zeros(num_players, dtype=int)
                # reset current player
                current_player = 0

        renderer.render(env_state, current_player)

        renderer.update()
        clock.tick(args.fps)


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--god", action="store_true")
    parser.add_argument("--players", type=int)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)


if __name__ == "__main__":
    entry_point()
