from typing import Sequence
import numpy as np

from craftax.craftax_classic.constants import Action
from craftax.craftax_classic.envs.craftax_state import EnvState
from craftax.craftax_classic.renderer import render_craftax_pixels

import pygame
import cv2


def replay_episode(
    states: Sequence[EnvState], actions: Sequence[Sequence[int]], num_players: int, player: int = 0
) -> None:
    pygame.init()
    pygame.key.set_repeat(250, 75)
    screen_surface = pygame.display.set_mode((576, 576))
    state_idx = 0
    done = False
    clock = pygame.time.Clock()
    while not done:
        # Render
        screen_surface.fill((0, 0, 0))

        pixels = render_craftax_pixels(
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
    pygame.quit()

def render_video(states: Sequence[EnvState], player: int, filename: str, fps=20) -> None:
    """
    Replays agent and saves to file
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # pyright: ignore
    out = cv2.VideoWriter('output1.mp4', fourcc, 15.0, (576, 576))

    for state in range(len(states)):
        data = render_craftax_pixels(state, 64, 4, player)
        frame = np.asarray(data, dtype=np.uint8)[..., ::-1]
        out.write(frame)

    out.release()
