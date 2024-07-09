from craftax.craftax_classic.envs.craftax_state import EnvState
from craftax.craftax_classic.constants import *
from jax import Array


def compute_score(state: EnvState, done: Array):
    achievements = state.achievements * done.reshape(-1, 1) * 100.0
    info = {}
    for achievement in Achievement:
        name = f"Achievements/{achievement.name.lower()}"
        info[name] = achievements[achievement.value]
    # Geometric mean with an offset of 1%
    info["score"] = jnp.exp(jnp.mean(jnp.log(1 + achievements))) - 1.0
    return info
