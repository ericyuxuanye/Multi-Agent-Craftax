from typing import Any

from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams


class TrainLogger:
    env_params: EnvParams
    static_env_params: StaticEnvParams
    model_snapshots: list[tuple[int, Any]]
    stats: dict[str, list[tuple[int, Any]]]

    def __init__(
        self, env_params: EnvParams, static_env_params: StaticEnvParams
    ) -> None:
        self.env_params = env_params
        self.static_env_params = static_env_params
        self.model_snapshots = []
        self.stats = {}

    def insert_model_snapshot(self, iteration: int, model: Any) -> None:
        self.model_snapshots.append((iteration, model))

    def insert_stat(self, iteration: int, key: str, data: Any) -> None:
        """
        Inserts statistic. Note that the data should be an array containing data for all agents
        """
        if key in self.stats:
            self.stats[key].append((iteration, data))
        else:
            self.stats[key] = [(iteration, data)]

    def __repr__(self) -> str:
        return f"Train Log with {len(self.model_snapshots)} snapshots and {len(next(iter(self.stats.values())))} items of {len(self.stats)} different statistics"
