from .config import RLPostTrainConfig, load_config


def run_toy_rl(cfg: RLPostTrainConfig):
    from .trainer import run_toy_rl as _run_toy_rl

    return _run_toy_rl(cfg)


def run_metric_only_eval(cfg: RLPostTrainConfig):
    from .trainer import run_metric_only_eval as _run_metric_only_eval

    return _run_metric_only_eval(cfg)


__all__ = ["RLPostTrainConfig", "load_config", "run_metric_only_eval", "run_toy_rl"]
