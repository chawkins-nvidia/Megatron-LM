# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Public training entry points with CPU-safe lazy imports."""

from importlib import import_module

_EXPORT_MODULES = {
    "get_adlr_autoresume": "global_vars",
    "get_args": "global_vars",
    "get_model": "training",
    "get_one_logger": "global_vars",
    "get_signal_handler": "global_vars",
    "get_tensorboard_writer": "global_vars",
    "get_timers": "global_vars",
    "get_tokenizer": "global_vars",
    "get_train_valid_test_num_samples": "training",
    "get_wandb_writer": "global_vars",
    "initialize_megatron": "initialize",
    "is_last_rank": "utils",
    "pretrain": "training",
    "print_rank_0": "utils",
    "print_rank_last": "utils",
    "set_startup_timestamps": "training",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    """Load one public training symbol on first use."""

    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
