"""
Glue between Hydra and the Pydantic models. Each script registers its model's defaults with Hydra
(register_defaults), composes them with its YAML in conf/ and the command line, validates the result (from_hydra),
then checks the configured paths (check_paths_exist).
"""

import os
from typing import Any, Dict, Type, TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from pydantic import BaseModel

# Fields checked by check_paths_exist, on whichever model has them
PATH_FIELDS = ("network_path", "expert_data_path", "curriculum_csv_path", "data_dir", "checkpoint", "stockfish_path")

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def register_defaults(name: str, model: Type[BaseModel]) -> None:
    """
    Store the model's defaults in Hydra's ConfigStore as a plain dict, so YAML files can list them under `defaults:`
    and command-line overrides of any field work without a `+` prefix. Required fields become MISSING (`???`).
    """
    defaults: Dict[str, Any] = {}
    for field_name, field_info in model.model_fields.items():
        if field_info.is_required():
            defaults[field_name] = MISSING
        else:
            value = field_info.get_default(call_default_factory=True)
            defaults[field_name] = value.model_dump() if isinstance(value, BaseModel) else value
    ConfigStore.instance().store(name=name, node=defaults)


def from_hydra(model: Type[ConfigT], hydra_config: DictConfig) -> ConfigT:
    """Validate a composed Hydra config. A required field left unset fails here with Hydra's MissingMandatoryValue."""
    return model.model_validate(OmegaConf.to_container(hydra_config, resolve=True, throw_on_missing=True))


def check_paths_exist(config: BaseModel) -> None:
    """
    Raise FileNotFoundError listing every configured path that doesn't exist. Not a model validator: the default
    network_path is absent on fresh clones and in CI, and TrainingConfig() must still construct there.
    """
    values = config.model_dump()
    missing = [name for name in PATH_FIELDS if values.get(name) is not None and not os.path.exists(values[name])]
    if missing:
        hint = " (scripts/download_weights.sh fetches the default network)" if "network_path" in missing else ""
        listed = ", ".join(f"{name}={values[name]}" for name in missing)
        raise FileNotFoundError(f"Configured paths not found: {listed}{hint}")
