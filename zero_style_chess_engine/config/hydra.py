"""
Glue between Hydra and the Pydantic models. Each script registers its model's defaults with Hydra
(register_defaults), composes them with its YAML in conf/ and the command line, validates the result (from_hydra),
then checks the configured paths (check_paths_exist).
"""

import os
import shutil
from typing import Any, Dict, List, Optional, Type, TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from pydantic import BaseModel

# Fields checked by check_paths_exist, on whichever model has them
PATH_FIELDS = ("network_path", "expert_data_path", "curriculum_csv_path", "data_dir", "checkpoint", "engine_path")
# Path fields that also accept a program name found on $PATH
EXECUTABLE_FIELDS = ("engine_path",)

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def register_defaults(name: str, model: Type[BaseModel], group: Optional[str] = None) -> None:
    """
    Store the model's defaults in Hydra's ConfigStore as a plain dict, so YAML files can list them under `defaults:`
    and command-line overrides of any field work without a `+` prefix. Required fields become MISSING (`???`).
    :param group: config group to store them in, so a YAML file can pick one per key (`player@white: zero_style`)
    """
    defaults: Dict[str, Any] = {}
    for field_name, field_info in model.model_fields.items():
        if field_info.is_required():
            defaults[field_name] = MISSING
        else:
            value = field_info.get_default(call_default_factory=True)
            defaults[field_name] = value.model_dump() if isinstance(value, BaseModel) else value
    ConfigStore.instance().store(name=name, node=defaults, group=group)


def from_hydra(model: Type[ConfigT], hydra_config: DictConfig) -> ConfigT:
    """Validate a composed Hydra config. A required field left unset fails here with Hydra's MissingMandatoryValue."""
    return model.model_validate(OmegaConf.to_container(hydra_config, resolve=True, throw_on_missing=True))


def _missing_paths(values: Dict[str, Any], prefix: str = "") -> List[str]:
    """Dotted names (black.engine_path) of the path fields, at any depth, set to a path that doesn't exist"""
    missing = []
    for name, value in values.items():
        if isinstance(value, dict):
            missing += _missing_paths(value, f"{prefix}{name}.")
        elif name in PATH_FIELDS and value is not None and not os.path.exists(value):
            if not (name in EXECUTABLE_FIELDS and shutil.which(value)):
                missing.append(f"{prefix}{name}={value}")
    return missing


def check_paths_exist(config: BaseModel) -> None:
    """
    Raise FileNotFoundError listing every configured path that doesn't exist, nested configs included. Not a model
    validator: the default network_path is absent on fresh clones and in CI, and TrainingConfig() must still construct
    there.
    """
    missing = _missing_paths(config.model_dump())
    if missing:
        hint = (
            " (scripts/download_weights.sh fetches the default network)"
            if any(entry.split("=")[0].endswith("network_path") for entry in missing)
            else ""
        )
        raise FileNotFoundError(f"Configured paths not found: {', '.join(missing)}{hint}")
