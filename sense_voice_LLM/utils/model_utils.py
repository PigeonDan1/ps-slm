from utils.dataset_utils import load_module_from_py_file
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def get_custom_model_factory(model_config, logger):
    custom_model_path = model_config.get(
        "file", None
    )
    if custom_model_path is None:
        raise ValueError(f"must set correct model path")

    if ":" in custom_model_path:
        module_path, func_name = custom_model_path.split(":")
    else:
        module_path, func_name = custom_model_path, "model_factory"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")
    
    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)
    except AttributeError as e:
        logger.info(f"It seems like the given method name ({func_name}) is not present in the model .py file ({module_path.as_posix()}).")
        raise e
    

def print_model_size(model, config, rank: int = 0) -> None:
    """
    log model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"--> {config.model_name} has {total_params / 1e6} Million params\n")


def print_module_size(module, module_name, rank: int = 0) -> None:
    """
    Print module name, the number of trainable parameters and initialization time.

    Args:
        module: The PyTorch module.
        module_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Module {module_name}")
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"--> {module_name} has {total_params / 1e6} Million params\n")