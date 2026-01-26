from argparse import ArgumentParser, Namespace
from pathlib import Path
from omegaconf import OmegaConf


def convert_omega_to_namespace(config: OmegaConf) -> Namespace:
    """Convert omega DictConfig to Namespace for simplicity"""
    namespace_args = Namespace()
    # Ensure the input is DictConf class
    omega_dict_config = OmegaConf.structured(config)
    config_dict = OmegaConf.to_container(omega_dict_config, resolve=True)
    namespace_args.__dict__ = config_dict
    return namespace_args


def post_process_args(args):

    # Check config
    for attr in ['input_scp', 'output_dir', 'transforms']:
        assert hasattr(args, attr), f'Missing {attr} in config file'
        assert getattr(args, attr), f'{attr} in config file in None'

    args.output_dir = Path(args.output_dir)
    return args


def parse_args():

    omega_parser = ArgumentParser('Yaml configuration interface')
    omega_parser.add_argument(
        'overrides', nargs='*',
        help='Any key=value arguments to override config '
        'values (use dots for.nested=overrides)',
    )

    omega_parser.add_argument(
        '--conf', '-c', action='append',
        help='Yaml configuration files',
    )

    omega_cli_args, _ = omega_parser.parse_known_args()

    assert omega_cli_args.conf is not None, "Configure file is None"
    defalt_conf = OmegaConf.create({"meta": None, "mono": True, "archive": False})
    conf_list = [defalt_conf]
    conf_list += [OmegaConf.load(conf_file) for conf_file in omega_cli_args.conf]
    conf_list.append(OmegaConf.from_dotlist(omega_cli_args.overrides))
    config = OmegaConf.merge(*conf_list)
    args = convert_omega_to_namespace(config)
    args = post_process_args(args)
    return args
