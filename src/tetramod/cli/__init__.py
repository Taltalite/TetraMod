from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib import import_module

from tetramod import __version__


_COMMANDS = {
    "train": "tetramod.cli.train",
    "basecaller": "tetramod.cli.basecaller",
}


def build_parser():
    parser = ArgumentParser(
        "tetramod",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        description="valid commands",
        help="additional help",
        dest="command",
    )
    subparsers.required = True

    for command, module_name in _COMMANDS.items():
        module = import_module(module_name)
        subparser = subparsers.add_parser(command, parents=[module.argparser()])
        subparser.set_defaults(func=module.main)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
