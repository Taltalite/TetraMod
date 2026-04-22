import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


class TetramodCliSmokeTest(unittest.TestCase):
    def test_cli_builds_expected_commands(self):
        from tetramod.cli import build_parser

        parser = build_parser()
        actions = [action for action in parser._actions if action.dest == "command"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(set(actions[0].choices), {"train", "basecaller"})

    def test_subcommand_help_parses_without_bonito_runtime_imports(self):
        from tetramod.cli import main

        for command in ("train", "basecaller"):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    main([command, "--help"])
            self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
