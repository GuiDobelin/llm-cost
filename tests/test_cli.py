import pytest

from llm_cost import __version__
from llm_cost.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_runs_without_arguments():
    assert main([]) == 0
