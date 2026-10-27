import pytest
from ieee14_fdia.cli import main


def test_installed_command_help_is_renderable(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "generate-attacks" in help_text and "import-legacy" in help_text
