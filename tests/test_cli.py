from cell_automata.cli import cli
from cell_automata.simulation import run_simulation


def test_headless_cli_writes_a_history(monkeypatch, tmp_path, capsys) -> None:
    output_path = tmp_path / "run" / "history.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cellular-automata",
            "--headless",
            "--width",
            "8",
            "--height",
            "4",
            "--generations",
            "3",
            "--seed",
            "7",
            "--output",
            str(output_path),
        ],
    )

    cli()

    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8").startswith(
        "Simulation Step,Minimal Target Distance"
    )
    assert "generations=" in capsys.readouterr().out


def test_headless_simulation_rejects_a_negative_generation_limit() -> None:
    try:
        run_simulation(max_generations=-1)
    except ValueError as error:
        assert str(error) == "max_generations cannot be negative"
    else:
        raise AssertionError("negative generation limit was accepted")
