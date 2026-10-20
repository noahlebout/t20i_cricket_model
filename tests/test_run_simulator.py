import pytest

from run_simulator import _parse_observed_delivery


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("4", (4, False, True, "four", None)),
        ("W", (0, True, True, "wicket", 0)),
        ("W:1", (1, True, True, "wicket+1", None)),
        ("WD:2:1", (2, False, False, "wide+2", 1)),
        ("NB:3:1", (3, False, False, "no_ball+3", 1)),
    ],
)
def test_parse_actual_delivery(entry, expected) -> None:
    assert _parse_observed_delivery(entry) == expected


@pytest.mark.parametrize("entry", ["WD:0", "NB:x", "WD:2:-1", "nonsense"])
def test_invalid_actual_delivery_is_rejected(entry) -> None:
    with pytest.raises(ValueError):
        _parse_observed_delivery(entry)


def test_parse_quit_command() -> None:
    assert _parse_observed_delivery("q") is None
