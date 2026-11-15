from dryrun.mjcf_station import layout_bricks


def test_explicit_twelve_brick_scene_is_seeded_and_bounded() -> None:
    sentence = "loose flat bricks, bowls on the sides, inspection camera in front"
    first = layout_bricks(1234, sentence, count=12)
    second = layout_bricks(1234, sentence, count=12)
    assert len(first) == 12
    assert [(b.x, b.y, b.yaw) for b in first] == [(b.x, b.y, b.yaw) for b in second]
    assert len({(b.x, b.y) for b in first}) == 12
    for brick in first:
        radius = ((brick.x - 0.0388) ** 2 + brick.y ** 2) ** 0.5
        assert 0.12 <= radius <= 0.24


def test_tier_defaults_keep_measured_ranges() -> None:
    sentences = {
        "loose": "loose flat bricks, bowls on the sides, inspection camera in front",
        "messy": "messy touching bricks, bowls on the sides, inspection camera in front",
        "pile": "dropped pile of bricks, bowls on the sides, inspection camera in front",
    }
    for tier, sentence in sentences.items():
        counts = [len(layout_bricks(seed, sentence)) for seed in range(8)]
        if tier == "loose":
            assert all(3 <= n <= 7 for n in counts)
        elif tier == "messy":
            assert all(5 <= n <= 9 for n in counts)
        else:
            assert all(6 <= n <= 10 for n in counts)
