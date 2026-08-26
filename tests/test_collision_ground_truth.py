"""What changes once the snapshot carries the whole decoded floor.

Before this, walkability was accumulated from 9x10 windows and the store could
not forget: "a tile seen passable once stays passable". A scripted player given
a perfect map of Mt Moon 1F still could not walk an 89-step leg, so the point of
these tests is the *precedence* rules, not the decoding.
"""

from pokemon_agent.capabilities import collision_basis, collision_from


def snapshot(*, terrain=None, truth=None, origin=(0, 0)):
    out = {"window_top_left": {"x": origin[0], "y": origin[1]}}
    if terrain is not None:
        out["terrain"] = terrain
    if truth is not None:
        out["map_terrain"] = truth
    return out


def floor(walkable, *, width=4, height=4):
    return {"width": width, "height": height, "walkable": set(walkable), "tile_ids": {}}


def test_ground_truth_replaces_the_store_rather_than_joining_it():
    """The store's defect was that it could not retract, not that it was thin.

    A stale walkable tile that ground truth calls solid has to disappear, or the
    false doorways survive the fix that was meant to remove them.
    """
    explored = {"walkable": {(0, 0), (3, 3)}, "seen": {(0, 0), (3, 3)}}

    got = collision_from(snapshot(truth=floor({(0, 0), (1, 0)})), explored)

    assert (3, 3) not in got["walkable"], "the store's stale tile is gone"
    assert got["walkable"] == {(0, 0), (1, 0)}


def test_a_decoded_floor_has_no_unexplored_ground():
    """`seen` covering everything is what lets a refusal say "wall", not "unlooked-at"."""
    got = collision_from(snapshot(truth=floor({(1, 1)}, width=2, height=2)), None)

    assert got["seen"] == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert got["ground_truth"] is True


def test_a_blocked_frame_tile_no_longer_deletes_terrain():
    """An NPC is not a wall.

    The frame blocks whatever a sprite stands on. Deleting that tile is how a
    trainer parked in a corridor became a permanent hole in the map.
    """
    truth = floor({(0, 0), (1, 0), (2, 0)})
    # The frame says (1,0) is blocked -- something is standing there.
    frame = [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

    got = collision_from(snapshot(terrain=frame, truth=truth), None)

    assert (1, 0) in got["walkable"], "terrain survives whatever is standing on it"


def test_the_frame_can_still_add_a_door_the_blockset_does_not_know():
    """Warp carpets and doors are walkable without being in the collision list.

    Measured: six such tiles on Route 4 that the player physically walked and
    the blockset calls solid. Positive frame evidence still counts.
    """
    got = collision_from(snapshot(terrain=[[1, 1]], truth=floor(set())), None)

    assert {(0, 0), (1, 0)} <= got["walkable"]


def test_without_ground_truth_nothing_changes():
    """The old precedence has to survive for maps a frame cannot decode."""
    explored = {"walkable": {(0, 0), (1, 0)}, "seen": {(0, 0), (1, 0)}}
    frame = [[1, 0]]  # the frame overrules the store's (1,0)

    got = collision_from(snapshot(terrain=frame), explored)

    assert got["walkable"] == {(0, 0)}
    assert got["ground_truth"] is False


def test_the_basis_line_says_which_map_the_answer_came_from():
    """ "Unreachable" from ground truth is a fact; from a mosaic of screens it is a guess."""
    truth_basis = collision_basis(collision_from(snapshot(truth=floor({(0, 0)})), None))
    window_basis = collision_basis(collision_from(snapshot(terrain=[[1]]), None))

    assert "real terrain" in truth_basis and "decoded" in truth_basis
    assert "window" in window_basis
