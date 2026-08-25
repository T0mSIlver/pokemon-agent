"""Image layout for the annotated frame the model navigates from.

The overlay is the agent's primary input, so every measurement here is load
bearing: the header band, the absolute-coordinate rulers, the tile grid, the
P/W/G glyphs, and the mini-map inset panel. Pure image work -- no emulator, no
paths, no state beyond what the caller passes in.
"""

from __future__ import annotations

from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from pokemon_agent.navigation import LiveNavigationSnapshot

JsonDict = dict[str, Any]


def measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    if not text:
        return 0, 0
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = str(text or "").split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if measure_text(draw, candidate, font=font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


# Mini-map inset palette. Same language as the main tile grid: unknown ground is
# near-black, passable green, walked dimmed green, wall red, warp purple, player
# cyan. Only the scale changes.
_INSET_MAX_WIDTH = 176
_INSET_MAX_HEIGHT = 320
_INSET_MIN_CELL = 3
_INSET_MAX_CELL = 10
_INSET_UNKNOWN = (9, 12, 19, 255)
_INSET_SEEN = (96, 220, 158, 255)
_INSET_WALKED = (26, 96, 62, 255)
_INSET_WALL = (176, 58, 58, 255)
_INSET_WARP = (213, 80, 255, 255)
_INSET_PLAYER = (55, 208, 255, 255)
_INSET_BORDER = (86, 102, 122, 255)


def normalise_map_grid(grid: Any) -> Optional[JsonDict]:
    """Coerce an explored-map grid payload into plain sets, or None if unusable."""
    if not isinstance(grid, dict):
        return None
    try:
        width = int(grid.get("width") or 0)
        height = int(grid.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or width > 512 or height > 512:
        return None

    layers: JsonDict = {"width": width, "height": height}
    for name in ("seen", "walkable", "walked", "warps"):
        tiles: set[tuple[int, int]] = set()
        for item in grid.get(name) or ():
            try:
                tile_x, tile_y = int(item[0]), int(item[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if 0 <= tile_x < width and 0 <= tile_y < height:
                tiles.add((tile_x, tile_y))
        layers[name] = tiles

    if not (layers["seen"] | layers["walkable"] | layers["walked"] | layers["warps"]):
        return None
    return layers


def render_map_inset(
    grid: JsonDict,
    *,
    player: Optional[tuple[int, int]],
) -> Image.Image:
    """Draw the whole explored map as a small image with the player marked."""
    width = int(grid["width"])
    height = int(grid["height"])
    cell = min(_INSET_MAX_WIDTH // width, _INSET_MAX_HEIGHT // height, _INSET_MAX_CELL)
    cell = max(cell, _INSET_MIN_CELL)

    map_width = width * cell
    map_height = height * cell
    border = 1
    inset = Image.new("RGBA", (map_width + (border * 2), map_height + (border * 2)), _INSET_BORDER)
    draw = ImageDraw.Draw(inset)
    draw.rectangle(
        (border, border, border + map_width - 1, border + map_height - 1),
        fill=_INSET_UNKNOWN,
    )

    seen = grid["seen"]
    walkable = grid["walkable"]
    walked = grid["walked"]
    warps = grid["warps"]
    for tile in seen | walkable | walked | warps:
        tile_x, tile_y = tile
        if tile in warps:
            fill = _INSET_WARP
        elif tile in walked:
            fill = _INSET_WALKED
        elif tile in walkable:
            fill = _INSET_SEEN
        else:
            fill = _INSET_WALL
        left = border + (tile_x * cell)
        top = border + (tile_y * cell)
        draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill=fill)

    if player is None:
        return inset
    player_x, player_y = int(player[0]), int(player[1])
    if not (0 <= player_x < width and 0 <= player_y < height):
        return inset

    # The player marker is the one pixel that has to survive downscaling and a
    # glance: full-width crosshair, dark halo, solid cyan block, white ring.
    blend = ImageDraw.Draw(inset, "RGBA")
    centre_x = border + (player_x * cell) + (cell // 2)
    centre_y = border + (player_y * cell) + (cell // 2)
    blend.line((border, centre_y, border + map_width - 1, centre_y), fill=(55, 208, 255, 120))
    blend.line((centre_x, border, centre_x, border + map_height - 1), fill=(55, 208, 255, 120))
    half = max(3, min(cell, 6))
    blend.rectangle(
        (centre_x - half - 2, centre_y - half - 2, centre_x + half + 2, centre_y + half + 2),
        fill=(7, 10, 16, 225),
    )
    blend.rectangle(
        (centre_x - half, centre_y - half, centre_x + half, centre_y + half),
        fill=_INSET_PLAYER,
    )
    blend.rectangle(
        (centre_x - half - 2, centre_y - half - 2, centre_x + half + 2, centre_y + half + 2),
        outline=(255, 255, 255, 240),
        width=1,
    )
    return inset


def render_navigation_overlay(
    image: Image.Image,
    snapshot: Optional[LiveNavigationSnapshot],
    *,
    objective: Optional[JsonDict] = None,
    goal: Optional[tuple[int, int]] = None,
    visited: Optional[set[tuple[int, int]]] = None,
    map_grid: Optional[JsonDict] = None,
) -> Image.Image:
    scale = 2
    frame = image.convert("RGBA").resize(
        (image.width * scale, image.height * scale),
        resample=Image.NEAREST,
    )
    font = ImageFont.load_default()
    measure_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    padding = 8
    line_height = max(measure_text(measure_draw, "Ag", font=font)[1], 11) + 3

    if not snapshot or not snapshot.width or not snapshot.height:
        canvas_width = frame.width + (padding * 2)
        wrap_width = max(40, canvas_width - (padding * 2))
        header_lines = [
            (line, (255, 255, 255, 255))
            for line in wrap_text(
                measure_draw,
                "Navigation overlay unavailable.",
                font=font,
                max_width=wrap_width,
            )
        ]
        header_lines.extend(
            (
                line,
                (165, 180, 196, 255),
            )
            for line in wrap_text(
                measure_draw,
                "No live collision window was captured for this frame.",
                font=font,
                max_width=wrap_width,
            )
        )
        header_height = padding + (len(header_lines) * line_height) + padding
        canvas = Image.new(
            "RGBA",
            (canvas_width, header_height + frame.height + padding),
            (7, 10, 16, 255),
        )
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, header_height), fill=(12, 17, 26, 235))
        canvas.alpha_composite(frame, (padding, header_height))
        draw.rectangle(
            (
                padding - 1,
                header_height - 1,
                padding + frame.width,
                header_height + frame.height,
            ),
            outline=(255, 138, 61, 220),
            width=1,
        )

        text_y = padding
        for line, fill in header_lines:
            draw.text((padding, text_y), line, fill=fill, font=font)
            text_y += line_height
        return canvas.convert("RGB")

    window_min_x = snapshot.window_top_left[0]
    window_max_x = snapshot.window_top_left[0] + snapshot.width - 1
    window_min_y = snapshot.window_top_left[1]
    window_max_y = snapshot.window_top_left[1] + snapshot.height - 1
    visited_locals: set[tuple[int, int]] = set()
    for tile_x, tile_y in visited or ():
        visited_local = snapshot.absolute_to_local(int(tile_x), int(tile_y))
        if visited_local is not None:
            visited_locals.add(visited_local)

    x_labels = [str(window_min_x + local_x) for local_x in range(snapshot.width)]
    y_labels = [str(window_min_y + local_y) for local_y in range(snapshot.height)]
    x_label_height = max(
        (measure_text(measure_draw, label, font=font)[1] for label in x_labels),
        default=0,
    )
    y_label_width = max(
        (measure_text(measure_draw, label, font=font)[0] for label in y_labels),
        default=0,
    )

    left_margin = y_label_width + (padding * 2)
    canvas_width = left_margin + frame.width + padding
    wrap_width = max(40, canvas_width - (padding * 2))
    pos = snapshot.player_position
    move_list = ", ".join(snapshot.valid_moves) or "none"
    objective_line = objective["summary"] if objective else "No objective"
    header_blocks = [
        (snapshot.map_name, (255, 255, 255, 255)),
        (
            f"Player ({pos[0]}, {pos[1]}) facing {snapshot.facing} | moves: {move_list}",
            (165, 180, 196, 255),
        ),
        (f"Objective: {objective_line}", (255, 214, 10, 255)),
        (
            (
                "Coords are absolute map tiles. "
                f"Columns show x={window_min_x}..{window_max_x}; "
                f"rows show y={window_min_y}..{window_max_y}. "
                "North is up: walk_up decreases y, walk_down increases y."
            ),
            (110, 230, 174, 255),
        ),
    ]
    if visited_locals:
        header_blocks.append(
            (
                "Dimmed tiles with a grey dot are ground you already walked "
                f"({len(visited_locals)} of {snapshot.width * snapshot.height} in view).",
                (165, 180, 196, 255),
            )
        )
    header_lines: list[tuple[str, tuple[int, int, int, int]]] = []
    for text, fill in header_blocks:
        for line in wrap_text(measure_draw, text, font=font, max_width=wrap_width):
            header_lines.append((line, fill))

    column_band_height = x_label_height + padding + 2
    header_height = padding + (len(header_lines) * line_height) + padding
    top_margin = header_height + column_band_height

    # Side panel: the whole explored map, drawn to the right of the game window
    # so it covers neither the frame nor the header. Absent grid, absent panel,
    # and the canvas is exactly what it has always been.
    normalised_grid = normalise_map_grid(map_grid)
    inset: Optional[Image.Image] = None
    panel_width = 0
    panel_title = ""
    panel_caption: list[str] = []
    if normalised_grid is not None:
        inset = render_map_inset(normalised_grid, player=pos)
        known = (
            normalised_grid["seen"]
            | normalised_grid["walkable"]
            | normalised_grid["walked"]
            | normalised_grid["warps"]
        )
        panel_title = "MINI-MAP: whole map so far"
        caption_width = max(inset.width, 132)
        caption_blocks = [
            (
                f"{normalised_grid['width']}x{normalised_grid['height']} tiles, "
                f"{len(known)} seen, {len(normalised_grid['walked'])} walked."
            ),
            (
                "Cyan box with crosshair is you. Near-black is map you have not seen. "
                "Green passable, dim green walked, red wall, purple warp."
            ),
        ]
        for block in caption_blocks:
            panel_caption.extend(wrap_text(measure_draw, block, font=font, max_width=caption_width))
        panel_content_width = max(
            inset.width,
            measure_text(measure_draw, panel_title, font=font)[0],
            max(
                (measure_text(measure_draw, line, font=font)[0] for line in panel_caption),
                default=0,
            ),
        )
        panel_width = panel_content_width + (padding * 2)

    canvas_height = top_margin + frame.height + padding
    if inset is not None:
        canvas_height = max(
            canvas_height,
            top_margin
            + line_height
            + inset.height
            + 4
            + (len(panel_caption) * line_height)
            + padding,
        )
    canvas = Image.new(
        "RGBA",
        (canvas_width + panel_width, canvas_height),
        (7, 10, 16, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header_height), fill=(12, 17, 26, 235))
    draw.rectangle((0, header_height, canvas.width, top_margin), fill=(12, 17, 26, 220))
    draw.rectangle((0, top_margin, left_margin, canvas.height), fill=(12, 17, 26, 220))
    if inset is not None:
        panel_left = canvas_width
        draw.rectangle(
            (panel_left, header_height, canvas.width, canvas.height),
            fill=(12, 17, 26, 220),
        )
        panel_y = top_margin
        draw.text(
            (panel_left + padding, panel_y), panel_title, fill=(255, 255, 255, 255), font=font
        )
        panel_y += line_height
        canvas.alpha_composite(inset, (panel_left + padding, panel_y))
        panel_y += inset.height + 4
        for line in panel_caption:
            draw.text((panel_left + padding, panel_y), line, fill=(165, 180, 196, 255), font=font)
            panel_y += line_height
    canvas.alpha_composite(frame, (left_margin, top_margin))
    draw.rectangle(
        (
            left_margin - 1,
            top_margin - 1,
            left_margin + frame.width,
            top_margin + frame.height,
        ),
        outline=(255, 138, 61, 220),
        width=1,
    )

    text_y = padding
    for line, fill in header_lines:
        draw.text((padding, text_y), line, fill=fill, font=font)
        text_y += line_height

    tile_width = frame.width / snapshot.width
    tile_height = frame.height / snapshot.height
    grid_line_width = max(1, scale)

    # Tiles whose centre already carries a letter (P, W, G); the walked dot is
    # skipped there so the glyph stays readable. The dimmed fill still shows.
    glyph_locals: set[tuple[int, int]] = {(4, 4)}
    if goal is not None:
        goal_glyph = snapshot.absolute_to_local(goal[0], goal[1])
        if goal_glyph is not None:
            glyph_locals.add(goal_glyph)
    for warp in snapshot.warps:
        if not isinstance(warp, dict):
            continue
        warp_x = warp.get("x")
        warp_y = warp.get("y")
        if warp_x is None or warp_y is None:
            continue
        warp_glyph = snapshot.absolute_to_local(int(warp_x), int(warp_y))
        if warp_glyph is not None:
            glyph_locals.add(warp_glyph)
    visited_dot_radius = max(2, scale * 2)

    for local_y, row in enumerate(snapshot.terrain):
        for local_x, tile in enumerate(row):
            left = int(left_margin + (local_x * tile_width))
            top = int(top_margin + (local_y * tile_height))
            right = int(left_margin + ((local_x + 1) * tile_width))
            bottom = int(top_margin + ((local_y + 1) * tile_height))
            walked = (local_x, local_y) in visited_locals
            if tile:
                fill = (13, 68, 41, 96) if walked else (24, 123, 73, 72)
                outline = (74, 156, 118, 190) if walked else (110, 230, 174, 190)
            else:
                fill = (108, 35, 35, 110) if walked else (180, 58, 58, 92)
                outline = (206, 96, 96, 200) if walked else (255, 120, 120, 200)
            draw.rectangle((left, top, right, bottom), outline=outline, fill=fill, width=1)
            if walked and (local_x, local_y) not in glyph_locals:
                dot_x = int(left_margin + ((local_x + 0.5) * tile_width))
                dot_y = int(top_margin + ((local_y + 0.5) * tile_height))
                draw.ellipse(
                    (
                        dot_x - visited_dot_radius,
                        dot_y - visited_dot_radius,
                        dot_x + visited_dot_radius,
                        dot_y + visited_dot_radius,
                    ),
                    fill=(203, 213, 225, 235),
                )

    for local_x, label in enumerate(x_labels):
        label_width, label_height = measure_text(measure_draw, label, font=font)
        x_center = int(left_margin + ((local_x + 0.5) * tile_width))
        draw.text(
            (x_center - (label_width // 2), header_height + 2),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    for local_y, label in enumerate(y_labels):
        label_width, label_height = measure_text(measure_draw, label, font=font)
        y_center = int(top_margin + ((local_y + 0.5) * tile_height))
        draw.text(
            (
                left_margin - padding - label_width,
                y_center - (label_height // 2),
            ),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    for sprite_x, sprite_y in snapshot.sprite_positions:
        local = snapshot.absolute_to_local(sprite_x, sprite_y)
        if local is None:
            continue
        left = int(left_margin + (local[0] * tile_width))
        top = int(top_margin + (local[1] * tile_height))
        right = int(left_margin + ((local[0] + 1) * tile_width))
        bottom = int(top_margin + ((local[1] + 1) * tile_height))
        inset = max(4, scale * 3)
        draw.rectangle(
            (left + inset, top + inset, right - inset, bottom - inset),
            fill=(255, 174, 66, 190),
        )

    for warp in snapshot.warps:
        wx = warp.get("x") if isinstance(warp, dict) else None
        wy = warp.get("y") if isinstance(warp, dict) else None
        if wx is None or wy is None:
            continue
        warp_local = snapshot.absolute_to_local(int(wx), int(wy))
        if warp_local is None:
            continue
        left = int(left_margin + (warp_local[0] * tile_width))
        top = int(top_margin + (warp_local[1] * tile_height))
        right = int(left_margin + ((warp_local[0] + 1) * tile_width))
        bottom = int(top_margin + ((warp_local[1] + 1) * tile_height))
        draw.rectangle(
            (left + 1, top + 1, right - 1, bottom - 1),
            outline=(213, 80, 255, 255),
            width=grid_line_width + 1,
        )
        warp_label_width, warp_label_height = measure_text(measure_draw, "W", font=font)
        draw.text(
            (
                left + int((tile_width - warp_label_width) / 2),
                top + int((tile_height - warp_label_height) / 2),
            ),
            "W",
            fill=(213, 80, 255, 255),
            font=font,
        )

    player_left = int(left_margin + (4 * tile_width))
    player_top = int(top_margin + (4 * tile_height))
    player_right = int(left_margin + (5 * tile_width))
    player_bottom = int(top_margin + (5 * tile_height))
    draw.rectangle(
        (player_left + 2, player_top + 2, player_right - 2, player_bottom - 2),
        outline=(55, 208, 255, 255),
        width=grid_line_width + 1,
    )
    player_label_width, player_label_height = measure_text(measure_draw, "P", font=font)
    draw.text(
        (
            player_left + int((tile_width - player_label_width) / 2),
            player_top + int((tile_height - player_label_height) / 2),
        ),
        "P",
        fill=(55, 208, 255, 255),
        font=font,
    )

    if goal is not None:
        goal_local = snapshot.absolute_to_local(goal[0], goal[1])
        if goal_local is not None:
            left = int(left_margin + (goal_local[0] * tile_width))
            top = int(top_margin + (goal_local[1] * tile_height))
            right = int(left_margin + ((goal_local[0] + 1) * tile_width))
            bottom = int(top_margin + ((goal_local[1] + 1) * tile_height))
            draw.rectangle(
                (left + 2, top + 2, right - 2, bottom - 2),
                outline=(255, 214, 10, 255),
                width=grid_line_width + 1,
            )
            goal_label_width, goal_label_height = measure_text(measure_draw, "G", font=font)
            draw.text(
                (
                    left + int((tile_width - goal_label_width) / 2),
                    top + int((tile_height - goal_label_height) / 2),
                ),
                "G",
                fill=(255, 214, 10, 255),
                font=font,
            )

    interaction = snapshot.interaction or {}
    target_coord = interaction.get("target_coord") or {}
    if target_coord.get("x") is not None and target_coord.get("y") is not None:
        local = snapshot.absolute_to_local(int(target_coord["x"]), int(target_coord["y"]))
        if local is not None:
            left = int(left_margin + (local[0] * tile_width))
            top = int(top_margin + (local[1] * tile_height))
            right = int(left_margin + ((local[0] + 1) * tile_width))
            bottom = int(top_margin + ((local[1] + 1) * tile_height))
            inset = max(4, scale * 3)
            draw.ellipse(
                (left + inset, top + inset, right - inset, bottom - inset),
                outline=(255, 125, 0, 255),
                width=grid_line_width + 1,
            )

    return canvas.convert("RGB")
