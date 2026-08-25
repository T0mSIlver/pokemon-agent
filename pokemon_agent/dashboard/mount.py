"""Mounting the operator dashboard onto a FastAPI app.

There is exactly one layout the page works under, and it is dictated by
``static/index.html``: the shell is served at ``/dashboard`` and every asset it
asks for is under ``/dashboard/assets/`` (``app.js``, ``style.css``). Mounting
``StaticFiles`` at ``/dashboard`` instead — the obvious-looking alternative —
serves the index fine and then 404s both assets, so this module deliberately
mounts the assets sub-path and serves the shell itself.

``pokemon_agent.server`` currently open-codes this same pair of routes at
module scope; see the note in :func:`mount_dashboard`.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

ASSETS_ROUTE = "/dashboard/assets"
INDEX_ROUTES = ("/dashboard", "/dashboard/")


def dashboard_static_dir():
    """The directory holding ``index.html``, or ``None`` if it is not installed."""

    if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").is_file():
        return STATIC_DIR
    return None


def mount_dashboard(app) -> bool:
    """Mount the dashboard on ``app``. Returns whether it is now reachable.

    Safe to call twice: an already-mounted ``/dashboard/assets`` is left alone,
    which is also what makes it safe to call from a server that mounts the same
    routes itself.

    ``pokemon_agent.server`` does not call this yet — it repeats the same two
    routes inline. Replacing that block with ``mount_dashboard(app)`` would
    leave one mounting path instead of two, but ``server.py`` belongs to another
    agent, so the duplication stands for now.
    """

    static_dir = dashboard_static_dir()
    if static_dir is None:
        logger.error("Dashboard static directory not found: %s", STATIC_DIR)
        return False

    try:
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        logger.warning("FastAPI not installed. Install with: pip install pokemon-agent[dashboard]")
        return False

    existing = {getattr(route, "path", None) for route in app.router.routes}

    if ASSETS_ROUTE not in existing:
        app.mount(
            ASSETS_ROUTE,
            StaticFiles(directory=str(static_dir), html=False),
            name="dashboard-assets",
        )

    index_file = static_dir / "index.html"

    async def dashboard_index():
        # no-store: the shell names its assets with a ?v= token, so a cached
        # shell is how an operator ends up staring at last week's dashboard.
        return FileResponse(
            index_file,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    for route in INDEX_ROUTES:
        if route not in existing:
            app.add_api_route(route, dashboard_index, methods=["GET"], include_in_schema=False)

    logger.info("Dashboard mounted at /dashboard (assets under %s)", ASSETS_ROUTE)
    return True
