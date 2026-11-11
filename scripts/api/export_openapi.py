"""Export the FastAPI OpenAPI spec and a standalone Swagger UI page.

The running API already serves Swagger at ``/docs`` (ReDoc at ``/redoc``, spec at ``/openapi.json``).
This writes the same spec to files, for the submission, reviewers and API clients that have no server:

    docs/api/openapi.json   OpenAPI 3.1 spec
    docs/api/openapi.yaml   the same, as YAML
    docs/api/swagger.html   Swagger UI with the spec embedded: open it from disk. "Try it out" calls
                            the server picked in the Servers list (CORS_ALLOW_ORIGINS must allow it)

    python -m scripts.api.export_openapi
    python -m scripts.api.export_openapi --server http://127.0.0.1:8000 --server https://copilot.example.org
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from scripts.common.paths import ROOT

DEFAULT_SERVERS = ("http://127.0.0.1:8000",)
SWAGGER_UI_VERSION = "5.17.14"     # cdnjs swagger-ui
CDN = f"https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/{SWAGGER_UI_VERSION}"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{cdn}/swagger-ui.min.css">
<style>
  body {{ margin: 0; background: #fafafa; }}
  .note {{ font: 14px/1.5 system-ui, sans-serif; padding: 12px 16px; background: #eef4fb; border-bottom: 1px solid #c9d9ec; }}
  .note code {{ background: #fff; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<div class="note">Static copy of the API contract (version {version}). To call the API, start it
(<code>uvicorn scripts.api.api_app:app --port 8000</code>), pick the server below, click <b>Authorize</b> and
enter the <code>API_AUTH_TOKEN</code> value if the server sets one. The live page is <code>/docs</code> on the server.</div>
<div id="swagger-ui"></div>
<script src="{cdn}/swagger-ui-bundle.min.js"></script>
<script>
  const spec = {spec};
  window.ui = SwaggerUIBundle({{
    spec, dom_id: "#swagger-ui", deepLinking: true, persistAuthorization: true,
    displayRequestDuration: true, tryItOutEnabled: false, docExpansion: "list", defaultModelsExpandDepth: 0
  }});
</script>
</body>
</html>
"""


def build_spec(servers=DEFAULT_SERVERS) -> dict:
    from scripts.api.api_app import app
    spec = json.loads(json.dumps(app.openapi()))          # a copy: the app's cached schema stays without servers
    if servers:
        spec["servers"] = [{"url": url.rstrip("/")} for url in servers]
    return spec


def swagger_page(spec: dict) -> str:
    embedded = json.dumps(spec, ensure_ascii=False).replace("</", "<\\/")   # safe inside <script>
    return PAGE.format(title=html.escape(spec["info"]["title"]), version=html.escape(spec["info"]["version"]),
                       cdn=CDN, spec=embedded)


def export(out_dir: Path, servers=DEFAULT_SERVERS) -> dict[str, Path]:
    import yaml
    spec = build_spec(servers)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"json": out_dir / "openapi.json", "yaml": out_dir / "openapi.yaml", "html": out_dir / "swagger.html"}
    files["json"].write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files["yaml"].write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    files["html"].write_text(swagger_page(spec), encoding="utf-8")
    return files


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "api")
    parser.add_argument("--server", action="append", help="Server URL for 'Try it out' (repeatable).")
    args = parser.parse_args(argv)
    files = export(args.out_dir, args.server or DEFAULT_SERVERS)
    spec = json.loads(files["json"].read_text(encoding="utf-8"))
    operations = sum(len(ops) for ops in spec["paths"].values())
    print(f"OpenAPI {spec['openapi']}: {len(spec['paths'])} paths, {operations} operations, "
          f"{len(spec['components']['schemas'])} schemas")
    for path in files.values():
        print(" ", path)


if __name__ == "__main__":
    main()
