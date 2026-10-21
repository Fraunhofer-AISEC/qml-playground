# QML Playground – legacy Dash version

> **This is the legacy version.** The maintained QML Playground is the browser-only TypeScript app in
> [`../web`](../web). The Dash app is kept because it is the **numerical reference** for the web app's
> parity tests, and for existing server deployments.

The original QML Playground is a [Plotly Dash](https://dash.plotly.com/) app with a custom PyTorch
state-vector simulator. The server is stateless: every callback rebuilds the model from parameters
stored in the browser (`dcc.Store`), trains or evaluates, and writes the results back.

## Contents

```
app/                         Dash app (layout, callbacks, models, PyTorch backends, visualizations)
requirements.txt             Python dependencies
Dockerfile                   gunicorn image (base: pytorch/pytorch)
docker-compose.yml           published image + nginx on port 80
docker-compose.build-test.yml  build the image locally and test it behind nginx
docker_compose.dev.yml       development override (mounts app/, DEBUG=1)
nginx.conf                   reverse proxy for /qml-playground/
tools/export_fixtures.py     exports the reference fixtures used by the web app's tests
```

## Requirements

* Python 3.11+
* A virtual environment
* The packages in `requirements.txt` (dash, plotly, pandas, torch, gunicorn)
* Docker and Docker Compose (optional)

## Installation

```bash
git clone https://github.com/fraunhofer-aisec/qml-playground
cd qml-playground/legacy

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running locally

The modules use flat imports, so run the app from inside `app/`:

```bash
cd app
python app_dev.py
```

Then visit [http://127.0.0.1:8050/qml-playground/](http://127.0.0.1:8050/qml-playground/).

## Docker

All Docker files live in this folder; run the commands from `legacy/`.

Use the published image from the GitHub Container Registry, served behind nginx on port 80:

```bash
docker pull ghcr.io/fraunhofer-aisec/qml-playground:latest
docker compose up -d    # start
docker compose down     # stop
```

Build the image locally and test it behind nginx:

```bash
docker compose -f docker-compose.build-test.yml up --build
```

In production, gunicorn serves `app:server` with **one worker**. The base image already contains torch, so the
Dockerfile filters it out of `requirements.txt`. The URL prefix `/qml-playground/` is set in `app/logic.py`
(`url_base_pathname`) and must match `nginx.conf` and the health checks in the Dockerfile and compose files.

GitHub Actions (`.github/workflows/docker-publish.yml`) builds the image and pushes it to GHCR on every push to
`main` that touches `legacy/`.

## Regenerating the web app's reference fixtures

The web app's Vitest suite compares its simulator, losses, gradients and optimizer against this implementation.
After changing the Python model, regenerate the fixtures from the repository root:

```bash
pip install -r legacy/requirements.txt
python legacy/tools/export_fixtures.py   # writes web/tests/fixtures/reference.json
```

## Troubleshooting

* Make sure all required packages are installed in the active environment.
* Check that port 8050 (local) or 80 (Docker) is free.
* Make sure Docker Desktop is running (for Docker users).
