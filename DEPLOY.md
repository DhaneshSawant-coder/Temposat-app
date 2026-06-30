# Deploying TempoSat

## 1. One-time: create a Google Earth Engine service account
Mode 2 (map picker) needs GEE access without an interactive login prompt.

1. In Google Cloud Console, create (or reuse) a project and enable the
   **Earth Engine API**.
2. Create a **Service Account** → grant it the `Earth Engine Resource Viewer`
   role (or register it at https://signup.earthengine.google.com/ as a
   service account user).
3. Create a JSON key for that service account and download it.
4. Open `secrets.toml.example` in this folder — copy the field names
   (`project_id`, `private_key`, `client_email`, etc.) from your downloaded
   JSON key into that format.

**Never commit the real JSON key or a filled-in secrets.toml to GitHub.**

## 2. Deploy on Streamlit Community Cloud (recommended, free)
1. Push this folder to a GitHub repo (must include `app.py` and
   `requirements.txt`).
2. Go to https://share.streamlit.io → "New app" → pick the repo/branch →
   set main file to `app.py`.
3. In the app's **Settings → Secrets**, paste the filled-in contents of
   `secrets.toml.example` (with your real key values).
4. Deploy. First boot installs torch + opencv, which can take a few minutes.

## 3. Deploy on Hugging Face Spaces (alternative, free)
1. Create a new Space → SDK: **Streamlit**.
2. Upload `app.py` and `requirements.txt` (HF auto-detects `requirements.txt`).
3. Go to Space **Settings → Repository secrets**, add a secret named
   `gee_service_account` containing the same TOML/JSON content as above
   (or adapt `init_earth_engine()` to read from `os.environ` if you prefer
   one JSON blob instead of Streamlit's TOML secrets).

## 4. Notes
- `opencv-python-headless` is used instead of `opencv-python` — the GUI
  build fails to install on headless cloud servers.
- Mode 1 (upload) works immediately with zero configuration — only Mode 2
  needs the Earth Engine secret.
- If `gee_service_account` isn't found in secrets, the app falls back to
  `ee.Initialize()` (interactive auth) — fine for local testing, not for
  a deployed server.
