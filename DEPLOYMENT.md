# Deploying TempoSat to Streamlit Community Cloud

## Before you start — what changes when you deploy

Streamlit Community Cloud runs on **CPU only, no GPU**. Your code already
falls back to CPU automatically (`DEVICE = torch.device("cuda" if
torch.cuda.is_available() else "cpu")`), so it won't crash — but training
will be noticeably slower than on your RTX 3050. For a live demo, consider
keeping epochs low (10-20) on the deployed version, or running the demo
from your own laptop and only using the public link as a backup/share link.

---

## Step 1 — Push to GitHub

1. Create a new repo on GitHub (public or private both work with Streamlit
   Cloud, but public is simpler for a hackathon submission).
2. In your project folder, with `app.py`, `requirements.txt`, and
   `.gitignore` all present:

```bash
git init
git add .
git commit -m "TempoSat — initial deploy"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/temposat.git
git push -u origin main
```

**Double check `.gitignore` is committed BEFORE you ever add a real
secrets file or service account JSON.** If you accidentally commit a
credential file, treat it as compromised — delete the GCP service account
key and make a new one, don't just remove the file from a later commit.

---

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, select your repo, branch `main`, and file path
   `app.py`.
3. Click **Deploy**. It will fail or show warnings until Step 3 is done
   (Earth Engine won't connect) — that's expected, upload mode will still
   work in the meantime.

---

## Step 3 — Earth Engine service account (for map mode to work)

Your personal Earth Engine login (`earthengine authenticate`) only works
on your own machine. The deployed app needs its own identity: a **service
account**.

### 3a. Create the service account

1. Go to https://console.cloud.google.com and select your project
   (`aerial-ether-500308-v5`, matching what's in `app.py`).
2. Navigate to **IAM & Admin → Service Accounts → Create Service Account**.
3. Name it something like `temposat-deploy`. No special roles are
   strictly required for Earth Engine access, but `Editor` is the safe
   default if you're unsure.
4. After creating it, open the service account, go to the **Keys** tab,
   click **Add Key → Create new key → JSON**. This downloads a `.json`
   file — guard this like a password.

### 3b. Register it with Earth Engine

1. Go to https://signup.earthengine.google.com and confirm the service
   account email (looks like
   `temposat-deploy@aerial-ether-500308-v5.iam.gserviceaccount.com`) has
   Earth Engine access — this may already be true if your GCP project
   itself is EE-enabled, but if you get a permissions error later, this
   is the step to revisit.

### 3c. Add it to Streamlit Cloud secrets

1. Open the downloaded JSON key file. It looks like:

```json
{
  "type": "service_account",
  "project_id": "aerial-ether-500308-v5",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "temposat-deploy@aerial-ether-500308-v5.iam.gserviceaccount.com",
  "client_id": "...",
  ...
}
```

2. In your Streamlit Cloud app dashboard, go to **Settings → Secrets**.
3. Copy the contents of `secrets.toml.example` (included in this repo)
   and paste it into the Secrets box, replacing every placeholder value
   with the matching field from your JSON key file.
   **Important:** the `private_key` field contains literal `\n` newline
   characters — keep them exactly as they appear in the JSON file, inside
   the quoted TOML string.
4. Save. Streamlit Cloud will restart your app automatically.

### 3d. Verify

Open your deployed app. If the sidebar warning "Earth Engine not
connected" is gone, it worked. Try map mode with a known-good location
(Delhi, wide date range) to confirm downloads actually succeed.

---

## Troubleshooting

**"Earth Engine not connected" warning persists after adding secrets**
Double-check the TOML formatting — a broken `\n` in the private key is
the most common cause. Re-copy directly from the JSON file rather than
retyping.

**App crashes on startup with a package error**
Check the deployed app's logs (Manage app → logs, in the Streamlit Cloud
dashboard). Usually a version pin in `requirements.txt` conflicting with
what Streamlit Cloud's base image already provides — try removing the
specific version pin and letting pip resolve it.

**Training is too slow for a live demo**
Expected — no GPU on the free tier. For judging, either: (a) demo from
your own laptop and just show the public link as proof it deploys, or
(b) lower default `epochs` in the sidebar slider for the deployed version
specifically.

**"OMP: Error #15" on the deployed server**
Shouldn't happen — the `KMP_DUPLICATE_LIB_OK` env var is already set at
the top of `app.py`, and Linux containers (which Streamlit Cloud uses)
are far less prone to this conflict than Windows anyway. If it does show
up, it's already handled by existing code.
