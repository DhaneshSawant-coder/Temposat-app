import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import cv2
from PIL import Image
import io
import ee
import geemap
import rasterio
import folium
from streamlit_folium import st_folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import datetime
import time

st.set_page_config(
    page_title="TempoSat — Satellite Time Predictor",
    page_icon="🛰️",
    layout="wide"
)

# ── Device detection ───────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("🛰️ TempoSat")
st.subheader("AI-Powered Satellite Image Temporal Prediction")
st.markdown("""
**BAH 2026 — Problem Statement 12**
> Upload your own images **or** click any location on the map to get real satellite data.
""")

if DEVICE.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    st.success(f"⚡ GPU Detected: **{gpu_name}** ({vram:.1f} GB VRAM) — Training will be fast!")
else:
    st.warning("⚠️ No GPU detected — running on CPU (slower). Check your PyTorch CUDA install.")

st.divider()

# ── Initialize Earth Engine ────────────────────────────────────────────────
@st.cache_resource
def init_ee():
    try:
        ee.Initialize(project='aerial-ether-500308-v5')
        return True
    except Exception:
        return False

ee_ready = init_ee()

# ══════════════════════════════════════════════════════════════════════════
# DEEP U-NET (temporal prediction model)
# ══════════════════════════════════════════════════════════════════════════
class DoubleConv(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class TempoSatDeepUNet(nn.Module):
    """
    4-level U-Net: 6-channel input (T1+T2 stacked) → 3-channel RGB output.
    Channels: 64 → 128 → 256 → 512 (bottleneck) → 256 → 128 → 64 → out

    RESIDUAL DESIGN: instead of predicting the output image from scratch,
    the network predicts a small correction on top of the simple T1/T2
    baseline blend. This makes training much easier — the model starts
    from a sensible answer (the baseline) and only has to learn the
    adjustments needed to beat it, rather than reconstruct everything.
    """
    def __init__(self, base=64):
        super().__init__()
        self.enc1 = DoubleConv(6, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)

        self.pool = nn.MaxPool2d(2)

        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)

        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)

        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        # No Sigmoid here — this predicts an unbounded correction (residual),
        # not a final pixel value. Tanh keeps the correction in a sane
        # range (-1, 1) which is then scaled down and added to the baseline.
        self.out  = nn.Sequential(
            nn.Conv2d(base, 3, 1),
            nn.Tanh()
        )
        # How much the correction is allowed to shift pixels, at most.
        # Small on purpose — the baseline should dominate, AI only refines it.
        self.residual_scale = 0.25

    def forward(self, t1, t2):
        x  = torch.cat([t1, t2], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        correction = self.out(d1) * self.residual_scale
        baseline   = (t1 + t2) / 2.0
        return torch.clamp(baseline + correction, 0.0, 1.0)


# ── Loss functions ─────────────────────────────────────────────────────────
def ssim_loss(pred, target):
    """
    Per-sample, per-channel SSIM approximation.
    Computing mean/var globally over the whole batch (the old behaviour)
    rewards matching batch-wide brightness statistics instead of actual
    per-image structural similarity, which fights against fine-grained
    corrections like the residual U-Net's local adjustments.
    """
    dims   = (2, 3)  # height, width — keep batch and channel separate
    mu_p   = pred.mean(dim=dims, keepdim=True)
    mu_t   = target.mean(dim=dims, keepdim=True)
    sig_p  = pred.var(dim=dims, keepdim=True, unbiased=False)
    sig_t  = target.var(dim=dims, keepdim=True, unbiased=False)
    sig_pt = ((pred - mu_p) * (target - mu_t)).mean(dim=dims, keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    ssim   = ((2*mu_p*mu_t + c1) * (2*sig_pt + c2)) / \
             ((mu_p**2 + mu_t**2 + c1) * (sig_p + sig_t + c2))
    return (1 - ssim).mean()

def gradient_loss(pred, target):
    """Penalise blurry edges — encourages sharper texture."""
    def grad(t):
        dy = t[:, :, 1:, :] - t[:, :, :-1, :]
        dx = t[:, :, :, 1:] - t[:, :, :, :-1]
        return dy, dx
    p_dy, p_dx = grad(pred)
    t_dy, t_dx = grad(target)
    return (p_dy - t_dy).abs().mean() + (p_dx - t_dx).abs().mean()

def combined_loss(pred, target):
    mse  = nn.functional.mse_loss(pred, target)
    ssim = ssim_loss(pred, target)
    gl   = gradient_loss(pred, target)
    return 0.4 * mse + 0.4 * ssim + 0.2 * gl


# ══════════════════════════════════════════════════════════════════════════
# SUPER-RESOLUTION MODELS
# ══════════════════════════════════════════════════════════════════════════
class FastSR(nn.Module):
    """Lightweight 4× SR: conv layers + PixelShuffle. ~60K params."""
    def __init__(self, scale=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


class DenseBlock(nn.Module):
    """Dense residual block used inside RRDB."""
    def __init__(self, ch=64, gc=32):
        super().__init__()
        self.c1 = nn.Conv2d(ch,      gc, 3, padding=1)
        self.c2 = nn.Conv2d(ch+gc,   gc, 3, padding=1)
        self.c3 = nn.Conv2d(ch+gc*2, gc, 3, padding=1)
        self.c4 = nn.Conv2d(ch+gc*3, gc, 3, padding=1)
        self.c5 = nn.Conv2d(ch+gc*4, ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""
    def __init__(self, ch=64, gc=32):
        super().__init__()
        self.db1 = DenseBlock(ch, gc)
        self.db2 = DenseBlock(ch, gc)
        self.db3 = DenseBlock(ch, gc)
    def forward(self, x):
        out = self.db3(self.db2(self.db1(x)))
        return out * 0.2 + x


class QualitySR(nn.Module):
    """ESRGAN-style RRDB network for 4× SR. ~2.7M params."""
    def __init__(self, scale=4, n_rrdb=6):
        super().__init__()
        self.head = nn.Conv2d(3, 64, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(64, 32) for _ in range(n_rrdb)])
        self.tail = nn.Conv2d(64, 64, 3, padding=1)
        self.up1  = nn.Sequential(
            nn.Conv2d(64, 64*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up2  = nn.Sequential(
            nn.Conv2d(64, 64*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out  = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.head(x)
        feat = feat + self.tail(self.body(feat))
        feat = self.up1(feat)
        feat = self.up2(feat)
        return self.out(feat)


def train_sr_model(t1: np.ndarray, t2: np.ndarray, mode: str) -> nn.Module:
    """
    Self-supervised SR training on the uploaded/downloaded satellite images.
    Downsample 64px HR patches 4x -> 16px LR input, learn to reconstruct HR.
    This makes the SR model learn THIS image's own colours/texture instead
    of producing flat grey output from random init.
    """
    if mode == "Quality (RRDB — sharpest)":
        sr = QualitySR(scale=4, n_rrdb=6).to(DEVICE)
        sr_epochs, lr = 40, 2e-4
    else:
        sr = FastSR(scale=4).to(DEVICE)
        sr_epochs, lr = 25, 3e-4

    opt    = optim.Adam(sr.parameters(), lr=lr)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    imgs_gpu = [
        torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE)
        for img in (t1, t2)
    ]

    PATCH = 64
    BATCH = 16
    STEPS = 8

    bar = st.progress(0, text="🔬 Training SR on your satellite images...")
    sr.train()
    loss = torch.tensor(0.0)

    for epoch in range(sr_epochs):
        for _ in range(STEPS):
            src = imgs_gpu[epoch % 2]
            _, h, w = src.shape
            ps = min(PATCH, h, w)

            xs = torch.randint(0, max(1, w - ps), (BATCH,))
            ys = torch.randint(0, max(1, h - ps), (BATCH,))

            hr_patches = torch.stack([
                src[:, ys[i]:ys[i]+ps, xs[i]:xs[i]+ps]
                for i in range(BATCH)
            ])

            lr_patches = nn.functional.interpolate(
                hr_patches, scale_factor=0.25,
                mode='bilinear', align_corners=False
            )

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(DEVICE.type == "cuda")):
                pred_hr = sr(lr_patches)
                loss    = (nn.functional.mse_loss(pred_hr, hr_patches) * 0.6 +
                           gradient_loss(pred_hr, hr_patches) * 0.4)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if (epoch + 1) % 5 == 0 or epoch == sr_epochs - 1:
            bar.progress(
                int((epoch + 1) / sr_epochs * 100),
                text=f"🔬 SR epoch {epoch+1}/{sr_epochs} — loss: {loss.item():.5f}"
            )

    bar.progress(100, text="✅ SR training complete!")
    sr.eval()
    return sr


def super_resolve(pred_np: np.ndarray, sr_model: nn.Module) -> np.ndarray:
    t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        with autocast(enabled=(DEVICE.type == "cuda")):
            out = sr_model(t)
    return out.squeeze(0).permute(1, 2, 0).cpu().float().numpy().clip(0, 1)


# ══════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════
TRAIN_SIZE = 256  # upgraded from 64

def prepare_arr(arr):
    arr = cv2.resize(arr, (TRAIN_SIZE, TRAIN_SIZE))
    return arr.astype(np.float32)

def prepare_pil(pil_img):
    img = pil_img.convert('RGB').resize((TRAIN_SIZE, TRAIN_SIZE), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0

def to_pil(arr, size=400):
    arr   = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr   = np.clip(arr, 0, 1)
    uint8 = (arr * 255).astype(np.uint8)
    h, w  = uint8.shape[:2]
    if h >= size and w >= size:
        s       = min(h, w)
        y_start = (h - s) // 2
        x_start = (w - s) // 2
        uint8   = uint8[y_start:y_start+s, x_start:x_start+s]
    big = cv2.resize(uint8, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(big)

def smooth(pred):
    uint8    = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    smoothed = cv2.bilateralFilter(uint8, 9, 75, 75)
    return smoothed.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════
# FAST GPU TRAINING (temporal model)
# ══════════════════════════════════════════════════════════════════════════
def random_crops_gpu(t1_gpu, t2_gpu, batch_size=16, patch_size=96):
    """Sample augmented random crops entirely on GPU — no per-batch CPU transfer."""
    _, h, w = t1_gpu.shape
    ps = min(patch_size, h, w)

    xs = torch.randint(0, max(1, w - ps), (batch_size,), device=DEVICE)
    ys = torch.randint(0, max(1, h - ps), (batch_size,), device=DEVICE)

    p1_list, p2_list = [], []
    for i in range(batch_size):
        p1_list.append(t1_gpu[:, ys[i]:ys[i]+ps, xs[i]:xs[i]+ps])
        p2_list.append(t2_gpu[:, ys[i]:ys[i]+ps, xs[i]:xs[i]+ps])

    p1 = torch.stack(p1_list)
    p2 = torch.stack(p2_list)

    alpha = torch.rand(batch_size, 1, 1, 1, device=DEVICE) * 0.4 + 0.3
    mid   = alpha * p1 + (1 - alpha) * p2

    flip_h = torch.rand(batch_size, device=DEVICE) > 0.5
    p1[flip_h]  = p1[flip_h].flip(-1)
    p2[flip_h]  = p2[flip_h].flip(-1)
    mid[flip_h] = mid[flip_h].flip(-1)

    flip_v = torch.rand(batch_size, device=DEVICE) > 0.5
    p1[flip_v]  = p1[flip_v].flip(-2)
    p2[flip_v]  = p2[flip_v].flip(-2)
    mid[flip_v] = mid[flip_v].flip(-2)

    jitter = torch.rand(batch_size, 1, 1, 1, device=DEVICE) * 0.3 + 0.85
    p1 = (p1 * jitter).clamp(0, 1)
    p2 = (p2 * jitter).clamp(0, 1)

    return p1, p2, mid


def train_model(t1, t2, epochs=30):
    t1_gpu = torch.from_numpy(t1).permute(2, 0, 1).float().to(DEVICE)
    t2_gpu = torch.from_numpy(t2).permute(2, 0, 1).float().to(DEVICE)

    model     = TempoSatDeepUNet(base=64).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler(enabled=(DEVICE.type == "cuda"))

    n_params = sum(p.numel() for p in model.parameters())
    st.caption(f"🧠 Model: {n_params:,} params | Device: {DEVICE} | "
               f"Batch: 16 crops/step × 10 steps/epoch | Patch: 96×96")

    STEPS_PER_EPOCH = 10
    BATCH           = 16

    bar     = st.progress(0, text="🧠 Training AI on your images...")
    t_start = time.time()

    model.train()
    ep_loss = 0.0
    for epoch in range(epochs):
        ep_loss = 0.0
        for _ in range(STEPS_PER_EPOCH):
            p1, p2, mid = random_crops_gpu(t1_gpu, t2_gpu, batch_size=BATCH, patch_size=96)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(DEVICE.type == "cuda")):
                pred = model(p1, p2)
                loss = combined_loss(pred, mid)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()

        scheduler.step()
        ep_loss /= STEPS_PER_EPOCH

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - t_start
            eta     = elapsed / (epoch + 1) * (epochs - epoch - 1)
            bar.progress(
                int((epoch + 1) / epochs * 100),
                text=f"🧠 Epoch {epoch+1}/{epochs} — Loss: {ep_loss:.5f} | ETA: {eta:.0f}s"
            )

    bar.progress(100, text=f"✅ Training complete in {time.time()-t_start:.1f}s!")
    model.eval()
    return model


def predict(model, t1, t2):
    with torch.no_grad():
        t1_t = torch.from_numpy(t1).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        t2_t = torch.from_numpy(t2).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        with autocast(enabled=(DEVICE.type == "cuda")):
            pred = model(t1_t, t2_t)
        pred = pred.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return np.clip(pred, 0, 1)


def psnr(p, r):
    mse = np.mean((p - r) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else 999.0


# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS FEATURES (kept from your original — unchanged logic)
# ══════════════════════════════════════════════════════════════════════════
def compute_ndvi(img):
    nir  = img[:,:,1].astype(np.float32)
    red  = img[:,:,0].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.clip(ndvi, -1, 1)

def ndvi_to_color(ndvi):
    norm      = (ndvi + 1) / 2.0
    uint8     = (norm * 255).astype(np.uint8)
    color_img = np.zeros((*uint8.shape, 3), dtype=np.uint8)
    color_img[:,:,0] = np.clip(255 - uint8 * 2, 0, 255)
    color_img[:,:,1] = np.clip(uint8,            0, 255)
    color_img[:,:,2] = np.clip(50,               0, 255)
    return color_img.astype(np.float32) / 255.0

def detect_clouds(img):
    brightness = img.mean(axis=2)
    r, g, b    = img[:,:,0], img[:,:,1], img[:,:,2]
    cloud_mask = (brightness > 0.75) & \
                 (np.abs(r-g) < 0.15) & \
                 (np.abs(g-b) < 0.15)
    coverage   = cloud_mask.mean() * 100
    vis        = img.copy()
    vis[cloud_mask] = [1.0, 0.2, 0.2]
    return vis, coverage

def make_error_map(pred, baseline):
    pred_err     = np.abs(pred - baseline).mean(axis=2)
    baseline_err = np.zeros_like(pred_err)
    diff         = pred_err - baseline_err
    error_map    = np.zeros((*diff.shape, 3), dtype=np.float32)
    error_map[:,:,1] = np.clip(-diff * 10, 0, 1)
    error_map[:,:,0] = np.clip( diff * 10, 0, 1)
    same_mask = np.abs(diff) < 0.02
    error_map[same_mask] = [1.0, 1.0, 0.0]
    return error_map

def make_confidence_map(pred, t1, t2):
    average    = (t1 + t2) / 2.0
    diff       = np.abs(pred - average).mean(axis=2)
    confidence = np.clip(diff * 5, 0, 1)
    conf_uint8 = (confidence * 255).astype(np.uint8)
    conf_color = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(conf_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def make_comparison_chart(b_score, ai_score):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    fig.patch.set_facecolor('#0E1117')
    bars = axes[0].bar(['Baseline', 'AI Model'],
                       [b_score, ai_score],
                       color=['#FF6B6B', '#4ECDC4'], width=0.5)
    axes[0].set_title('PSNR Comparison (dB)', color='white', fontsize=12)
    axes[0].set_facecolor('#1E2329')
    axes[0].tick_params(colors='white')
    axes[0].set_ylim([min(b_score, ai_score)-2, max(b_score, ai_score)+2])
    for bar, val in zip(bars, [b_score, ai_score]):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.05,
                     f'{val:.2f} dB', ha='center', color='white', fontsize=11)
    improvement = ai_score - b_score
    color       = '#4ECDC4' if improvement > 0 else '#FF6B6B'
    axes[1].bar(['Improvement'], [improvement], color=color, width=0.4)
    axes[1].set_title('AI vs Baseline', color='white', fontsize=12)
    axes[1].set_facecolor('#1E2329')
    axes[1].tick_params(colors='white')
    axes[1].axhline(y=0, color='white', linestyle='--', alpha=0.5)
    sign = '+' if improvement > 0 else ''
    axes[1].text(0, improvement+0.02,
                 f'{sign}{improvement:.2f} dB',
                 ha='center', color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0E1117',
                bbox_inches='tight', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

def load_tif(path):
    with rasterio.open(path) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
    img = np.stack([r, g, b], axis=-1)

    # Sanitize raw input — Sentinel-2 downloads can contain NaN/Inf
    # from nodata pixels, which silently break PIL image rendering.
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.zeros_like(img)
    for i in range(3):
        ch       = img[:,:,i]
        non_zero = ch[ch > 0]
        if len(non_zero) < 10:
            # Not enough real data in this band — fall back to a
            # simple normalize so the channel isn't left all-black/NaN.
            ch_max = ch.max()
            out[:,:,i] = (ch / ch_max) if ch_max > 0 else 0.0
            continue
        lo = np.percentile(non_zero, 1)
        hi = np.percentile(non_zero, 95)
        if hi <= lo:
            hi = lo + 1.0
        out[:,:,i] = np.clip((ch - lo) / (hi - lo), 0, 1)

    out = np.power(np.clip(out, 0, 1), 0.6)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0, 1).astype(np.float32)

def download_sentinel(lat, lon, date_start, date_end, out_path):
    delta  = 0.15
    region = ee.Geometry.Rectangle([
        lon-delta, lat-delta, lon+delta, lat+delta
    ])
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(region)
             .filterDate(date_start, date_end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))

    # Fail loudly instead of silently downloading an empty/null image.
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(
            f"No cloud-free Sentinel-2 scenes found for this location "
            f"between {date_start} and {date_end}. Try a wider date range "
            f"(e.g. 20-30 days) or a different location."
        )

    img = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()

    geemap.download_ee_image(
        image    = img.select(['B4','B3','B2']),
        filename = out_path,
        region   = region,
        scale    = 60,
        crs      = 'EPSG:4326',
        overwrite= True
    )

    # Validate the downloaded file is actually usable before returning.
    if not os.path.exists(out_path):
        raise ValueError(f"Download did not produce a file at {out_path}.")

    with rasterio.open(out_path) as src:
        if src.count < 3:
            raise ValueError(
                f"Downloaded GeoTIFF has only {src.count} band(s), expected 3 "
                f"(B4/B3/B2). The Earth Engine download may have failed partially."
            )
        if src.width < 10 or src.height < 10:
            raise ValueError(
                f"Downloaded GeoTIFF is too small ({src.width}x{src.height}px). "
                f"Try a larger area or different coordinates."
            )


# ══════════════════════════════════════════════════════════════════════════
# SHARED RESULTS RENDERER
# ══════════════════════════════════════════════════════════════════════════
def show_results(t1, t2, t1_display, t2_display,
                 label1, label2, epochs, sr_mode, mode="upload"):
    """Shared results section used by both upload and map modes.
    t1/t2 are float32 (TRAIN_SIZE, TRAIN_SIZE, 3) arrays in [0,1]."""

    # ── Train temporal model + predict (GPU, fast) ─────────────────────────
    model = train_model(t1, t2, epochs=epochs)
    pred  = predict(model, t1, t2)
    pred  = smooth(pred)
    baseline = np.clip((t1 + t2) / 2.0, 0, 1)

    n_unet_params = sum(p.numel() for p in model.parameters())
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── Super-resolution (self-supervised on this image) ───────────────────
    sr_model = train_sr_model(t1, t2, sr_mode)
    with st.spinner("🔬 Upscaling prediction to 1024px..."):
        pred_sr = super_resolve(pred, sr_model)
        base_sr = super_resolve(baseline, sr_model)

    n_sr_params = sum(p.numel() for p in sr_model.parameters())
    del sr_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    st.success("✅ Prediction + super-resolution complete!")
    st.divider()

    # ── 1. Main results ────────────────────────────────────────────────────
    st.markdown("### 🖼️ 1. Before → Predicted → After")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.image(t1_display, use_column_width=True)
        st.caption(f"📅 Real — {label1}")
    with r2:
        st.image(to_pil(pred, size=400), use_column_width=True)
        st.caption("🤖 AI Predicted Middle Day (256px)")
    with r3:
        st.image(t2_display, use_column_width=True)
        st.caption(f"📅 Real — {label2}")

    st.divider()

    # ── 2. Super-resolution before/after ───────────────────────────────────
    st.markdown("### 🔬 2. Super-Resolution — 4× Upscale to 1024px")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Before SR — 256px**")
        st.image(to_pil(pred, 512), use_column_width=True)
    with s2:
        st.markdown(f"**After SR — 1024px ({sr_mode.split('(')[0].strip()})**")
        st.image(to_pil(pred_sr, 1024), use_column_width=True)

    st.divider()

    # ── 3. AI vs Baseline ──────────────────────────────────────────────────
    st.markdown("### 🆚 3. AI vs Simple Blend Comparison (SR applied)")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**📊 Simple Average (Baseline) + SR**")
        st.image(to_pil(base_sr, 512), use_column_width=True)
        st.caption("Just averaging T1 + T2, then super-resolved")
    with b2:
        st.markdown("**🤖 AI Prediction + SR**")
        st.image(to_pil(pred_sr, 512), use_column_width=True)
        st.caption("Deep U-Net (MSE+SSIM+Gradient) + RRDB SR")

    st.divider()

    # ── 4. Confidence map ──────────────────────────────────────────────────
    st.markdown("### 🔵 4. Confidence Map")
    st.markdown("🔴 Red = AI is confident it learned something | 🔵 Blue = uncertain area")
    conf = make_confidence_map(pred, t1, t2)
    cf1, cf2 = st.columns(2)
    with cf1:
        st.image(to_pil(pred), use_column_width=True)
        st.caption("AI Prediction")
    with cf2:
        st.image(to_pil(conf), use_column_width=True)
        st.caption("Confidence Map")

    st.divider()

    # ── 5. Error map ───────────────────────────────────────────────────────
    st.markdown("### 🔴 5. Error Map — Where AI Beat Baseline")
    st.markdown("🟢 Green = AI better | 🔴 Red = Baseline better | 🟡 Yellow = same")
    err = make_error_map(pred, baseline)
    e1, e2, e3 = st.columns(3)
    with e1:
        st.image(to_pil(baseline), use_column_width=True)
        st.caption("Baseline")
    with e2:
        st.image(to_pil(pred), use_column_width=True)
        st.caption("AI Prediction")
    with e3:
        st.image(to_pil(err), use_column_width=True)
        st.caption("Error Map")

    st.divider()

    # ── 6. NDVI ────────────────────────────────────────────────────────────
    st.markdown("### 🌿 6. Vegetation Health (NDVI)")
    st.markdown("🟢 Dark green = healthy crops | 🟡 Yellow = sparse | 🔴 Red = urban/soil")
    ndvi1  = compute_ndvi(t1)
    ndvi_p = compute_ndvi(pred)
    ndvi2  = compute_ndvi(t2)
    n1, n2, n3 = st.columns(3)
    with n1:
        st.image(to_pil(ndvi_to_color(ndvi1)), use_column_width=True)
        st.caption(f"T1 NDVI: {ndvi1.mean():.3f}")
    with n2:
        st.image(to_pil(ndvi_to_color(ndvi_p)), use_column_width=True)
        st.caption(f"Predicted NDVI: {ndvi_p.mean():.3f}")
    with n3:
        st.image(to_pil(ndvi_to_color(ndvi2)), use_column_width=True)
        st.caption(f"T2 NDVI: {ndvi2.mean():.3f}")

    change = ndvi2.mean() - ndvi1.mean()
    if change > 0.01:
        st.success(f"🌱 Vegetation INCREASED by {change:.3f} — crops growing!")
    elif change < -0.01:
        st.warning(f"🍂 Vegetation DECREASED by {abs(change):.3f} — harvest or drought!")
    else:
        st.info("🌿 Vegetation stayed roughly the same.")

    st.divider()

    # ── 7. Cloud detection ─────────────────────────────────────────────────
    st.markdown("### ☁️ 7. Cloud Detection")
    st.markdown("🔴 Red overlay = detected cloud areas")
    c_vis1, cov1 = detect_clouds(t1)
    c_vis2, cov2 = detect_clouds(t2)
    cl1, cl2 = st.columns(2)
    with cl1:
        st.image(to_pil(c_vis1), use_column_width=True)
        label = "HIGH ⚠️" if cov1>20 else "MODERATE ⛅" if cov1>5 else "CLEAR ☀️"
        st.caption(f"T1 Cloud cover: {cov1:.1f}% — {label}")
    with cl2:
        st.image(to_pil(c_vis2), use_column_width=True)
        label = "HIGH ⚠️" if cov2>20 else "MODERATE ⛅" if cov2>5 else "CLEAR ☀️"
        st.caption(f"T2 Cloud cover: {cov2:.1f}% — {label}")

    st.divider()

    # ── 8. Performance chart + real PSNR ───────────────────────────────────
    st.markdown("### 📊 8. Performance Comparison")
    st.caption(
        "⚠️ Note: this metric measures similarity to T1/T2. A simple pixel "
        "average is mathematically optimal by this measure — it minimizes "
        "distance to both frames by definition. A model that correctly "
        "predicts real change (e.g. new vegetation growth) will score lower "
        "here even when it's more accurate, since it deviates more from "
        "both endpoints. Use this as a sanity check, not a leaderboard."
    )

    b_score_t1 = psnr(baseline, t1)
    b_score_t2 = psnr(baseline, t2)
    ai_score_t1 = psnr(pred, t1)
    ai_score_t2 = psnr(pred, t2)
    b_score  = (b_score_t1 + b_score_t2) / 2
    ai_score = (ai_score_t1 + ai_score_t2) / 2

    st.image(make_comparison_chart(b_score, ai_score), use_column_width=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Baseline PSNR",  f"{b_score:.2f} dB")
    with m2: st.metric("AI Model PSNR",  f"{ai_score:.2f} dB",
                       delta=f"{ai_score-b_score:+.2f} dB")
    with m3: st.metric("U-Net params", f"{n_unet_params/1e6:.2f}M")
    with m4: st.metric("SR params", f"{n_sr_params/1e6:.2f}M")

    m5, m6 = st.columns(2)
    with m5: st.metric("T1 Clouds", f"{cov1:.1f}%")
    with m6: st.metric("T2 Clouds", f"{cov2:.1f}%")

    st.divider()

    # ── 9. Downloads ───────────────────────────────────────────────────────
    st.markdown("### ⬇️ Download Results")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        buf = io.BytesIO()
        to_pil(pred_sr, 1024).save(buf, 'PNG')
        buf.seek(0)
        st.download_button("⬇️ AI Prediction (1024px SR)",
                           buf, "TempoSat_prediction_SR_1024.png",
                           "image/png", use_container_width=True, type="primary")
    with d2:
        buf = io.BytesIO()
        to_pil(pred, 512).save(buf, 'PNG')
        buf.seek(0)
        st.download_button("⬇️ Raw Prediction (512px)",
                           buf, "TempoSat_prediction_512.png",
                           "image/png", use_container_width=True)
    with d3:
        buf = io.BytesIO()
        to_pil(conf, 512).save(buf, 'PNG')
        buf.seek(0)
        st.download_button("⬇️ Confidence Map",
                           buf, "TempoSat_confidence.png",
                           "image/png", use_container_width=True)
    with d4:
        buf = io.BytesIO()
        to_pil(ndvi_to_color(ndvi_p), 512).save(buf, 'PNG')
        buf.seek(0)
        st.download_button("⬇️ NDVI Map",
                           buf, "TempoSat_ndvi.png",
                           "image/png", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("### 🗂️ Choose Your Input Method")
    input_mode = st.radio(
        "How do you want to provide images?",
        ["📁 Upload My Own Images", "🗺️ Pick Location on Map"],
        index=0
    )
    st.divider()

    st.markdown("### 🧠 Training Config")
    epochs = st.slider("Training epochs", 10, 80, 30, 5)
    st.caption("30 epochs ≈ 30–45s on a GPU")
    st.divider()

    st.markdown("### 🔬 Super-Resolution")
    sr_mode = st.radio(
        "SR mode",
        ["Fast (lightweight — <1s/epoch)", "Quality (RRDB — sharpest)"],
        index=1,
        help="Fast: simple conv upscaler. Quality: ESRGAN-style RRDB network."
    )
    st.caption({
        "Fast (lightweight — <1s/epoch)": "~60K params · ~10-15s total",
        "Quality (RRDB — sharpest)":      "~2.7M params · ~20-30s total",
    }[sr_mode])
    st.divider()

    if input_mode == "🗺️ Pick Location on Map":
        st.markdown("### 📅 Date Range")
        date1 = st.date_input("📅 Date T1 (earlier)", value=None)
        date2 = st.date_input("📅 Date T2 (later)",   value=None)
        st.divider()
        st.markdown("### 📍 Manual Coordinates")
        manual_lat = st.number_input("Latitude",  value=28.6, format="%.4f")
        manual_lon = st.number_input("Longitude", value=77.2, format="%.4f")
        st.divider()
        POPULAR = {
            "🗺️ Choose a city...":       None,
            "🏙️ Delhi, India":           (28.6,  77.2),
            "🌊 Mumbai, India":          (19.0,  72.8),
            "🌿 Bengaluru, India":       (12.9,  77.6),
            "🏔️ Himalayan Range":        (30.0,  79.5),
            "🌾 Punjab Farmlands":       (30.9,  75.8),
            "🏜️ Thar Desert":            (27.0,  71.0),
            "🌊 Ganga Delta, WB":        (22.0,  89.0),
            "🗼 Paris, France":          (48.85,  2.35),
            "🗽 New York, USA":          (40.71, -74.0),
            "🏝️ Maldives":               ( 4.17,  73.51),
            "🌴 Amazon Rainforest":      (-3.47, -62.21),
        }
        city = st.selectbox("🏙️ Quick City Select", list(POPULAR.keys()))

    st.divider()
    st.markdown("**Features:**")
    st.markdown("""
    - 🖼️ AI Prediction (Deep U-Net)
    - 🔬 4× Super-Resolution (1024px)
    - 🆚 AI vs Baseline
    - 🔵 Confidence Map
    - 🔴 Error Map
    - 🌿 NDVI Vegetation
    - ☁️ Cloud Detection
    - 📊 PSNR Chart (real metrics)
    - ⬇️ 4 Downloads
    """)
    st.divider()
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown("**BAH 2026 — PS12**")

# ══════════════════════════════════════════════════════════════════════════
# MODE A: UPLOAD YOUR OWN IMAGES
# ══════════════════════════════════════════════════════════════════════════
if input_mode == "📁 Upload My Own Images":

    st.markdown("## 📁 Upload Your Satellite Images")
    st.markdown("Drag and drop two images of the **same location** taken on **different dates**.")
    st.divider()

    col1, col2 = st.columns(2)
    img1_pil   = None
    img2_pil   = None

    with col1:
        st.markdown("### 📅 Day T1 — Earlier Date")
        file1 = st.file_uploader(
            "Drag & drop or click to upload Day T1",
            type=['png','jpg','jpeg'],
            key="upload_f1",
            help="Upload an earlier satellite image"
        )
        if file1:
            img1_pil = Image.open(file1).convert('RGB')
            st.image(img1_pil,
                     caption=f"✅ Day T1 — {img1_pil.size[0]}×{img1_pil.size[1]}px",
                     use_column_width=True)

    with col2:
        st.markdown("### 📅 Day T2 — Later Date")
        file2 = st.file_uploader(
            "Drag & drop or click to upload Day T2",
            type=['png','jpg','jpeg'],
            key="upload_f2",
            help="Upload a later satellite image"
        )
        if file2:
            img2_pil = Image.open(file2).convert('RGB')
            st.image(img2_pil,
                     caption=f"✅ Day T2 — {img2_pil.size[0]}×{img2_pil.size[1]}px",
                     use_column_width=True)

    st.divider()

    if img1_pil and img2_pil:
        st.success("✅ Both images ready!")
        st.info(f"ℹ️ Deep U-Net trains on your images on GPU, then super-resolves to 1024px and runs full analysis. Device: `{DEVICE}`")

        if st.button("🤖 Train AI and Run Full Analysis",
                     type="primary",
                     use_container_width=True,
                     key="upload_btn"):

            t1 = prepare_pil(img1_pil)
            t2 = prepare_pil(img2_pil)

            show_results(
                t1, t2,
                t1_display = img1_pil,
                t2_display = img2_pil,
                label1     = "Day T1 (uploaded)",
                label2     = "Day T2 (uploaded)",
                epochs     = epochs,
                sr_mode    = sr_mode,
                mode       = "upload"
            )
    else:
        st.info("👆 Upload both images above to get started!")
        st.markdown("### 💡 Test images you can use:")
        st.code("""
Day T1 → Documents/TempoSat_Project/data/real_satellite/fixed_T1.png
Day T2 → Documents/TempoSat_Project/data/real_satellite/fixed_T2.png
        """)

# ══════════════════════════════════════════════════════════════════════════
# MODE B: PICK LOCATION ON MAP
# ══════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## 🗺️ Pick Any Location on Earth")
    st.markdown("Click anywhere on the map to select a location. "
                "AI will download real Sentinel-2 satellite images and predict the missing day.")

    if not ee_ready:
        st.error("❌ Google Earth Engine not connected. Check your project ID in the code.")
        st.stop()

    st.divider()

    if city and POPULAR[city]:
        map_center = list(POPULAR[city])
    else:
        map_center = [manual_lat, manual_lon]

    m = folium.Map(
        location   = map_center,
        zoom_start = 5,
        tiles      = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr       = "Google Satellite"
    )
    folium.Marker(
        location = map_center,
        popup    = f"📍 {map_center[0]:.4f}, {map_center[1]:.4f}",
        icon     = folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    folium.Circle(
        location     = map_center,
        radius       = 15000,
        color        = '#FF4444',
        fill         = True,
        fill_opacity = 0.1,
        popup        = "Download area (~30km × 30km)"
    ).add_to(m)

    map_result  = st_folium(m, width=700, height=450, key="main_map")

    clicked_lat = map_center[0]
    clicked_lon = map_center[1]

    if map_result and map_result.get("last_clicked"):
        clicked_lat = map_result["last_clicked"]["lat"]
        clicked_lon = map_result["last_clicked"]["lng"]
        st.success(f"📍 Selected: **{clicked_lat:.4f}°N, {clicked_lon:.4f}°E**")
    else:
        st.info(f"📍 Current: **{clicked_lat:.4f}°N, {clicked_lon:.4f}°E** — Click map to change")

    st.divider()

    st.markdown("### 📋 Confirm Settings")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("📍 Latitude",  f"{clicked_lat:.4f}°")
    with c2: st.metric("📍 Longitude", f"{clicked_lon:.4f}°")
    with c3:
        if date1 and date2:
            st.metric("📅 Days Apart", f"{(date2-date1).days} days")
        else:
            st.metric("📅 Dates", "Not set yet")

    if date1 and date2 and date1 < date2:
        st.success(f"✅ Ready! Dates: **{date1}** and **{date2}**")
    else:
        st.warning("⚠️ Set two dates in the sidebar (T1 must be earlier than T2)")

    st.divider()

    if date1 and date2 and date1 < date2:
        if st.button("🛰️ Download Real Satellite Images + Run AI",
                     type="primary",
                     use_container_width=True,
                     key="map_btn"):

            os.makedirs("data/map_download", exist_ok=True)
            t1_path = "data/map_download/map_T1.tif"
            t2_path = "data/map_download/map_T2.tif"

            with st.spinner("⬇️ Downloading T1 from Google Earth Engine..."):
                try:
                    end1 = date1 + datetime.timedelta(days=10)
                    download_sentinel(clicked_lat, clicked_lon,
                                      str(date1), str(end1), t1_path)
                    st.success("✅ T1 downloaded!")
                except Exception as e:
                    st.error(f"❌ T1 download failed: {e}")
                    st.info("💡 Try different dates or choose a location with clearer skies.")
                    st.stop()

            with st.spinner("⬇️ Downloading T2 from Google Earth Engine..."):
                try:
                    end2 = date2 + datetime.timedelta(days=10)
                    download_sentinel(clicked_lat, clicked_lon,
                                      str(date2), str(end2), t2_path)
                    st.success("✅ T2 downloaded!")
                except Exception as e:
                    st.error(f"❌ T2 download failed: {e}")
                    st.info("💡 Try different dates or choose a location with clearer skies.")
                    st.stop()

            with st.spinner("🔄 Processing images..."):
                t1_raw = load_tif(t1_path)
                t2_raw = load_tif(t2_path)
                t1     = prepare_arr(t1_raw)
                t2     = prepare_arr(t2_raw)

            mid_date = date1 + datetime.timedelta(days=(date2-date1).days//2)
            st.markdown(f"## 📍 {clicked_lat:.4f}°N, {clicked_lon:.4f}°E")
            st.markdown(f"**T1:** {date1} → **Predicted:** {mid_date} → **T2:** {date2}")
            st.divider()

            show_results(
                t1, t2,
                t1_display = to_pil(t1_raw, size=400),
                t2_display = to_pil(t2_raw, size=400),
                label1     = str(date1),
                label2     = str(date2),
                epochs     = epochs,
                sr_mode    = sr_mode,
                mode       = "map"
            )

    else:
        st.info("👈 Set two dates in the sidebar then click the button!")
        st.markdown("### 💡 Suggested Locations & Dates:")
        st.markdown("""
| Location | Lat | Lon | T1 Date | T2 Date |
|---|---|---|---|---|
| Delhi | 28.6 | 77.2 | 2024-11-01 | 2024-11-25 |
| Mumbai | 19.0 | 72.8 | 2024-01-05 | 2024-01-25 |
| Punjab Farms | 30.9 | 75.8 | 2024-10-01 | 2024-10-25 |
| Amazon Forest | -3.47 | -62.21 | 2024-07-01 | 2024-07-25 |
        """)