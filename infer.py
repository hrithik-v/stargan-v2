"""
StarGAN v2 Inference Script (Latent-Guided)
Translates 7 random images into 5 target domains using the mapping network.
"""
import os
import random
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from core.model import build_model
from core.utils import denormalize

# --------- CONFIG ---------
IMG_SIZE = 256
NUM_DOMAINS = 5
LATENT_DIM = 16
STYLE_DIM = 64
MAX_CONV_DIM = 512
HIDDEN_DIM = 512

EXPR_NAME = 'First_Run' 
CKPT_NO = '003000' 
USE_EMA = False  # Use EMA weights for better quality

if USE_EMA:
    CHECKPOINT_PATH = f'expr/checkpoints/{EXPR_NAME}/{CKPT_NO}_nets_ema.ckpt'
else:
    CHECKPOINT_PATH = f'expr/checkpoints/{EXPR_NAME}/{CKPT_NO}_nets.ckpt'

OUT_PATH = f'samples/{EXPR_NAME}/{CKPT_NO}.png'
SRC_DIR = '/kaggle/input/five-weather-23k'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# --------- SETUP ---------
# torch.manual_seed(SEED)
# random.seed(SEED)
# torch.cuda.manual_seed_all(SEED)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# --------- LOAD MODEL ---------
class Args:
    img_size = IMG_SIZE
    num_domains = NUM_DOMAINS
    latent_dim = LATENT_DIM
    style_dim = STYLE_DIM
    max_conv_dim = MAX_CONV_DIM
    w_hpf = 1.0
    ema = USE_EMA
    wing_path = ''
    hidden_dim = HIDDEN_DIM
    seg_classes = 7
args = Args()

nets, nets_ema = build_model(args)

if USE_EMA:
    nets = nets_ema

# Move models to device
nets.generator.to(DEVICE)
nets.mapping_network.to(DEVICE)

# Load checkpoint
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

def load_state(model, state):
    if hasattr(model, 'module'):
        if 'module' in state:
            state = state['module']
        else:
            state = {f"module.{k}": v for k, v in state.items()}
    elif 'module' in state:
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)

load_state(nets.generator, ckpt['generator'])
load_state(nets.mapping_network, ckpt['mapping_network'])

nets.generator.eval()
nets.mapping_network.eval()

# --------- SAMPLE IMAGES ---------
img_files = []
img_labels = []
for class_name in os.listdir(SRC_DIR):
    class_dir = os.path.join(SRC_DIR, class_name)
    if not os.path.isdir(class_dir) or class_name not in ['0', '1', '2', '3', '4']:
        continue
    for f in os.listdir(class_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_files.append(os.path.join(class_dir, f))
            img_labels.append(int(class_name))

# Randomly sample 7 images
sampled = random.sample(list(zip(img_files, img_labels)), 15)
sampled_files, sampled_labels = zip(*sampled)
imgs = [transform(Image.open(f).convert('RGB')) for f in sampled_files]
imgs = torch.stack(imgs).to(DEVICE)

# --------- INFERENCE ---------
results = []
with torch.no_grad():
    for i in range(15):
        row = [denormalize(imgs[i].cpu())]  # Start with original
        x = imgs[i].unsqueeze(0)
        for y_trg in range(NUM_DOMAINS):
            y = torch.tensor([y_trg]).to(DEVICE)
            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            s_trg = nets.mapping_network(z, y)
            out = nets.generator(x, s_trg)
            row.append(denormalize(out[0][0].cpu()))
        results.append(torch.stack(row))

# --------- SAVE OUTPUT ---------
grid = make_grid(torch.cat(results, dim=0), nrow=NUM_DOMAINS + 1, padding=2)
if not os.path.exists(os.path.dirname(OUT_PATH)):
    os.makedirs(os.path.dirname(OUT_PATH))
save_image(grid, OUT_PATH)
print(f"Saved inference results to {OUT_PATH}")
