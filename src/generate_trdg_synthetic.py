"""
Complete 50,000-Line Procedural Synthetic Historical Document Generator (TRDG Pipeline)
Assembles:
1. Ingredient 1: 18th-Century Archaic Legal Corpus (from Train_Cleaned.csv + historical deed templates)
2. Ingredient 2: 12 Historical Cursive/Secretary/Copperplate TTF Fonts
3. Ingredient 3: 30 Procedural Aged Parchment Textures with foxing stains & iron gall ink degradation
4. Orchestrator: Multi-Core TRDG / Pillow Rendering Engine
"""

import os
import sys
import random
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FONTS_DIR = os.path.join(DATA_DIR, 'fonts')
BACKGROUNDS_DIR = os.path.join(DATA_DIR, 'backgrounds')
SYNTH_IMG_DIR = os.path.join(DATA_DIR, 'synthetic_images')
CORPUS_PATH = os.path.join(DATA_DIR, 'archaic_corpus.txt')
SYNTH_CSV_PATH = os.path.join(DATA_DIR, 'synthetic_train.csv')

# Top 12 Historical Cursive & Calligraphy Google Fonts
HISTORICAL_FONTS = {
    "PinyonScript-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/pinyonscript/PinyonScript-Regular.ttf",
    "Tangerine-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/tangerine/Tangerine-Regular.ttf",
    "Tangerine-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/tangerine/Tangerine-Bold.ttf",
    "GreatVibes-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/greatvibes/GreatVibes-Regular.ttf",
    "AlexBrush-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/alexbrush/AlexBrush-Regular.ttf",
    "Allura-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/allura/Allura-Regular.ttf",
    "MonsieurLaDoulaise-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/monsieurladoulaise/MonsieurLaDoulaise-Regular.ttf",
    "Meddon.ttf": "https://github.com/google/fonts/raw/main/ofl/meddon/Meddon.ttf",
    "HomemadeApple-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/homemadeapple/HomemadeApple-Regular.ttf",
    "MarckScript-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/marckscript/MarckScript-Regular.ttf",
    "Satisfy-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/satisfy/Satisfy-Regular.ttf",
    "DancingScript-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf",
}

# 18th-century Barbados legal deed templates
LEGAL_FORMULAS = [
    "Know all men by these presents that I {NAME} of the parish of {PARISH} in the Island of Barbados",
    "To have and to hold all and singular the sd {NOUN} and premises unto the sd {NAME}",
    "In witness whereof the parties to these presents have hereunto interchangeably set their hands and seals",
    "Signed sealed and delivered in the presence of us {NAME} and {NAME}",
    "yielding and paying therefore yearly and every year the sum of {SUM} p. Ann:",
    "all that plantation or tract of land situate lying and being in the parish of {PARISH} containing by estimation {NUM} acres",
    "together with all houses outhouses edifices buildings barns stables and hereditaments",
    "and also all that parcel of land bounded on the East by the lands of the sd {NAME}",
    "which said indenture was duly proved and recorded in the Secretary's Office of this Island",
    "Be it remembered that on the {DAY} day of {MONTH} in the year of our Lord {YEAR}",
    "the said {NAME} doth hereby for himself his heirs executors and administrators covenant grant and agree",
    "and the sd {NAME} for the considerations aforesaid hath bargained sold aliened and confirmed",
    "all and every the negro and other slaves mentioned and expressed in the schedule hereunto annexed",
    "protesting against all and every the sd matters and things in the sd protest mentioned",
    "By this public act and instrument of protest be it known and made manifest unto all whom it may concern",
    "to the only proper use and behoof of him the said {NAME} his heirs and assigns forever",
    "free and clear of and from all and all manner of former and other gifts grants bargains sales mortgages",
]

PARISHES = ["St. Michael", "St. George", "St. Philip", "St. John", "St. James", "St. Peter", "St. Lucy", "St. Joseph", "St. Andrew", "St. Thomas", "Christ Church"]
NAMES = ["John Alleyne Esqr", "William Fortescue Gent:", "Thomas Applewhaite", "Richard Walter Esqr", "Edward Jordan", "Elizabeth Walrond Executrix", "James Dotin", "Samuel Husbands", "Henry Lascelles", "Francis Ford"]
NOUNS = ["plantation", "lands", "hereditaments", "premises", "slaves", "negroes", "sugar works", "coppers and stills", "dwelling house", "appurtenances"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


# ==============================================================================
# INGREDIENT 1: BUILD ARCHAIC 18TH-CENTURY CORPUS
# ==============================================================================
def build_archaic_corpus():
    print("Building Ingredient 1: 18th-Century Archaic Corpus...")
    corpus = []

    # 1. Ground Truth from Train_Cleaned.csv
    train_csv = os.path.join(PROJECT_ROOT, 'Train_Cleaned.csv')
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        for text in df['Target'].dropna().tolist():
            t = str(text).strip()
            if len(t) > 3:
                corpus.append(t)
        print(f"  ✓ Extracted {len(df):,} lines from Train_Cleaned.csv")

    # 2. Procedural 18th-century formula generation with shorthand contractions
    for template in LEGAL_FORMULAS:
        for _ in range(80):
            t = template.format(
                NAME=random.choice(NAMES),
                PARISH=random.choice(PARISHES),
                NOUN=random.choice(NOUNS),
                MONTH=random.choice(MONTHS),
                DAY=random.randint(1, 28),
                YEAR=random.randint(1710, 1798),
                SUM=f"{random.randint(5, 600)} Pounds Current Money",
                NUM=random.randint(5, 1200)
            )
            # Inject historical shorthand abbreviations
            t = t.replace(" which ", random.choice([" which ", " wch ", " wch "]))
            t = t.replace(" said ", random.choice([" said ", " sd ", " sd "]))
            t = t.replace(" the ", random.choice([" the ", " ye ", " the "]))
            t = t.replace(" that ", random.choice([" that ", " yt ", " that "]))
            t = t.replace(" and ", random.choice([" and ", " & ", " & "]))
            corpus.append(t)

    # 3. N-gram sub-phrase segmentation (lines of 4 to 15 words)
    all_tokens = " ".join(corpus).split()
    for _ in range(15000):
        length = random.randint(4, 16)
        start = random.randint(0, max(0, len(all_tokens) - length - 1))
        phrase = " ".join(all_tokens[start:start + length])
        if len(phrase) > 8:
            corpus.append(phrase)

    unique_corpus = sorted(list(set(corpus)))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        for line in unique_corpus:
            f.write(f"{line}\n")

    print(f"  [OK] Saved {len(unique_corpus):,} unique archaic lines to: {CORPUS_PATH}\n")
    return unique_corpus


# ==============================================================================
# INGREDIENT 2: DOWNLOAD HISTORICAL TTF FONTS
# ==============================================================================
def setup_fonts():
    print("Setting Up Ingredient 2: Historical Cursive Fonts...")
    os.makedirs(FONTS_DIR, exist_ok=True)
    valid_fonts = []
    for font_name, url in HISTORICAL_FONTS.items():
        dst_path = os.path.join(FONTS_DIR, font_name)
        if not os.path.exists(dst_path) or os.path.getsize(dst_path) < 1000:
            try:
                urllib.request.urlretrieve(url, dst_path)
            except Exception as e:
                pass
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1000:
            valid_fonts.append(dst_path)

    print(f"  [OK] Ready with {len(valid_fonts)} Historical Cursive Fonts in: {FONTS_DIR}\n")
    return valid_fonts


# ==============================================================================
# INGREDIENT 3: GENERATE PROCEDURAL AGED PARCHMENT BACKGROUNDS
# ==============================================================================
def setup_parchment_backgrounds(num_backgrounds: int = 30):
    print("Setting Up Ingredient 3: Procedural Parchment Textures...")
    os.makedirs(BACKGROUNDS_DIR, exist_ok=True)
    bg_paths = []

    for i in range(num_backgrounds):
        bg_file = os.path.join(BACKGROUNDS_DIR, f"parchment_{i:02d}.jpg")
        if not os.path.exists(bg_file):
            w, h = 1800, 300
            base_r = random.randint(220, 248)
            base_g = random.randint(205, 235)
            base_b = random.randint(170, 205)

            # Paper fiber grain
            canvas = np.full((h, w, 3), [base_r, base_g, base_b], dtype=np.float32)
            grain = np.random.normal(0, random.uniform(4.0, 8.0), (h, w, 3))
            canvas = np.clip(canvas + grain, 0, 255).astype(np.uint8)
            img = Image.fromarray(canvas)

            # Foxing stains & aging spots
            draw = ImageDraw.Draw(img)
            for _ in range(random.randint(2, 6)):
                sx, sy = random.randint(0, w), random.randint(0, h)
                rad = random.randint(5, 25)
                draw.ellipse([sx - rad, sy - rad, sx + rad, sy + rad], fill=(base_r - 40, base_g - 45, base_b - 50))

            img = img.filter(ImageFilter.GaussianBlur(radius=0.9))
            img.save(bg_file, quality=95)

        bg_paths.append(bg_file)

    print(f"  [OK] Ready with {len(bg_paths)} Parchment Backgrounds in: {BACKGROUNDS_DIR}\n")
    return bg_paths


# ==============================================================================
# SYNTHETIC RENDERING ENGINE (FUSING TEXT + FONTS + PARCHMENT + INK)
# ==============================================================================
def render_single_synthetic_crop(text: str, font_path: str, bg_path: str, target_height: int = 128) -> Image.Image:
    """Renders text in historical iron gall ink with slant, baseline jitter, and parchment background."""
    font_size = random.randint(48, 70)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Calculate text bounding box
    dummy = Image.new("RGBA", (1, 1))
    draw_d = ImageDraw.Draw(dummy)
    bbox = draw_d.textbbox((0, 0), text, font=font)
    text_w = max(120, bbox[2] - bbox[0] + 70)
    text_h = max(50, bbox[3] - bbox[1] + 45)

    # Transparent text layer
    text_layer = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    t_draw = ImageDraw.Draw(text_layer)

    # Iron gall ink: dark brown/sepia with opacity jitter
    ink_color = (
        random.randint(30, 70),
        random.randint(22, 55),
        random.randint(18, 45),
        random.randint(215, 255)
    )
    t_draw.text((35 - bbox[0], 22 - bbox[1]), text, font=font, fill=ink_color)

    # Cursive shear slant (-10 to +10 degrees)
    shear = random.uniform(-0.16, 0.16)
    text_layer = text_layer.transform(
        text_layer.size,
        Image.AFFINE,
        (1, shear, 0, 0, 1, 0),
        resample=Image.Resampling.BICUBIC
    )

    # Crop random tile from parchment background
    bg_img = Image.open(bg_path).convert("RGB")
    max_x = max(1, bg_img.width - text_layer.width)
    max_y = max(1, bg_img.height - text_layer.height)
    crop_x = random.randint(0, max_x)
    crop_y = random.randint(0, max_y)

    bg_tile = bg_img.crop((crop_x, crop_y, crop_x + text_layer.width, crop_y + text_layer.height))
    if bg_tile.size != text_layer.size:
        bg_tile = bg_tile.resize(text_layer.size, Image.Resampling.BICUBIC)

    # Composite ink onto parchment
    bg_tile.paste(text_layer, (0, 0), text_layer)

    # Scale to target height preserving aspect ratio
    aspect = bg_tile.width / float(bg_tile.height)
    new_w = max(64, int(round(target_height * aspect)))
    final_img = bg_tile.resize((new_w, target_height), Image.Resampling.BICUBIC)

    return final_img


def worker_task(args):
    texts, start_idx, font_files, bg_files = args
    results = []
    for i, text in enumerate(texts):
        img_id = f"synth_{start_idx + i:06d}"
        font_p = random.choice(font_files)
        bg_p = random.choice(bg_files)

        try:
            img = render_single_synthetic_crop(text, font_p, bg_p, target_height=128)
            dst_path = os.path.join(SYNTH_IMG_DIR, f"{img_id}.jpg")
            img.save(dst_path, format="JPEG", quality=92)
            results.append({"ID": img_id, "Target": text})
        except Exception:
            continue
    return results


def run_pipeline(num_samples: int = 50000, num_workers: int = None):
    print("==================================================================")
    print(f" 50,000-LINE PROCEDURAL SYNTHETIC GENERATION PIPELINE ")
    print("==================================================================")

    os.makedirs(SYNTH_IMG_DIR, exist_ok=True)
    corpus = build_archaic_corpus()
    fonts = setup_fonts()
    backgrounds = setup_parchment_backgrounds(30)

    selected_texts = [random.choice(corpus) for _ in range(num_samples)]
    if num_workers is None:
        num_workers = min(16, os.cpu_count() or 4)

    batch_size = max(100, num_samples // (num_workers * 4))
    tasks = []
    for chunk_start in range(0, num_samples, batch_size):
        chunk = selected_texts[chunk_start:chunk_start + batch_size]
        tasks.append((chunk, chunk_start, fonts, backgrounds))

    print(f"Baking {num_samples:,} images in parallel across {num_workers} CPU cores...\n")
    all_records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Rendering Synthetic Lines"):
            all_records.extend(f.result())

    df = pd.DataFrame(all_records)
    df.to_csv(SYNTH_CSV_PATH, index=False)

    print("\n--- SYNTHETIC BAKING COMPLETE ---")
    print(f"[OK] Generated & Saved : {len(df):,} Images in {SYNTH_IMG_DIR}")
    print(f"[OK] Metadata CSV Saved: {SYNTH_CSV_PATH}")


if __name__ == '__main__':
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
    run_pipeline(num_samples=count)
