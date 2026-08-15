"""
Authentic 17th/18th-Century Barbados Notary & Deed Synthetic Generator
Engineered directly from linguistic analysis of Barbados Ground-Truth Archive (Train_Cleaned.csv):
- Exact Notary boilerplate: 'To all Xpian people', 'dd in the prsence of', 'Lett suite trouble eviction'
- Authentic 17th-c. commodities & currency: 'pounds of Tobaccoe', 'Sterling money of Barbados', 'muscovado sugar'
- Archaic orthography: 'househould stuffe', 'writeing', 'heires and assignes', 'wherof', 'lawfull', 'ffrancis'
- Historical superscripts & contractions: 'y^t', 'Cap^t.', 'Esq:^r', 'prsents', 'pnts', 'wch', 'sd', 'ye'
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

# Authentic Barbados Notary & Deed Formulas extracted from Ground-Truth
BARBADOS_FORMULAS = [
    "To all Xpian people to whome theis pnts shall come Know ye y^t I Cap^t. {NAME}",
    "To all Christian people to whome these presents Shall come I {NAME} of the Island of Barbados",
    "To all Christian People to whom this present wrighting shall come greeting",
    "Barbados- {DAY}th {MONTH} {YEAR} - Knowe all men by these prsents that I {NAME}",
    "the Parish of {PARISH} in the same Island {TITLE}",
    "withall houses edifices househould stuffe and implements of househould stuffe",
    "Signe Seale and Deliver the within writeing as his voluntary Act and Deed",
    "Sealed and dd in the prsence of, {NAME} -",
    "without the Lett suite trouble eviction molest or hindrance of any person",
    "to the only proper use benefitt and behoofe of him the said {NAME} his heires and assignes",
    "thousand pounds of Tobaccoe to me in hand paid by {NAME}",
    "Sterling money of Barbados to mee in hand paid & secured to be paid",
    "annity wherof is Masters under God for this present Voyage",
    "thereof my full power and lawfull authority for me and in my name",
    "Know yee that wee the said {NAME} and my wife for divers good causes and considerations",
    "Come, know yee that I {NAME} of the parish of {PARISH} in the Island of Barbados",
    "protesting against all and every the sd matters and things in the sd protest mentioned",
    "By this public act and Instrument of protest be it known and made manifest",
    "all that plantation or parcell of land situate lying and being in the parish of {PARISH}",
    "bounded on the East by the lands of the sd {NAME} and on the West by {NAME}",
    "all and singular the sd negroes slaves cattle horses coppers stills and utensils",
    "yielding and paying therefore yearly and every year the sum of {SUM} p. Ann:",
    "Either by my meanes or procurement Provided that the said {NAME} att anie tyme",
    "in witness wherof I have hereunto sett my hand and seale the day and year first above written",
    "Signed Sealed and delivered in the presence of us {NAME} and {NAME}",
]

PARISHES = ["St Michael", "St George", "St Philip", "St John", "St James", "St Peter", "St Lucy", "St Joseph", "St Andrew", "St Thomas", "Christ Church"]
NAMES = ["John Alleyne", "William Fortescue", "Thomas Applewhaite", "Richard Walter", "ffrancis Cockraine", "Edward Jordan", "Elizabeth Walrond", "James Dotin", "Samuel Husbands", "Henry Lascelles", "John Richardson", "William Whittaker", "George Dane", "John Holder"]
TITLES = ["Esq:^r", "Esqr", "Gent:", "Gent", "Cap^t.", "Planter", "Merchant", "Executrix"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def build_barbados_corpus():
    print("Building Ingredient 1: Authentic Barbados Archive Corpus...")
    corpus = []

    # 1. Add all clean ground-truth lines from Train_Cleaned.csv
    train_csv = os.path.join(PROJECT_ROOT, 'Train_Cleaned.csv')
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        for t in df['Target'].dropna().tolist():
            t = str(t).strip()
            if len(t) > 3:
                corpus.append(t)
        print(f"  [OK] Loaded {len(df):,} verbatim lines from Train_Cleaned.csv")

    # 2. Add Barbados notary template permutations
    for template in BARBADOS_FORMULAS:
        for _ in range(120):
            t = template.format(
                NAME=random.choice(NAMES),
                PARISH=random.choice(PARISHES),
                TITLE=random.choice(TITLES),
                MONTH=random.choice(MONTHS),
                DAY=random.randint(1, 28),
                YEAR=random.randint(1640, 1795),
                SUM=f"{random.randint(5, 500)} Pounds",
            )
            # Inject verbatim Barbados contractions
            t = t.replace(" which ", random.choice([" which ", " wch ", " w^ch ", " which "]))
            t = t.replace(" said ", random.choice([" said ", " sd ", " Said ", " sd "]))
            t = t.replace(" the ", random.choice([" the ", " ye ", " the ", " theis "]))
            t = t.replace(" that ", random.choice([" that ", " y^t ", " that ", " yt "]))
            t = t.replace(" and ", random.choice([" and ", " & ", " & "]))
            corpus.append(t)

    # 3. Sub-line phrase slicing (lines of 4 to 14 words matching competition line lengths)
    all_words = " ".join(corpus).split()
    for _ in range(20000):
        length = random.randint(4, 15)
        start = random.randint(0, max(0, len(all_words) - length - 1))
        phrase = " ".join(all_words[start:start + length])
        if len(phrase) > 10:
            corpus.append(phrase)

    unique_corpus = sorted(list(set(corpus)))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        for line in unique_corpus:
            f.write(f"{line}\n")

    print(f"  [OK] Saved {len(unique_corpus):,} authentic Barbados phrases to: {CORPUS_PATH}\n")
    return unique_corpus


def setup_fonts():
    print("Setting Up Ingredient 2: Historical Cursive Fonts...")
    os.makedirs(FONTS_DIR, exist_ok=True)
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
    }
    valid_fonts = []
    for font_name, url in HISTORICAL_FONTS.items():
        dst_path = os.path.join(FONTS_DIR, font_name)
        if not os.path.exists(dst_path) or os.path.getsize(dst_path) < 1000:
            try:
                urllib.request.urlretrieve(url, dst_path)
            except Exception:
                pass
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1000:
            valid_fonts.append(dst_path)
    print(f"  [OK] Ready with {len(valid_fonts)} Historical Cursive Fonts\n")
    return valid_fonts


def setup_backgrounds(num_backgrounds=30):
    print("Setting Up Ingredient 3: Procedural Parchment Textures...")
    os.makedirs(BACKGROUNDS_DIR, exist_ok=True)
    bg_paths = []
    for i in range(num_backgrounds):
        bg_file = os.path.join(BACKGROUNDS_DIR, f"parchment_{i:02d}.jpg")
        if not os.path.exists(bg_file):
            w, h = 1800, 300
            base_r, base_g, base_b = random.randint(220, 248), random.randint(205, 235), random.randint(170, 205)
            canvas = np.full((h, w, 3), [base_r, base_g, base_b], dtype=np.float32)
            grain = np.random.normal(0, random.uniform(4.0, 8.0), (h, w, 3))
            canvas = np.clip(canvas + grain, 0, 255).astype(np.uint8)
            img = Image.fromarray(canvas)

            draw = ImageDraw.Draw(img)
            for _ in range(random.randint(2, 6)):
                sx, sy = random.randint(0, w), random.randint(0, h)
                rad = random.randint(5, 25)
                draw.ellipse([sx - rad, sy - rad, sx + rad, sy + rad], fill=(base_r - 40, base_g - 45, base_b - 50))

            img = img.filter(ImageFilter.GaussianBlur(radius=0.9))
            img.save(bg_file, quality=95)
        bg_paths.append(bg_file)
    print(f"  [OK] Ready with {len(bg_paths)} Parchment Backgrounds\n")
    return bg_paths


def render_single_crop(text: str, font_path: str, bg_path: str, target_height: int = 128) -> Image.Image:
    font_size = random.randint(48, 70)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGBA", (1, 1))
    draw_d = ImageDraw.Draw(dummy)
    bbox = draw_d.textbbox((0, 0), text, font=font)
    text_w = max(120, bbox[2] - bbox[0] + 70)
    text_h = max(50, bbox[3] - bbox[1] + 45)

    text_layer = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    t_draw = ImageDraw.Draw(text_layer)

    ink_color = (random.randint(30, 70), random.randint(22, 55), random.randint(18, 45), random.randint(215, 255))
    t_draw.text((35 - bbox[0], 22 - bbox[1]), text, font=font, fill=ink_color)

    shear = random.uniform(-0.16, 0.16)
    text_layer = text_layer.transform(text_layer.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), resample=Image.Resampling.BICUBIC)

    bg_img = Image.open(bg_path).convert("RGB")
    crop_x = random.randint(0, max(1, bg_img.width - text_layer.width))
    crop_y = random.randint(0, max(1, bg_img.height - text_layer.height))
    bg_tile = bg_img.crop((crop_x, crop_y, crop_x + text_layer.width, crop_y + text_layer.height))
    if bg_tile.size != text_layer.size:
        bg_tile = bg_tile.resize(text_layer.size, Image.Resampling.BICUBIC)

    bg_tile.paste(text_layer, (0, 0), text_layer)
    aspect = bg_tile.width / float(bg_tile.height)
    new_w = max(64, int(round(target_height * aspect)))
    return bg_tile.resize((new_w, target_height), Image.Resampling.BICUBIC)


def worker_task(args):
    texts, start_idx, font_files, bg_files = args
    results = []
    for i, text in enumerate(texts):
        img_id = f"synth_bb_{start_idx + i:06d}"
        font_p = random.choice(font_files)
        bg_p = random.choice(bg_files)
        try:
            img = render_single_crop(text, font_p, bg_p, target_height=128)
            dst_path = os.path.join(SYNTH_IMG_DIR, f"{img_id}.jpg")
            img.save(dst_path, format="JPEG", quality=92)
            results.append({"ID": img_id, "Target": text})
        except Exception:
            continue
    return results


def run_pipeline(num_samples: int = 50000, num_workers: int = None):
    print("==================================================================")
    print(f" AUTHENTIC BARBADOS 50,000-LINE SYNTHETIC GENERATOR ")
    print("==================================================================")

    os.makedirs(SYNTH_IMG_DIR, exist_ok=True)
    corpus = build_barbados_corpus()
    fonts = setup_fonts()
    backgrounds = setup_backgrounds(30)

    selected_texts = [random.choice(corpus) for _ in range(num_samples)]
    if num_workers is None:
        num_workers = min(16, os.cpu_count() or 4)

    batch_size = max(100, num_samples // (num_workers * 4))
    tasks = []
    for chunk_start in range(0, num_samples, batch_size):
        chunk = selected_texts[chunk_start:chunk_start + batch_size]
        tasks.append((chunk, chunk_start, fonts, backgrounds))

    print(f"Rendering {num_samples:,} images in parallel across {num_workers} CPU cores...\n")
    all_records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Baking Barbados Cursive"):
            all_records.extend(f.result())

    df = pd.DataFrame(all_records)
    df.to_csv(SYNTH_CSV_PATH, index=False)

    print("\n--- SYNTHETIC BAKING COMPLETE ---")
    print(f"[OK] Generated & Saved : {len(df):,} Images in {SYNTH_IMG_DIR}")
    print(f"[OK] Metadata CSV Saved: {SYNTH_CSV_PATH}")


if __name__ == '__main__':
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
    run_pipeline(num_samples=count)
