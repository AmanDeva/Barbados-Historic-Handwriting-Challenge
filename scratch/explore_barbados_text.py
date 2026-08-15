import pandas as pd
from collections import Counter
import re

df = pd.read_csv('Train_Cleaned.csv')
print("=== BARBADOS DATASET TEXT EXPLORATION ===")
print(f"Total Rows: {len(df):,}")
print(f"Unique Lines: {df['Target'].nunique():,}")
print(f"Average Length: {df['Target'].str.len().mean():.1f} chars")
print(f"Max Length: {df['Target'].str.len().max()} chars")

print("\n--- 20 REAL GROUND-TRUTH SAMPLES FROM Train_Cleaned.csv ---")
for i, t in enumerate(df['Target'].sample(20, random_state=42)):
    print(f"{i+1:2d}. {t}")

all_text = " ".join(df['Target'].dropna().astype(str))
words = all_text.split()
word_counts = Counter(words)

print("\n--- TOP 35 MOST COMMON TOKENS IN BARBADOS CHALLENGE ---")
for w, c in word_counts.most_common(35):
    print(f"  {w:<15}: {c:>5} occurrences")

# Check for historical shorthand abbreviations in Barbados dataset
shorthands = ["wch", "sd", "ye", "yt", "&c", "Esqr", "Gent:", "p. Ann:", "vizt", "viz:", "Lds", "Xber", "7ber", "8ber", "9ber", "Ditto", "Do"]
print("\n--- HISTORICAL SHORTHAND FREQUENCIES IN REAL DATA ---")
for sh in shorthands:
    matches = len(re.findall(r'\b' + re.escape(sh) + r'\b', all_text, flags=re.IGNORECASE))
    print(f"  '{sh}': {matches} times in dataset")

# Identify recurring legal formula clauses
print("\n--- TOP 10 MOST FREQUENT N-GRAMS (PHRASES) ---")
from collections import defaultdict
trigrams = Counter()
for text in df['Target'].dropna():
    toks = str(text).split()
    for j in range(len(toks) - 3):
        trigrams[" ".join(toks[j:j+4])] += 1

for phrase, count in trigrams.most_common(12):
    print(f"  \"{phrase}\": {count} times")
