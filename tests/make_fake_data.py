"""Writes small synthetic files in the exact UCI Drug Review format, for offline smoke tests.
    python tests/make_fake_data.py data_fake && python -m src.train --data_dir data_fake --epochs 2"""
import os
import random
import sys

import pandas as pd

POS = ["works great", "no side effects", "life changing", "highly recommend", "feel amazing"]
NEU = ["it is okay", "some improvement", "mixed results", "not sure yet", "average"]
NEG = ["terrible headache", "did not work", "awful nausea", "stopped taking it", "worst experience"]
FILL = "i was prescribed this medication by my doctor for my condition and took it for a few weeks".split()
CONDS = ["Depression", "Birth Control", "Pain", "Anxiety", "Acne", "Insomnia"]


def review(rating, rng):
    bank = NEG if rating <= 4 else NEU if rating <= 6 else POS
    words = rng.sample(FILL, rng.randint(5, len(FILL)))
    words.insert(rng.randrange(len(words)), rng.choice(bank))
    return '"' + " ".join(words).replace("i was", "I&#039;ve been") + '"'


def write(path, n, rng):
    ratings = rng.choices(range(1, 11), weights=[13, 4, 4, 3, 5, 4, 6, 12, 17, 32], k=n)
    rows = [{"": 100000 + k, "drugName": f"Drug{rng.randint(1, 40)}", "condition": rng.choice(CONDS),
             "review": review(r, rng), "rating": float(r), "date": "May 20, 2012", "usefulCount": rng.randint(0, 50)}
            for k, r in enumerate(ratings)]
    rows += rows[:20]  # a few duplicates, like the real data
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data_fake"
    os.makedirs(out, exist_ok=True)
    rng = random.Random(0)
    write(os.path.join(out, "drugsComTrain_raw.tsv"), 3000, rng)
    write(os.path.join(out, "drugsComTest_raw.tsv"), 800, rng)
    print("fake data ->", out)
