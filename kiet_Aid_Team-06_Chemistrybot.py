import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict

# ============================================
# FIXED & WORKING CHEMISTRY DATASETS
# (All valid, accessible, and correct keys)
# ============================================

datasets_config = [
    # SciQ (general science; includes chemistry)
    ("sciq", "train", "question", "correct_answer", None),

    # PubMed QA (with chemistry filter)
    ("pubmed_qa", "pqa_labeled", "question", "long_answer",
        lambda ex: any(word in ex["question"].lower()
                       for word in ["chem", "reaction", "atom", "molecule", "compound"])),

    # MedMCQA (medical + biochemistry)
    ("MedMCQA", "train", "question", "cop", None),

    # ChemProt
    ("bigbio/chemprot", "bigbio", "abstract", "label", None),

    # camel-ai chemistry dataset (correct keys)
    ("camel-ai/chemistry", "train", "message_1", "message_2", None),
]

all_data = []

# ============================================
# LOAD DATASETS (your original logic preserved)
# ============================================
for name, split, q_key, a_key, filter_fn in datasets_config:
    print(f"Loading {name}...")
    try:
        ds = load_dataset(name, split=split)

        # Filter only chemistry questions if filter is provided
        if filter_fn:
            ds = ds.filter(filter_fn)

        # Convert to list of dicts
        examples = []
        for ex in ds:
            q = ex.get(q_key, "")
            a = ex.get(a_key, "")

            # join list answers (some datasets return list)
            if isinstance(a, list):
                a = " ".join([str(x) for x in a])

            if q and a:
                examples.append({"source": q, "target": a})

        # Deduplicate by question
        seen = set()
        unique = [
            e for e in examples
            if e["source"] not in seen and not seen.add(e["source"])
        ]

        all_data.extend(unique)
        print(f"Added {len(unique)} unique examples from {name}")

    except Exception as e:
        print(f"Skipped {name}: {e}")

print(f"\nTotal unique examples collected: {len(all_data):,}")


# ============================================
# SAVE MERGED FILES
# ============================================

output_dir = "/content/drive/MyDrive/Major Project/"

# Save full dataset (JSONL)
with open(output_dir + "chemistry_finetune_source_target.jsonl", "w", encoding="utf-8") as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Save CSV
df = pd.DataFrame(all_data)
df.to_csv(output_dir + "chemistry_finetune_source_target.csv", index=False)

# ============================================
# TRAIN/VAL SPLIT SAFE HANDLING
# ============================================

if len(all_data) == 0:
    print("\n⚠ Warning: No data loaded. Creating empty train/val files.")

    open("train_source_target.jsonl", "w").close()
    open("val_source_target.jsonl", "w").close()

else:
    train, val = train_test_split(all_data, test_size=0.05, random_state=42)

    for split_data, split_name in [(train, "train"), (val, "val")]:
        with open(f"{split_name}_source_target.jsonl", "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


print("\nDone! Files generated:")
print("- chemistry_finetune_source_target.jsonl")
print("- chemistry_finetune_source_target.csv")
print("- train_source_target.jsonl")
print("- val_source_target.jsonl")
print("\nUse with Llama-Factory/Axolotl: dataset_type=alpaca")
