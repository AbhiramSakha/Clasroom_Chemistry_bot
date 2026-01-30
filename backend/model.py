import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ----------------------------
# Performance safety (Railway)
# ----------------------------
torch.set_num_threads(1)

BASE_MODEL = "google/flan-t5-base"
ADAPTER_PATH = "MyFinetunedModel"

# ----------------------------
# Lazy-loaded globals
# ----------------------------
tokenizer = None
model = None


def load_model():
    global tokenizer, model

    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32
        )

        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()


def generate_answer(text: str):
    load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
