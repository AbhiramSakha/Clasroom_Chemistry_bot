import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL = "google/flan-t5-base"
ADAPTER_PATH = "MyFinetunedModel"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

def generate_answer(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
