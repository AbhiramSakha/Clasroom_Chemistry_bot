import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL = "google/flan-t5-base"
ADAPTER_PATH = "MyFinetunedModel"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            low_cpu_mem_usage=True
        )
        _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        _model.eval()


def generate_answer(text: str) -> str:
    _load()

    inputs = _tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=128,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
