# peft_patch.py
try:
    import accelerate.utils.memory as _mem

    if not hasattr(_mem, "clear_device_cache"):
        def clear_device_cache():
            return None

        _mem.clear_device_cache = clear_device_cache
except Exception:
    pass
