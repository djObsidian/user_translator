"""Language codes accepted by --to. Hy-MT2 wants full English language names in English prompts."""

LANGS = {
    "zh": "Chinese",
    "zh-hant": "Traditional Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ru": "Russian",
    "uk": "Ukrainian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "pl": "Polish",
    "cs": "Czech",
    "nl": "Dutch",
    "tr": "Turkish",
    "ar": "Arabic",
    "fa": "Persian",
    "he": "Hebrew",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "ms": "Malay",
    "id": "Indonesian",
    "tl": "Filipino",
    "km": "Khmer",
    "my": "Burmese",
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "gu": "Gujarati",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "kk": "Kazakh",
    "mn": "Mongolian",
    "bo": "Tibetan",
    "ug": "Uyghur",
}

# Scripts written without spaces between sentences
NO_SPACE = {"zh", "zh-hant", "yue", "ja"}


def resolve(target: str) -> tuple[str, str]:
    """'ru' or 'Russian' -> ('ru', 'Russian'). Unknown names pass through as-is."""
    key = target.strip().lower()
    if key in LANGS:
        return key, LANGS[key]
    for code, name in LANGS.items():
        if name.lower() == key:
            return code, name
    return key, target.strip()
