"""Cut AI tells from generated prose.

Distilled from the Cursor unslop skill
(https://github.com/cursor/plugins/blob/main/pstack/skills/unslop/SKILL.md)
so every model call — local or cloud — is held to the same bar.

Two layers, because a prompt rule the model ignores is not a rule:

1. ``UNSLOP_RULE`` is appended to system prompts (compact: prompt tokens cost).
2. ``unslop_text`` is a deterministic pass over Markdown we are about to store.
   It is a safety net, not a stylist. It does not invent opinions or facts.
"""

from __future__ import annotations

import re

# Kept short on purpose. Prepended to every enrich / brief / wiki call.
UNSLOP_RULE = (
    "Write like a practitioner, not a chatbot. Short sentences. Concrete nouns. "
    "Name the tradeoff. Take a position. "
    "No em dashes. No en dashes. No curly quotes. "
    "Do not use: delve, tapestry, pivotal, landscape, robust, leverage, harness, "
    "elevate, underscore, testament, vibrant, intricate, fostering, showcase. "
    "Do not open with: Here's the thing, Let's dive in, It's important to note, "
    "In today's, Of course, Certainly, I hope this helps. "
    "Do not write 'not just X, but Y'. Do not hedge twice. Skip filler. "
    "Have opinions. Vary sentence length. Be specific: names, versions, hardware, licenses."
)

_CURLY = str.maketrans({
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
})

# Em dash, or a spaced double-hyphen standing in for one. Not `---` rules.
_EM_DASH = re.compile(r"\s*[\u2014]\s*|\s+--\s+")
_EN_DASH = re.compile(r"\u2013")

_CHATBOT_LINE = re.compile(
    r"^(?:of course!?|certainly!?|absolutely!?|great question!?|"
    r"i hope this helps!?|let me know if you (?:need|have|would like).*"
    r"|here(?:'s| is) the thing:?|let'?s (?:dive|take a deep dive) in:?|"
    r"it'?s important to note that|in today'?s (?:fast[- ]paced |rapidly changing )?world[,:]?|"
    r"as an ai (?:language )?model[,:]?|"
    r"hope this helps!?)\s*",
    re.IGNORECASE,
)

_FILLER: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bin order to\b", re.IGNORECASE), "to"),
    (re.compile(r"\bdue to the fact that\b", re.IGNORECASE), "because"),
    (re.compile(r"\bit is important to note that\s*", re.IGNORECASE), ""),
    (re.compile(r"\bit'?s worth (?:noting|mentioning) that\s*", re.IGNORECASE), ""),
    (re.compile(r"\bat the end of the day,?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bin the event that\b", re.IGNORECASE), "if"),
    (re.compile(r"\ba wide range of\b", re.IGNORECASE), "many"),
    (re.compile(r"\bplays a (?:vital|crucial|important) role in\b", re.IGNORECASE), "matters for"),
    (re.compile(r"\bnot only\s+(.+?)\s+but also\b", re.IGNORECASE), r"\1 and"),
    (re.compile(r"\butilize[sd]?\b", re.IGNORECASE), "use"),
    (re.compile(r"\bleverage[sd]?\b", re.IGNORECASE), "use"),
    (re.compile(r"\bfacilitate[sd]?\b", re.IGNORECASE), "help"),
    (re.compile(r"\bnumerous\b", re.IGNORECASE), "many"),
]

_AI_MAP = {
    "delve": "look",
    "delves": "looks",
    "delved": "looked",
    "delving": "looking",
    "tapestry": "mix",
    "pivotal": "important",
    "underscore": "show",
    "underscores": "shows",
    "underscored": "showed",
    "underscoring": "showing",
    "testament": "sign",
    "vibrant": "active",
    "intricate": "detailed",
    "fostering": "helping",
    "showcase": "show",
    "showcases": "shows",
    "showcased": "showed",
    "showcasing": "showing",
    "garner": "get",
    "garners": "gets",
    "garnered": "got",
    "garnering": "getting",
    "enduring": "lasting",
    "crucial": "key",
    "additionally": "also",
}

_AI_WORDS = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_AI_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def unslop_text(text: str) -> str:
    """Strip the cheapest AI tells from model Markdown. Meaning stays put."""
    if not text:
        return text

    text = text.translate(_CURLY)
    text = _EM_DASH.sub(", ", text)
    text = _EN_DASH.sub("-", text)

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        cleaned = _CHATBOT_LINE.sub("", stripped).lstrip()
        lines.append(indent + cleaned if cleaned else indent.rstrip())
    text = "\n".join(lines)

    for pat, repl in _FILLER:
        text = pat.sub(repl, text)

    def _ai_word(match: re.Match[str]) -> str:
        return _AI_MAP.get(match.group(0).lower(), match.group(0))

    text = _AI_WORDS.sub(_ai_word, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # ", ," from stacked substitutions; leading ", " on a line from an em dash
    # that opened a sentence.
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"^,\s*", "", text, flags=re.MULTILINE)
    return text
