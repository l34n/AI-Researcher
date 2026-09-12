"""app.css under a third-party browser overlay extension.

A CRT filter, a VHS tint or a dark-mode inverter injects a `position: fixed`
full-viewport layer at the top of the stacking order, usually painting through
`mix-blend-mode`. The page cannot out-stack it and cannot remove it, so the
only budget it controls is its own: how much darkening app.css has already
spent before the extension's lands, and whether prose contrast is carried by
opaque colour or borrowed from whatever happens to be behind.

These tests read app.css as a stylesheet rather than as text: they resolve the
`:root` palette, composite the page's own CRT layers over each prose pair the
way the cascade actually would, then composite a worst-case extension overlay
on top and check the surviving WCAG ratio. The compositing model is derived
from the stylesheet's own declarations, so moving the reading column back under
the CRT layers, or dimming a prose token, fails here rather than in someone's
browser.
"""

from __future__ import annotations

import re

import pytest

from ai_researcher.config import PROJECT_ROOT

CSS_PATH = PROJECT_ROOT / "src" / "ai_researcher" / "web" / "static" / "app.css"

# The layer this page can never see coming: a full-viewport multiply at 25%
# black is the worst of the common CRT/night-shift overlay presets.
EXTENSION_BLACK_ALPHA = 0.25

PROSE_TOKENS = ("--text", "--text-mid", "--text-dim")

# The page's own two full-viewport decoration layers.
PAGE_OVERLAY_LAYERS = ("body::before", "body::after")

# Where prose is painted: the reading column. Everything inside it is a
# reading surface; everything outside it is chrome.
READING_COLUMN = ".wrap"

# Containers that hold body copy. A rule addressing one of these, or setting a
# prose token as its colour, is prose unless it is chrome below.
PROSE_CONTAINER = re.compile(
    r"(^|[\s,>+~])"
    r"(\.panel|\.story|\.item|\.brief|\.reader|\.wrap|\.others|\.also-cover"
    r"|\.drop|\.empty|\.sr-only|table|thead|tbody|tr|td|th)"
    r"(?![\w-])"
)

# The chrome the stylesheet's own header reserves for glow and chroma: bars,
# badges, chips, rules, decorative pseudo-elements, gauges and controls.
CHROME = re.compile(
    r"\.topbar|\.brand|nav\.tabs|\.topstats|\.badge|\.pill|\.chip|\.term"
    r"|\.btn|button|\.toast|\.ingest-|\.verbose-|\.switch|\.meters|\.bars"
    r"|\.bar(?![\w-])|\.save(?![\w-])|\.imp(?![\w-])|\.st(?![\w-])"
    r"|\bhr\b|summary|::before|::after|::selection|::-webkit|:-webkit"
)

# Selectors that must be pinned to a flat colour when the reader has already
# told the browser they cannot take another composited layer.
# Ids deliberately drop the feature name: `-k contrast` must select the ratio
# test alone, not this one by way of `prefers-contrast`.
A11Y_CONDITIONS = (
    pytest.param("prefers-contrast: more", id="more"),
    pytest.param("prefers-reduced-transparency: reduce", id="reduced-transparency"),
    pytest.param("forced-colors: active", id="forced-colors"),
)
A11Y_OPAQUE_SURFACES = (
    ".panel",
    ".story",
    ".item:hover",
    "tr:hover td",
    "th",
    "input[type=search]",
)


# ── stylesheet reading ──────────────────────────────────────────────────


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _split_top(value: str, sep: str = ",") -> list[str]:
    """Split on `sep` at parenthesis depth 0, so gradients stay whole."""
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts]


def _rules(css: str, media: str = "") -> list[tuple[str, str, dict[str, str]]]:
    """Every style rule as (media prelude, selector, declarations)."""
    out: list[tuple[str, str, dict[str, str]]] = []
    i, n = 0, len(css)
    while i < n:
        open_at = css.find("{", i)
        if open_at < 0:
            break
        prelude = css[i:open_at].strip()
        depth, k = 1, open_at + 1
        while k < n and depth:
            if css[k] == "{":
                depth += 1
            elif css[k] == "}":
                depth -= 1
            k += 1
        block = css[open_at + 1 : k - 1]
        if prelude.startswith("@media"):
            out.extend(_rules(block, prelude[len("@media") :].strip()))
        elif not prelude.startswith("@"):
            out.append((media, prelude, _decls(block)))
        i = k
    return out


def _decls(block: str) -> dict[str, str]:
    decls: dict[str, str] = {}
    for part in _split_top(block, ";"):
        prop, sep, value = part.partition(":")
        if sep and "{" not in prop:
            decls[prop.strip()] = value.strip()
    return decls


def _selector_matches(selector: str, wanted: str) -> bool:
    return any(part.strip() == wanted for part in _split_top(selector))


def _rule_for(rules, selector: str, media: str = "") -> dict[str, str]:
    """Declarations for `selector`, later rules winning, within one media."""
    merged: dict[str, str] = {}
    for rule_media, sel, decls in rules:
        if rule_media == media and _selector_matches(sel, selector):
            merged.update(decls)
    return merged


# ── colour ──────────────────────────────────────────────────────────────


def _hex_rgb(value: str) -> tuple[float, float, float]:
    digits = value.lstrip("#")
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _channel(c: float) -> float:
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(rgb) -> float:
    r, g, b = rgb
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def _contrast(fg, bg) -> float:
    a, b = _luminance(fg), _luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _over_black(rgb, alpha: float):
    """Black composited at `alpha` — identical for `normal` and `multiply`."""
    return tuple(c * (1.0 - alpha) for c in rgb)


_RGBA = re.compile(r"\brgba?\(([^)]*)\)", re.I)
_HSLA = re.compile(r"\bhsla?\(([^)]*)\)", re.I)


def _translucent_alphas(text: str) -> list[float]:
    found = []
    for match in list(_RGBA.finditer(text)) + list(_HSLA.finditer(text)):
        parts = [p.strip() for p in match.group(1).replace("/", ",").split(",")]
        parts = [p for p in parts if p]
        if len(parts) == 4:
            try:
                alpha = float(parts[3].rstrip("%")) / (100 if "%" in parts[3] else 1)
            except ValueError:
                continue
            if alpha < 1:
                found.append(alpha)
    return found


def _is_flat_colour(layer: str, tokens: dict[str, str]) -> bool:
    layer = layer.strip()
    var_match = re.fullmatch(r"var\((--[\w-]+)\)", layer)
    if var_match:
        return _is_flat_colour(tokens.get(var_match.group(1), ""), tokens)
    if re.fullmatch(r"#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}", layer):
        return True
    if re.fullmatch(r"rgb\([^)]*\)|hsl\([^)]*\)", layer, re.I):
        return True
    if "(" in layer:
        return False  # a gradient or url() backs nothing; it is not a flat colour
    return bool(layer) and layer not in (
        "transparent",
        "none",
        "inherit",
        "initial",
        "currentColor",
    )


def _background_translucency(decls: dict[str, str], tokens: dict[str, str]) -> list[str]:
    """Translucent colours a background stack does not seal behind an opaque layer."""
    offenders = []
    for prop in ("background", "background-color", "background-image"):
        value = decls.get(prop)
        if not value:
            continue
        layers = _split_top(value)
        if prop == "background" and _is_flat_colour(layers[-1], tokens):
            continue  # an opaque final layer backs the whole stack
        for layer in layers:
            if _translucent_alphas(layer):
                offenders.append(f"{prop}: {layer}")
    return offenders


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def css() -> str:
    return _strip_comments(CSS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules(css):
    return _rules(css)


@pytest.fixture(scope="module")
def tokens(rules) -> dict[str, str]:
    return {k: v for k, v in _rule_for(rules, ":root").items() if k.startswith("--")}


@pytest.fixture(scope="module")
def page_overlay_factor(rules) -> float:
    """How much of its own darkening the page still lands on prose.

    Each full-viewport CRT layer only reaches a reading surface if the reading
    column does not paint above it, so the factor is read off the stylesheet's
    own stacking order rather than assumed.
    """
    column = _rule_for(rules, READING_COLUMN)
    column_z = column.get("z-index")
    column_z = float(column_z) if column_z and column.get("position") not in (None, "static") else None
    factor = 1.0
    for layer in PAGE_OVERLAY_LAYERS:
        decls = _rule_for(rules, layer)
        assert decls, f"{layer} is missing from app.css"
        layer_z = float(decls.get("z-index", 0))
        if column_z is not None and column_z > layer_z:
            continue  # the reading column paints above this layer
        alphas = _translucent_alphas(decls.get("background", ""))
        factor *= 1.0 - (max(alphas) if alphas else 0.0)
    return factor


# ── 1. contrast under the whole composited stack ────────────────────────


def _surface_tokens(tokens) -> list[str]:
    return [t for t in tokens if t == "--bg" or t.startswith(("--bg-", "--panel"))]


@pytest.mark.parametrize("prose_token", PROSE_TOKENS)
def test_prose_pairs_hold_contrast_under_page_and_extension_overlay(
    prose_token, tokens, page_overlay_factor
):
    """Every prose-on-surface pair survives the page's CRT plus a hostile overlay."""
    failures = []
    for surface_token in _surface_tokens(tokens):
        fg = _hex_rgb(tokens[prose_token])
        bg = _hex_rgb(tokens[surface_token])
        page_fg = _over_black(fg, 1 - page_overlay_factor)
        page_bg = _over_black(bg, 1 - page_overlay_factor)
        full_fg = _over_black(page_fg, EXTENSION_BLACK_ALPHA)
        full_bg = _over_black(page_bg, EXTENSION_BLACK_ALPHA)
        ratio = _contrast(full_fg, full_bg)
        floor = 7.0 if (prose_token, surface_token) == ("--text", "--bg") else 4.5
        if ratio < floor:
            failures.append(
                f"{prose_token} on {surface_token}: "
                f"{_contrast(fg, bg):.2f} bare, "
                f"{_contrast(page_fg, page_bg):.2f} under the page's own overlay, "
                f"{ratio:.2f} under page+extension (needs {floor})"
            )
    assert not failures, "\n".join(failures)


# ── 2. the accessibility signals an overlay user already carries ─────────


@pytest.mark.parametrize("condition", A11Y_CONDITIONS)
def test_media_queries_neutralize_the_pages_own_overlay(condition, css, tokens):
    """Each signal turns the page's CRT off and flattens reading surfaces."""
    scoped = [
        (media, sel, decls)
        for media, sel, decls in _rules(css)
        if condition.replace(" ", "") in media.replace(" ", "")
    ]
    assert scoped, f"app.css answers no @media ({condition})"

    for layer in PAGE_OVERLAY_LAYERS:
        hidden = any(
            _selector_matches(sel, layer) and decls.get("display") == "none"
            for _, sel, decls in scoped
        )
        assert hidden, f"@media ({condition}) leaves {layer} painting"

    for surface in A11Y_OPAQUE_SURFACES:
        matching = [decls for _, sel, decls in scoped if _selector_matches(sel, surface)]
        assert matching, f"@media ({condition}) never flattens {surface}"
        for decls in matching:
            assert not _background_translucency(decls, tokens), (
                f"@media ({condition}) leaves {surface} translucent: {decls}"
            )

    for _, sel, decls in scoped:
        assert not _background_translucency(decls, tokens), (
            f"@media ({condition}) still declares a translucent background on {sel}"
        )


# ── 3. prose contrast comes from opaque colour, never from compositing ──


def _prose_rules(rules):
    for media, sel, decls in rules:
        if CHROME.search(sel):
            continue
        colour = decls.get("color", "").replace(" ", "")
        is_prose = PROSE_CONTAINER.search(sel) or any(
            f"var({t})" == colour for t in PROSE_TOKENS
        )
        if is_prose:
            yield media, sel, decls


def test_prose_is_opaque(rules, tokens):
    """No prose rule borrows its contrast from opacity, blending or translucency."""
    offenders = []
    for media, sel, decls in _prose_rules(rules):
        where = f"@media {media} " if media else ""
        if "opacity" in decls:
            try:
                if float(decls["opacity"]) < 1:
                    offenders.append(f"{where}{sel} -> opacity: {decls['opacity']}")
            except ValueError:
                pass
        blend = decls.get("mix-blend-mode", "normal")
        if blend != "normal":
            offenders.append(f"{where}{sel} -> mix-blend-mode: {blend}")
        for hit in _background_translucency(decls, tokens):
            offenders.append(f"{where}{sel} -> {hit}")
    assert not offenders, "\n".join(offenders)


def test_prose_rule_scan_is_not_vacuous(rules):
    """Guard the classifier: the prose set must still be most of the stylesheet."""
    assert len(list(_prose_rules(rules))) > 30
