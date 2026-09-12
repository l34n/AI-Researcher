"""app.css under a third-party browser overlay extension.

A CRT filter, a VHS tint or a dark-mode inverter injects a `position: fixed`
full-viewport layer at the top of the stacking order, usually painting through
`mix-blend-mode`. The page cannot out-stack it and cannot remove it, so the
only budget it controls is its own: how much darkening app.css has already
spent before the extension's lands, and whether prose contrast is carried by
opaque colour or borrowed from whatever happens to be behind.

app.css spends none of that budget: it paints no layer of its own over its own
content, which is what `test_no_crt_layers_remain` holds. The compositing model
stays anyway, because a layer comes back quietly — so these tests
read app.css as a stylesheet rather than as text, resolve the `:root` palette,
composite whatever the page does stack over a reading surface the way the
cascade would, then composite a worst-case extension overlay on top and check
the surviving WCAG ratio. A layer put back, a prose token dimmed, or a ground
left taking its colour from whatever is behind it fails here rather than in
someone's browser.

Which grounds get read is derived from the rules, not listed: every rule that
paints a background on a box the stylesheet shows can hold a line of text, the
chrome outside the reading column included. Each one has to state its own
colour -- the stack has to seal -- because "whatever is behind it" is exactly
what an overlay extension takes away, and a ground that borrows from the
backdrop is a ground no check can put a number on. What sits above that seal is
composited rather than skipped, so a wash cannot hide behind the opaque layer
under it.
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

# The page's own two full-viewport decoration layers. The stylesheet no longer
# paints either; the names stay because the compositing model is stated in terms
# of them, and a layer put back under one of these names has to be seen.
PAGE_OVERLAY_LAYERS = ("body::before", "body::after")

# Every decoration layer app.css used to paint over its own content: the two
# body layers and the pair the bars carried. A rule addressing any of them is a
# layer coming back, and so is a media block switching off one that is gone.
REMOVED_LAYERS = PAGE_OVERLAY_LAYERS + (".topbar::after", ".ingest-status::after")

# The chrome bars: sticky strips that paint a ground of their own and carry
# their own text, outside the reading column.
CHROME_BARS = (".topbar", ".ingest-status")

# Text each bar carries at a prose token. Which bare class sits inside which bar
# is the template's fact, not the stylesheet's, so these are named and asserted
# present; every other prose-coloured rule on a bar is derived from its selector.
CHROME_TEXT_ANCHORS = {
    ".topbar": (".topstats",),
    ".ingest-status": (".ingest-current",),
}

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

# Rules the prose classifier must keep calling prose. Not a frozen count: a
# count is a reading of the stylesheet at one moment, and a classifier quietly
# narrowed until it passes is exactly what this has to catch. These are the
# containers that carry the page's body copy.
PROSE_ANCHORS = (
    ".panel",
    ".story",
    ".item:hover",
    "tr:hover td",
    "th",
    ".brief li",
    ".story .sum",
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


# At-rules whose blocks hold ordinary style rules the browser may apply. Their
# contents are read like any other rule; only the condition they arrive under
# differs. `@keyframes` and `@font-face` are not here: their blocks are frames
# and descriptors, not rules.
_GROUP_AT_RULE = re.compile(r"@(media|supports|container|layer|scope)\b", re.I)


def _rules(css: str, condition: str = "") -> list[tuple[str, str, dict[str, str]]]:
    """Every style rule as (condition prelude, selector, declarations).

    A rule inside `@supports`, `@layer` or `@container` is a rule this page can
    paint, so it is read like a rule inside `@media`. Reading only `@media` left
    every other group block invisible to every check in this module — a deleted
    layer written back inside one would have been seen by nothing here.
    """
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
        group = _GROUP_AT_RULE.match(prelude)
        if group:
            # `@media` keeps its bare condition text, which is what the
            # accessibility test matches on; any other group keeps its keyword,
            # so a failure names the block the rule actually sits in.
            inner = prelude[len(group.group(0)) :].strip() if group.group(1).lower() == "media" else prelude
            out.extend(_rules(block, f"{condition} and {inner}" if condition else inner))
        elif not prelude.startswith("@"):
            out.append((condition, prelude, _decls(block)))
        i = k
    return out


def _where(condition: str) -> str:
    """A rule's condition, labelled the way the stylesheet writes it."""
    if not condition:
        return ""
    return condition if condition.lstrip().startswith("@") else f"@media {condition}"


def _decls(block: str) -> dict[str, str]:
    decls: dict[str, str] = {}
    for part in _split_top(block, ";"):
        prop, sep, value = part.partition(":")
        if sep and "{" not in prop:
            decls[prop.strip()] = value.strip()
    return decls


def _selector_parts(selector: str) -> set[str]:
    """A rule's comma-separated selectors, whitespace-normalised."""
    return {" ".join(part.split()) for part in _split_top(selector) if part.strip()}


def _selector_matches(selector: str, wanted: str) -> bool:
    return " ".join(wanted.split()) in _selector_parts(selector)


def _overrides(block_selector: str, base_selector: str) -> bool:
    """Whether a rule in a media block restyles a rule in the base cascade.

    Compared part by part: `.panel, .story` inside the block overrides the base
    `.panel`, and a rule written as one list overrides each rule in it.
    """
    return bool(_selector_parts(block_selector) & _selector_parts(base_selector))


def _rules_targeting(rules, selectors) -> list[str]:
    """Every rule, under any condition, whose selector addresses one of `selectors`."""
    return [
        f"{_where(condition)} {sel}" if condition else sel
        for condition, sel, _decls in rules
        if any(_selector_matches(sel, wanted) for wanted in selectors)
    ]


def _rule_for(rules, selector: str, media: str = "") -> dict[str, str]:
    """Declarations for `selector`, later rules winning, within one media."""
    merged: dict[str, str] = {}
    for rule_media, sel, decls in rules:
        if rule_media == media and _selector_matches(sel, selector):
            merged.update(decls)
    return merged


def _decls_anywhere(rules, selector: str) -> dict[str, str]:
    """Declarations for `selector` under any condition, later rules winning.

    For the layers this page must not paint, a condition is not an excuse: a
    rule inside a group block is one the browser can apply, so the worst of
    what the stylesheet says about a layer is what the model has to assume.
    """
    merged: dict[str, str] = {}
    for _condition, sel, decls in rules:
        if _selector_matches(sel, selector):
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
        # `rgb(0 0 0 / 30%)` and `rgba(0,0,0,.3)` are the same colour: split on
        # both separators so the modern form is not silently read as opaque.
        parts = [p for p in re.split(r"[,\s]+", match.group(1).replace("/", ",")) if p]
        if len(parts) == 4:
            try:
                alpha = float(parts[3].rstrip("%")) / (100 if "%" in parts[3] else 1)
            except ValueError:
                continue
            if alpha < 1:
                found.append(alpha)
    return found


def _rgba_terms(value: str) -> list[tuple[tuple[float, float, float], float]]:
    """Every `rgb()`/`rgba()` colour in a value, as (rgb, alpha)."""
    out: list[tuple[tuple[float, float, float], float]] = []
    for match in _RGBA.finditer(value):
        parts = [p for p in re.split(r"[,\s]+", match.group(1).replace("/", ",")) if p]
        if len(parts) < 3:
            continue
        try:
            rgb = tuple(float(p.rstrip("%")) for p in parts[:3])
            alpha = float(parts[3].rstrip("%")) / (100 if "%" in parts[3] else 1) if len(parts) == 4 else 1.0
        except ValueError:
            continue
        out.append((rgb, alpha))  # type: ignore[arg-type]
    return out


def _is_flat_colour(layer: str, tokens: dict[str, str]) -> bool:
    layer = layer.strip()
    var_match = re.fullmatch(r"var\((--[\w-]+)\)", layer)
    if var_match:
        return _is_flat_colour(tokens.get(var_match.group(1), ""), tokens)
    if re.fullmatch(r"#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}", layer):
        return True
    if re.fullmatch(r"(rgba?|hsla?)\([^)]*\)", layer, re.I):
        # Opaque only if it carries no alpha below 1 — in either syntax. Reading
        # `rgb(0 0 0 / 30%)` as flat would let it seal a stack it cannot back.
        return not _translucent_alphas(layer)
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


def _opaque_ground(decls: dict[str, str], tokens: dict[str, str]) -> str | None:
    """The flat opaque colour a background stack ends in, if it declares one.

    The absence of a translucent background is not the presence of an opaque one:
    a rule that declares no background at all, or declares `transparent`, clears
    every translucency check and still leaves its text standing on whatever
    happens to be behind it — which under an overlay extension is not this page.
    """
    for prop in ("background", "background-color"):
        value = decls.get(prop)
        if value and _is_flat_colour(_split_top(value)[-1], tokens):
            return _split_top(value)[-1]
    return None


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


# A length that is zero, in any of the spellings the cascade accepts.
_ZERO_LENGTH = re.compile(r"[+-]?0(?:\.0+)?(?:px|pt|em|rem|vh|vw|%)?\Z", re.I)


def _covers_its_box(decls: dict[str, str]) -> bool:
    """Whether a rule pins all four edges to zero: the shape of a layer laid
    across everything behind it.

    Read in every spelling that means the same thing — `inset: 0`, `inset: 0px`,
    `inset: 0 0 0 0`, and the four longhand offsets — because the string `inset:
    0` is a house style, not a property of the layer. Matching only that one
    spelling let the deleted layers come back written the ordinary way and be
    counted as nothing at all. What is still outside the shape: a layer sized
    with viewport units instead of pinned edges, and anything a script paints.
    """
    inset = decls.get("inset")
    if inset is not None:
        parts = inset.split()
        return bool(parts) and len(parts) <= 4 and all(_ZERO_LENGTH.fullmatch(p) for p in parts)
    edges = [decls.get(edge) for edge in ("top", "right", "bottom", "left")]
    return all(edge is not None and _ZERO_LENGTH.fullmatch(edge.strip()) for edge in edges)


def _full_viewport_layers(rules) -> set[str]:
    """Selectors that paint a fixed layer across the whole viewport.

    Under any condition, not only in the base cascade: a layer inside a group
    block is one the page can still paint.
    """
    return {
        sel
        for _condition, sel, decls in rules
        if decls.get("position") == "fixed"
        and _covers_its_box(decls)
        and any(decls.get(p) for p in ("background", "background-color", "background-image"))
    }


@pytest.fixture(scope="module")
def page_overlay_factor(rules) -> float:
    """How much of its own darkening the page still lands on prose.

    Each full-viewport CRT layer only reaches a reading surface if the reading
    column does not paint above it, so the factor is read off the stylesheet's
    own stacking order rather than assumed. A layer app.css no longer paints
    composites nothing: the page that drops its overlays satisfies the criterion
    more completely than the page that keeps them, so its absence is a factor of
    1, never a failure. What must not happen is a full-viewport layer this model
    has never heard of, and that is what is asserted instead.
    """
    unmodelled = _full_viewport_layers(rules) - set(PAGE_OVERLAY_LAYERS)
    assert not unmodelled, f"app.css paints full-viewport layers this model does not know: {sorted(unmodelled)}"

    column = _rule_for(rules, READING_COLUMN)
    column_z = column.get("z-index")
    column_z = float(column_z) if column_z and column.get("position") not in (None, "static") else None
    factor = 1.0
    for layer in PAGE_OVERLAY_LAYERS:
        decls = _decls_anywhere(rules, layer)
        if not decls:
            continue  # app.css does not paint this layer at all
        layer_z = float(decls.get("z-index", 0))
        if column_z is not None and column_z > layer_z:
            continue  # the reading column paints above this layer
        alphas = _translucent_alphas(decls.get("background", ""))
        factor *= 1.0 - (max(alphas) if alphas else 0.0)
    return factor


# ── 1. the page paints no overlay layer of its own ──────────────────────


def test_no_crt_layers_remain(css, rules):
    """app.css composites nothing over its own content.

    Darkening is multiplicative and contrast is not, so every layer the page
    paints over its own text is spent out of the budget the one layer it cannot
    see — the extension's — is about to take. The page that paints none of its
    own hands that budget over whole, and this is the check that keeps it so:
    not the reordering, which only moved the spending around, but the absence.
    """
    revived = _rules_targeting(rules, REMOVED_LAYERS)
    assert not revived, f"app.css still addresses a removed CRT layer: {revived}"

    blended = [
        f"{_where(condition)} {sel}" if condition else sel
        for condition, sel, decls in rules
        if decls.get("mix-blend-mode", "normal") != "normal"
    ]
    assert not blended, f"app.css still blends its own layers: {blended}"

    assert not _full_viewport_layers(rules), (
        f"app.css still paints a full-viewport layer: {sorted(_full_viewport_layers(rules))}"
    )

    scanlines = [
        f"{sel} -> {prop}"
        for _media, sel, decls in rules
        for prop in ("background", "background-color", "background-image")
        if "repeating-linear-gradient" in decls.get(prop, "")
    ]
    assert not scanlines, f"a scanline gradient is still painted: {scanlines}"


# ── 2. contrast under the whole composited stack ────────────────────────


_HEX_ONLY = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\Z")
_COLOUR_TERM = re.compile(r"var\((--[\w-]+)\)|#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b")


def _opaque_stops(value: str, tokens: dict[str, str]) -> list[tuple[str, str]]:
    """Every opaque colour a background value can paint, as (label, hex).

    A gradient is read by every stop it names, not by one of them: all of them
    are painted somewhere in the box and a stylesheet cannot know which one a
    glyph lands on, so the lightest counts like the rest. Translucent stops are
    not grounds — whatever is behind them is, and `test_prose_is_opaque` is what
    keeps prose off them in the first place.
    """
    found: list[tuple[str, str]] = []
    for match in _COLOUR_TERM.finditer(value):
        name = match.group(1)
        literal = (tokens.get(name, "") if name else match.group(0)).strip()
        if _HEX_ONLY.match(literal):
            found.append((name or literal, literal))
    return found


def _prose_grounds(rules, tokens) -> dict[str, str]:
    """Every ground app.css paints prose on, as label -> hex.

    Derived, never listed. Naming the surfaces by token prefix covered only the
    tokens that happened to be called `--bg*` or `--panel*`, so a ground written
    as a literal, or as a token named anything else, was outside the check
    entirely — and the lightest ground on the page was one of those. Reading the
    grounds off the prose rules instead means a surface is covered the day it is
    declared rather than the day someone remembers to widen a tuple. The `:root`
    surface tokens are unioned in so a token used only inside the accessibility
    block is still checked.
    """
    grounds: dict[str, str] = {
        token: value.strip()
        for token, value in tokens.items()
        if (token == "--bg" or token.startswith(("--bg-", "--panel")))
        and _HEX_ONLY.match(value.strip())
    }
    seen = set(grounds.values())
    for _media, sel, decls in _prose_rules(rules):
        for prop in ("background", "background-color", "background-image"):
            for name, literal in _opaque_stops(decls.get(prop, ""), tokens):
                if literal in seen:
                    continue
                grounds[name if name.startswith("--") else f"{literal} on {sel}"] = literal
                seen.add(literal)
    return grounds


@pytest.fixture(scope="module")
def prose_grounds(rules, tokens) -> dict[str, str]:
    return _prose_grounds(rules, tokens)


@pytest.mark.parametrize("prose_token", PROSE_TOKENS)
def test_prose_pairs_hold_contrast_under_page_and_extension_overlay(
    prose_token, rules, tokens, prose_grounds, page_overlay_factor
):
    """Every prose-on-ground pair survives the page's CRT plus a hostile overlay."""
    # The derivation must not be able to go quiet: every prose rule that declares
    # a background has to contribute a ground, or the colour reader has stopped
    # understanding a syntax the stylesheet is using.
    for _media, sel, decls in _prose_rules(rules):
        declared = [decls[p] for p in ("background", "background-color", "background-image") if decls.get(p)]
        if declared and not any(_opaque_stops(v, tokens) for v in declared):
            assert all(_translucent_alphas(v) for v in declared), (
                f"no ground could be read out of {sel}: {declared}"
            )

    failures = []
    for label, ground in prose_grounds.items():
        fg = _hex_rgb(tokens[prose_token])
        bg = _hex_rgb(ground)
        page_fg = _over_black(fg, 1 - page_overlay_factor)
        page_bg = _over_black(bg, 1 - page_overlay_factor)
        full_fg = _over_black(page_fg, EXTENSION_BLACK_ALPHA)
        full_bg = _over_black(page_bg, EXTENSION_BLACK_ALPHA)
        ratio = _contrast(full_fg, full_bg)
        floor = 7.0 if (prose_token, label) == ("--text", "--bg") else 4.5
        if ratio < floor:
            failures.append(
                f"{prose_token} on {label} ({ground}): "
                f"{_contrast(fg, bg):.2f} bare, "
                f"{_contrast(page_fg, page_bg):.2f} under the page's own overlay, "
                f"{ratio:.2f} under page+extension (needs {floor})"
            )
    assert not failures, "\n".join(failures)


# ── 3. the text the chrome paints on its own ground ─────────────────────


def _backdrop(rules, tokens) -> tuple[float, float, float]:
    """What a translucent chrome ground is painted over: the body's final layer."""
    layers = _split_top(_rule_for(rules, "body").get("background", ""))
    assert layers and _is_flat_colour(layers[-1], tokens), (
        f"the body background no longer ends in a flat colour: {layers[-1:]}"
    )
    return _hex_rgb(_resolve(layers[-1], tokens))


def _resolve(layer: str, tokens: dict[str, str]) -> str:
    var_match = re.fullmatch(r"var\((--[\w-]+)\)", layer.strip())
    return tokens[var_match.group(1)].strip() if var_match else layer.strip()


def _bar_grounds(rules, bar: str, tokens, backdrop) -> list[tuple[str, tuple]]:
    """Every colour a bar's own background can put under its text.

    A gradient is read by every stop it names; a translucent stop is composited
    over the page's backdrop, because that is what is actually behind the bar.
    """
    value = _rule_for(rules, bar).get("background", "") or _rule_for(rules, bar).get("background-color", "")
    grounds = [(literal, _hex_rgb(literal)) for _name, literal in _opaque_stops(value, tokens)]
    for rgb, alpha in _rgba_terms(value):
        composited = tuple(rgb[i] * alpha + backdrop[i] * (1 - alpha) for i in range(3))
        grounds.append((f"rgba({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f},{alpha}) over backdrop", composited))
    return grounds


def _bar_overlay_factor(rules, bar: str) -> float:
    """How much the page's own layers darken this bar before anything else does.

    Two sources, both read off the stylesheet: a pseudo-element the bar paints
    across itself, and any full-viewport layer that stacks at or above the bar.
    """
    factor = 1.0
    bar_z = float(_rule_for(rules, bar).get("z-index", 0))
    for _condition, sel, decls in rules:
        if decls.get("position") not in ("absolute", "fixed") or not _covers_its_box(decls):
            continue
        if not any(part.startswith(f"{bar}::") for part in _selector_parts(sel)):
            continue
        alphas = _translucent_alphas(decls.get("background", ""))
        factor *= 1.0 - (max(alphas) if alphas else 0.0)
    for sel in _full_viewport_layers(rules):
        decls = _decls_anywhere(rules, sel)
        if float(decls.get("z-index", 0)) >= bar_z:
            alphas = _translucent_alphas(decls.get("background", ""))
            factor *= 1.0 - (max(alphas) if alphas else 0.0)
    return factor


def _bar_text(rules, bar: str) -> dict[str, str]:
    """Prose-coloured text a bar carries, as selector -> token."""
    found: dict[str, str] = {}
    for media, sel, decls in rules:
        if media:
            continue
        colour = decls.get("color", "").replace(" ", "")
        token = next((t for t in PROSE_TOKENS if f"var({t})" == colour), None)
        if token is None:
            continue
        for part in _selector_parts(sel):
            if part == bar or part.startswith((f"{bar} ", f"{bar}.")):
                found[part] = token
    for anchor in CHROME_TEXT_ANCHORS[bar]:
        colour = _rule_for(rules, anchor).get("color", "").replace(" ", "")
        token = next((t for t in PROSE_TOKENS if f"var({t})" == colour), None)
        assert token, f"{anchor} no longer carries a prose token colour: {colour!r}"
        found[anchor] = token
    return found


@pytest.mark.parametrize("bar", CHROME_BARS)
def test_chrome_text_survives_the_extension_overlay(bar, rules, tokens):
    """The bars' own text clears AA under the extension, with nothing of ours on top.

    The reading column is not the only place the page paints words. Each bar
    carries its own scale text, and anything this page composites over that text
    is spent out of the same budget the extension is about to claim.
    """
    backdrop = _backdrop(rules, tokens)
    grounds = _bar_grounds(rules, bar, tokens, backdrop)
    assert grounds, f"{bar} paints no ground this model can read"
    factor = _bar_overlay_factor(rules, bar)

    failures = []
    for sel, token in _bar_text(rules, bar).items():
        fg = _hex_rgb(tokens[token])
        for label, bg in grounds:
            page_fg = _over_black(fg, 1 - factor)
            page_bg = _over_black(bg, 1 - factor)
            ratio = _contrast(
                _over_black(page_fg, EXTENSION_BLACK_ALPHA),
                _over_black(page_bg, EXTENSION_BLACK_ALPHA),
            )
            if ratio < 4.5:
                failures.append(
                    f"{sel} ({token}) on {bar} {label}: "
                    f"{_contrast(fg, bg):.2f} bare, "
                    f"{_contrast(page_fg, page_bg):.2f} under the page's own layers, "
                    f"{ratio:.2f} under page+extension (needs 4.5)"
                )
    assert not failures, "\n".join(failures)


# ── 4. the accessibility signals an overlay user already carries ─────────


@pytest.mark.parametrize("condition", A11Y_CONDITIONS)
def test_media_queries_neutralize_the_pages_own_overlay(condition, css, tokens):
    """Each signal turns the page's CRT off and flattens reading surfaces."""
    base_rules = _rules(css)
    scoped = [
        (media, sel, decls)
        for media, sel, decls in base_rules
        if condition.replace(" ", "") in media.replace(" ", "")
    ]
    assert scoped, f"app.css answers no @media ({condition})"

    # Nothing here switches a CRT layer off any more, because there is none to
    # switch off. A rule addressing one is either a layer that came back or a
    # dead rule promising a flattening it no longer performs.
    revived = _rules_targeting(scoped, REMOVED_LAYERS)
    assert not revived, (
        f"@media ({condition}) addresses a layer app.css no longer paints: {revived}"
    )

    # Which surfaces have to end up flat is read off the stylesheet, not listed
    # here: every ground a prose rule declares, whether or not anyone thought to
    # name it. A tuple of surface names can be trimmed until it passes, and a
    # presence test on it cannot tell a rule that flattens something from a rule
    # that restates what was already flat.
    prose_with_ground = [
        (sel, decls)
        for media, sel, decls in _prose_rules(base_rules)
        if not media
        and any(decls.get(p) for p in ("background", "background-color", "background-image"))
    ]
    assert prose_with_ground, "no prose rule declares a background; this scan is vacuous"
    for sel, decls in prose_with_ground:
        override: dict[str, str] = {}
        for _m, block_sel, block_decls in scoped:
            if _overrides(block_sel, sel):
                override.update(block_decls)
        # The cascade, not the override by itself. A block rule that restyles a
        # prose surface without declaring a background leaves the base one
        # painting, and reading the override alone let that base ground pass as
        # though the block had replaced it.
        effective = {**decls, **override}
        assert not _background_translucency(effective, tokens), (
            f"@media ({condition}) leaves {sel} translucent: {effective}"
        )
        painted = " ".join(
            effective.get(p, "") for p in ("background", "background-color", "background-image")
        )
        assert "gradient(" not in painted, (
            f"@media ({condition}) leaves {sel} on a gradient: {painted.strip()}"
        )
        assert _opaque_ground(effective, tokens), (
            f"@media ({condition}) gives {sel} no opaque ground of its own: {painted.strip()!r}"
        )

    for _, sel, decls in scoped:
        assert not _background_translucency(decls, tokens), (
            f"@media ({condition}) still declares a translucent background on {sel}"
        )


# ── 5. prose contrast comes from opaque colour, never from compositing ──


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
        where = f"{_where(media)} " if media else ""
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
    """Guard the classifier: it must still see the containers that carry body copy.

    A bare floor was the wrong guard. It stood at 30 against a live reading of
    84, so the classifier could lose every panel, story card, feed row and table
    cell — the containers holding nearly all of the page's prose — and still
    clear it by a wide margin. Name what has to stay classified instead.
    """
    prose = {sel for _media, sel, _decls in _prose_rules(rules)}
    seen = set().union(*(_selector_parts(sel) for sel in prose)) if prose else set()
    missing = [anchor for anchor in PROSE_ANCHORS if anchor not in seen]
    assert not missing, f"the prose classifier no longer sees: {missing}"
    assert len(prose) > 30


# ── 6. the ground model: every layer, not only down to the first opaque one ──
#
# A background stack paints bottom layer first. The reader above stops at the
# first opaque layer it finds and calls that the ground, which is right about
# what seals the stack and wrong about what the browser paints: a translucent
# wash sitting *above* that opaque layer still lands on the glyph. `.panel`
# under a 10% white wash measures a real 4.25 and the first-opaque read scores
# it 4.93 — a number no browser will produce. The model below composites the
# whole stack instead, and the opaque layer keeps its one real job: sealing the
# stack so the colour does not depend on what is behind the element.

# Colour keywords a background layer can name that contribute nothing.
_TRANSPARENT = ("transparent", "none")

# Functions that paint an image rather than name a colour. A layer built from
# one of these is read by the colour terms inside it.
_IMAGE_FN = re.compile(r"\b(?:repeating-)?(?:linear|radial|conic)-gradient\b|\burl\b|\bimage-set\b", re.I)


def _top_level_terms(value: str) -> list[str]:
    """A value's space-separated terms, keeping each function call whole."""
    terms, depth, cur = [], 0, ""
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch.isspace() and depth == 0:
            if cur.strip():
                terms.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        terms.append(cur.strip())
    return terms


def _paints(term: str, tokens: dict[str, str]) -> list[tuple[str, tuple[float, float, float], float]]:
    """Every colour a term can put on a pixel, as (label, rgb, alpha).

    A flat colour contributes one. A gradient contributes one per stop, because
    all of its stops are painted somewhere in the box and a stylesheet cannot
    know which one a glyph lands on — the same reading `_opaque_stops` already
    uses. `transparent` is a stop like any other: it is how a radial wash says
    "nothing here", and dropping it would leave the model unable to see that a
    corner wash does not reach the middle of the page.
    """
    term = term.strip()
    if term.lower() in _TRANSPARENT:
        return [("transparent", (0.0, 0.0, 0.0), 0.0)]
    resolved = _resolve(term, tokens)
    if resolved != term:
        return _paints(resolved, tokens)
    if _HEX_ONLY.match(term):
        return [(term, _hex_rgb(term), 1.0)]
    if re.fullmatch(r"(?:rgba?|hsla?)\([^)]*\)", term, re.I):
        return [(term, rgb, alpha) for rgb, alpha in _rgba_terms(term)] or []
    if _IMAGE_FN.search(term):
        out: list[tuple[str, tuple[float, float, float], float]] = []
        inner = term[term.index("(") + 1 : term.rindex(")")]
        for part in _split_top(inner):
            for word in _top_level_terms(part):
                out.extend(_paints(word, tokens))
        return out
    return []


def _layers(value: str, tokens: dict[str, str]) -> list[list[tuple[str, tuple, float]]]:
    """A background value as paint alternatives per layer, bottom layer last.

    The final comma-separated layer may carry the background colour alongside
    its image (`background: var(--bg) linear-gradient(...)`), and the colour is
    painted under every image in the stack — so it is lifted out as its own
    bottom layer rather than read as a sibling of the gradient above it.
    """
    out: list[list[tuple[str, tuple, float]]] = []
    parts = _split_top(value)
    for index, part in enumerate(parts):
        terms = _top_level_terms(part)
        colours, images = [], []
        for term in terms:
            if _IMAGE_FN.search(term):
                images.append(term)
            elif _paints(term, tokens):
                colours.append(term)
        if index == len(parts) - 1 and images and colours:
            out.append([p for term in images for p in _paints(term, tokens)])
            out.append([p for term in colours for p in _paints(term, tokens)])
            continue
        alternatives = [p for term in terms for p in _paints(term, tokens)]
        if alternatives:
            out.append(alternatives)
    return out


def _seal_index(layers) -> int | None:
    """The lowest layer that is opaque on every alternative, if the stack has one.

    Below that layer nothing shows through, so the stack's colour does not
    depend on what is behind the element. A stack with no such layer borrows
    its ground from the backdrop, which under an overlay extension is not a
    colour this page controls or this model can name.
    """
    for index in range(len(layers) - 1, -1, -1):
        if layers[index] and all(alpha >= 1.0 for _label, _rgb, alpha in layers[index]):
            return index
    return None


def _grounds(value: str, tokens: dict[str, str]) -> list[tuple[str, tuple[float, float, float]]] | None:
    """Every colour a sealed background stack can paint, or None if it is unsealed.

    Every alternative of every layer above the seal is composited down onto it,
    so a wash the first-opaque read skipped is in the answer.
    """
    layers = _layers(value, tokens)
    seal = _seal_index(layers)
    if seal is None:
        return None
    stack = layers[: seal + 1]
    grounds = [("", base_rgb) for _label, base_rgb, _a in stack[-1]]
    for layer in reversed(stack[:-1]):
        nxt = []
        for label, under in grounds:
            for name, rgb, alpha in layer:
                over = tuple(rgb[i] * alpha + under[i] * (1 - alpha) for i in range(3))
                nxt.append((f"{label}+{name}" if label else name, over))
        grounds = nxt
    seen, unique = set(), []
    for label, rgb in grounds:
        key = tuple(round(c, 3) for c in rgb)
        if key not in seen:
            seen.add(key)
            unique.append((label or "sealed", rgb))
    return unique


def test_the_ground_model_composites_a_sealed_wash(tokens):
    """A wash above the opaque layer is in the ground, not skipped with it.

    The number in this check is the one the first-opaque read got wrong, kept
    as a literal so the shortcut cannot come back quietly: `.panel` under a 10%
    white wash is 4.25 against --text-dim, and reading only as far as the
    opaque layer scores the same stack 4.93.
    """
    sealed = "#0b0b1e linear-gradient(rgba(255,255,255,.10), rgba(255,255,255,.10))"
    grounds = _grounds(sealed, tokens)
    assert grounds is not None, "an opaque bottom layer must seal the stack"
    assert len(grounds) == 1, f"a flat wash over a flat colour is one ground: {grounds}"
    _label, ground = grounds[0]
    assert [round(c) for c in ground] == [35, 35, 52], (
        f"the wash above the opaque layer was dropped: {[round(c) for c in ground]}"
    )

    dim = _hex_rgb(tokens["--text-dim"])
    composited = _contrast(_over_black(dim, EXTENSION_BLACK_ALPHA), _over_black(ground, EXTENSION_BLACK_ALPHA))
    skipped = _contrast(
        _over_black(dim, EXTENSION_BLACK_ALPHA),
        _over_black(_hex_rgb("#0b0b1e"), EXTENSION_BLACK_ALPHA),
    )
    assert round(composited, 2) == 4.25, f"--text-dim on the composited wash is 4.25, got {composited:.2f}"
    assert round(skipped, 2) == 4.93, f"the first-opaque read scores the same stack 4.93, got {skipped:.2f}"
    assert composited < skipped, "compositing the wash must cost contrast, never add it"

    # A darkening wash is read the same way. The check below once required each
    # sealed stack to read *lighter* than its seal, which is true of the five washes
    # this stylesheet happens to write and not of the model: this stack reads darker
    # and is read correctly, and that spelling failed it saying the wash was dropped.
    darker = _grounds("#0b0b1e linear-gradient(rgba(0,0,0,.20), rgba(0,0,0,.20))", tokens)
    assert darker is not None and len(darker) == 1
    assert [round(c) for c in darker[0][1]] == [9, 9, 24], (
        f"a darkening wash above the seal is composited too: {[round(c) for c in darker[0][1]]}"
    )


def test_sealed_wash_reading_is_load_bearing_in_this_stylesheet(rules, tokens):
    """The stylesheet really does seal washes, so the model is not theory.

    Without this the check above could pass against a stylesheet that never
    stacks anything, and the first-opaque shortcut would cost nothing to
    restore. Each sealed stack is also required to read as something other than
    its seal: a wash that composites to exactly its own base is a wash the
    reader lost.

    Other than, not lighter than. `lightest > base` was the same claim plus an
    assumption about sign, and the assumption is this stylesheet's five washes
    rather than the model's: a darkening wash stacked over an opaque seal is an
    ordinary thing to write, reads correctly here, and failed that spelling with
    a message saying the wash had been dropped.
    """
    sealed = []
    for condition, sel, decls in rules:
        for prop in ("background", "background-color", "background-image"):
            value = decls.get(prop)
            if not value:
                continue
            layers = _layers(value, tokens)
            seal = _seal_index(layers)
            if seal is None or seal == 0:
                continue  # nothing is stacked above the seal
            grounds = _grounds(value, tokens)
            assert grounds, f"{sel} seals its stack but reads no ground"
            seal_colours = {tuple(round(c, 3) for c in rgb) for _l, rgb, _a in layers[seal]}
            read = {tuple(round(c, 3) for c in rgb) for _l, rgb in grounds}
            assert read - seal_colours, (
                f"{_where(condition)} {sel} stacks {seal} layer(s) over its seal and still "
                f"reads as the seal alone: the wash was dropped"
            )
            sealed.append(f"{_where(condition)} {sel}".strip())
    assert sealed, "no rule in app.css stacks a wash over an opaque layer; this model is untested by it"


# ── 7. every ground that carries text, chrome included ──────────────────
#
# `PROSE_CONTAINER` is a hand-written list, so a container outside it could
# paint a ground no check read, and the chrome grounds -- bars, tabs, badges,
# pills, chips, terms -- were outside every oracle by construction. Deleting
# the `.topbar` overrides from the accessibility block left the bar translucent
# and every check green.
#
# The set of grounds is derived here instead: every rule that paints a
# background, in any condition, on a box the stylesheet shows can hold a line
# of text. Nothing is listed; `GROUND_ANCHORS` below is an anti-vacuity guard on
# the derivation, not its input.
#
# What text lands on a ground is derived the same way -- the rule's own colour,
# colours declared by rules that also match the element or descend from it, and
# (only where the rule pins no colour of its own, so inheritance reaches it) the
# page's three reading steps.
#
# The floor applies to the reading palette. Chrome chroma is outside it, and not
# by oversight: under a 25% black multiply the WCAG flare term caps
# `--violet` at 3.20, `--magenta` at 3.61 and `--red` at 3.58 against a ground
# of *pure black* -- no ground this page or any other could paint brings those
# three to 4.5, so gating them would be a demand on the overlay, not on app.css.
# Their measured values are reported with the ticket rather than asserted here.
# What is asserted for every text ground, chroma included, is that its colour is
# determinate: the stack seals, so the ground does not depend on a backdrop that
# under an overlay extension is not this page's to name.

# Pseudo-elements that are text, or that style text, rather than a widget part a
# UA paints. A ground on one of these carries text by definition; a ground on
# `::-webkit-scrollbar-thumb` carries none and never can.
TEXT_PSEUDO_ELEMENTS = (
    "::selection",
    "::first-line",
    "::first-letter",
    "::placeholder",
    "::-webkit-input-placeholder",
    "::marker",
)

# Declarations that typeset a box. A rule that sets any of them, or pads the box
# out, is a rule about somewhere text goes.
TYPESETTING = (
    "font",
    "font-size",
    "font-family",
    "font-weight",
    "font-style",
    "font-variant-numeric",
    "line-height",
    "letter-spacing",
    "text-transform",
    "text-align",
    "text-overflow",
    "white-space",
)

# Grounds that must stay inside the derivation. Not the input to it and not a
# frozen count: a classifier quietly narrowed until it passes is what this has
# to catch, and these are the surfaces F7 named -- the chrome that carries text
# outside the reading column, plus the reading surfaces tickets 01 and 02 cover.
GROUND_ANCHORS = (
    ".topbar",
    ".ingest-status",
    ".badge",
    ".pill",
    ".chip",
    ".term",
    ".btn",
    ".ingest-stage",
    "nav.tabs a.on",
    ".panel",
    ".story",
    ".panel > h3",
    "::selection",
)

# Of those, the ones whose ink is the reading palette, so the 4.5 floor is
# actually applied to them. A ground pinned to chroma -- `.ingest-stage`, an
# active tab, a selection -- is reached and determinate but carries no reading
# step, and the anchors above are what keep it reached. These are the ones a
# narrowed ink derivation would quietly stop weighing, which is the other half
# of the same hole: F7 was demonstrated by the bars staying translucent, and it
# would be demonstrated just as well by the bars staying unweighed.
READING_GROUND_ANCHORS = (
    ".topbar",
    ".ingest-status",
    ".badge",
    ".pill",
    ".chip",
    ".term",
    ".btn",
    ".panel",
    ".story",
    ".panel > h3",
)


def _compounds(selector: str) -> list[str]:
    """One complex selector's compounds, outermost first."""
    return [p for p in re.split(r"\s*[>+~]\s*|\s+", selector.strip()) if p]


_SIMPLE = re.compile(
    r"::[-\w]+(?:\([^)]*\))?"  # pseudo-element
    r"|:[-\w]+(?:\([^)]*\))?"  # pseudo-class, arguments kept whole
    r"|#[-\w]+"
    r"|\.[-\w]+"
    r"|\[[^\]]*\]"
    r"|\*"
    r"|^[-\w]+"
)


def _simples(compound: str) -> frozenset[str]:
    return frozenset(_SIMPLE.findall(compound))


def _key(selector: str) -> frozenset[str]:
    """The simple selectors the subject of a complex selector must carry."""
    compounds = _compounds(selector)
    return _simples(compounds[-1]) if compounds else frozenset()


def _ancestor_keys(selector: str) -> list[frozenset[str]]:
    return [_simples(c) for c in _compounds(selector)[:-1]]


def _subsequence(short: list[frozenset[str]], long: list[frozenset[str]]) -> bool:
    """Whether every constraint in `short` appears, in order, inside `long`."""
    i = 0
    for want in short:
        while i < len(long) and not want <= long[i]:
            i += 1
        if i == len(long):
            return False
        i += 1
    return True


def _also_matches(other: str, subject: str) -> bool:
    """Whether `other` can style the very element `subject` styles.

    `other` may be the same selector, a tighter variant of it (`.pill.v-skip`
    beside `.pill`) or a looser one (`td` beside `tr:hover td`). What it may not
    be is a different element.
    """
    if not (_key(subject) <= _key(other) or _key(other) <= _key(subject)):
        return False
    a, b = _ancestor_keys(other), _ancestor_keys(subject)
    return _subsequence(a, b) or _subsequence(b, a)


def _descends_from(other: str, subject: str) -> bool:
    """Whether `other` styles something inside the box `subject` styles."""
    return any(anc and anc <= _key(subject) for anc in _ancestor_keys(other))


def _selector_rules(rules):
    """Every rule split per selector, numbered in source order."""
    out = []
    for index, (condition, selector, decls) in enumerate(rules):
        for part in sorted(_selector_parts(selector)):
            out.append((index, condition, part, decls))
    return out


def _declares_background(decls: dict[str, str]) -> bool:
    return any(decls.get(p) for p in ("background", "background-color", "background-image"))


def _background_value(decls: dict[str, str]) -> str:
    """The one value that decides the stack, longhands folded in below it.

    Declaration order decides, because within one rule the `background` shorthand
    resets `background-color` and a `background-color` written after it sets the
    colour back. Reading the shorthand whenever it is present -- which is what
    this did -- let a rule seal on paper and composite in the browser: an opaque
    `background` followed by a translucent `background-color` was read as the
    opaque one, so the stack looked sealed and the page painted a wash with
    nothing behind it but the backdrop.
    """
    value = decls.get("background") or decls.get("background-color") or ""
    image = decls.get("background-image")
    if image and not decls.get("background"):
        value = f"{image} {value}".strip() if value else image
    colour = decls.get("background-color")
    if decls.get("background") and colour:
        order = list(decls)
        if order.index("background-color") > order.index("background"):
            # The shorthand keeps only its image layers; the longhand is the
            # bottom colour. A colour can only legally sit in the last layer, so
            # that is the only one whose colour terms are replaced.
            parts = _split_top(value)
            images = [t for t in _top_level_terms(parts[-1]) if _IMAGE_FN.search(t)]
            parts[-1] = " ".join(images + [colour]) if images else colour
            value = ", ".join(parts)
    return value


def _holds_text(subject: str, decls: dict[str, str], all_rules) -> bool:
    """Whether the stylesheet shows this box can hold a line of text.

    Five ways it can, each read off declarations the stylesheet makes rather
    than guessed from the name: the rule colours text; the box *is* text, being
    a text-bearing pseudo-element; it is the page root, which holds everything;
    the stylesheet writes rules for things inside it; or some rule that styles
    the same element typesets or pads it.

    What falls outside is the page's graphics -- a 7px status dot, a 4px meter
    track, the gradient fill inside it, a scrollbar thumb. None of them is
    named: each is excluded because no rule in app.css colours text in it,
    typesets it, pads it or puts anything inside it.

    `content: ""` excludes an empty decorative pseudo-element, and it is tested
    *after* the rule's own colour rather than before it: a rule that colours text
    holds text whatever its content string says. Ordered the other way round,
    adding `content: ""` to any rule took it and its ground out of this gate
    entirely -- an unsealed wash under a reading token passed every criterion.
    """
    if decls.get("color"):
        return True
    if decls.get("content", "").strip() in ('""', "''"):
        return False
    key = _key(subject)
    if any(simple in TEXT_PSEUDO_ELEMENTS for simple in key):
        return True
    if key & {"body", "html", ":root"}:
        return True
    for _i, _c, other, other_decls in all_rules:
        if _descends_from(other, subject):
            return True
        if _also_matches(other, subject) and (
            any(prop in other_decls for prop in TYPESETTING)
            or any(prop == "padding" or prop.startswith("padding-") for prop in other_decls)
        ):
            return True
    return False


def _ink_on(subject: str, all_rules, tokens) -> dict[str, str]:
    """Every text colour that can land on this box's ground, as label -> colour.

    A rule carrying its own background is skipped: it brings its own ground and
    is measured there, so `.chip.on`'s near-black ink is never weighed against
    the plain `.chip` wash it replaces.
    """
    found: dict[str, str] = {}
    pinned = False
    for _i, _c, other, other_decls in all_rules:
        colour = other_decls.get("color")
        if not colour:
            continue
        same = _also_matches(other, subject)
        if same:
            pinned = True
        if not (same or _descends_from(other, subject)):
            continue
        if _declares_background(other_decls) and other != subject:
            continue
        literal = _resolve(colour.strip(), tokens)
        if _HEX_ONLY.match(literal):
            found[f"{other} ({colour.strip()})"] = literal
    if not pinned:
        # Nothing pins this box's colour, so the reading palette inherits into
        # it -- every step of it, because which step lands where is the
        # template's fact and not this stylesheet's.
        for token in PROSE_TOKENS:
            found[f"inherited {token}"] = tokens[token].strip()
    return found


def _paints_a_ground(value: str, tokens: dict[str, str]) -> bool:
    """Whether a background value puts any colour down at all.

    `background: none` and `background: transparent` clear a UA background
    rather than paint one. Counting them as grounds would have demanded that
    every bare `<button>` seal a stack it does not paint, while the ground its
    text really lands on -- the panel, story or row containing it -- is already
    weighed against the whole reading palette on its own rule.
    """
    layers = _layers(value, tokens)
    return any(alpha > 0 for layer in layers for _label, _rgb, alpha in layer)


# Everything a background value may say that is not a colour: positions, sizes,
# repeat, attachment and box keywords, the gradient geometry and interpolation
# words, and any length, percentage, angle or bare number. Closed vocabularies
# both, so a term that is in neither this set nor `_paints` is a term the reader
# does not understand -- and the unknown direction fails loudly rather than
# quietly, which is the whole point of separating the two.
_BACKGROUND_KEYWORD = frozenset(
    """
    / auto cover contain none
    top right bottom left center
    repeat repeat-x repeat-y no-repeat space round
    scroll fixed local border-box padding-box content-box
    at to in from circle ellipse
    closest-side closest-corner farthest-side farthest-corner
    srgb srgb-linear display-p3 a98-rgb prophoto-rgb rec2020 xyz xyz-d50 xyz-d65
    lab oklab lch oklch hsl hwb shorter longer increasing decreasing hue
    """.split()
)
_NUMERIC_TERM = re.compile(
    r"[+-]?(?:\d+\.?\d*|\.\d+)(?:px|pt|pc|in|cm|mm|q|em|rem|ex|ch|vw|vh|vmin|vmax|%|deg|grad|rad|turn)?\Z",
    re.I,
)


def _unreadable_terms(value: str, tokens: dict[str, str]) -> list[str]:
    """Terms in a background value this reader cannot account for.

    `_paints` returns nothing both for a term that paints no colour and for a
    colour syntax it does not implement -- a named colour, `hsl()`, `color-mix()`,
    `oklch()`, an eight-digit hex. Read as "paints nothing", the second case took
    the ground out of every check below without failing anything: a chrome surface
    at `background: white` or `background: hsl(0 0% 97% / 30%)` was gated by
    nothing, which is `f5-unread-ground` rebuilt one syntax at a time. On an
    anchored surface `GROUND_ANCHORS` caught it, which means the protection was a
    list of thirteen names -- the thing this ticket is meant to stop maintaining.

    A term reported here is not necessarily a defect in the stylesheet: it may be
    a colour `_paints` should learn, or a background keyword this set should. What
    it may not be is invisible.
    """
    out: list[str] = []
    for part in _split_top(value):
        for term in _top_level_terms(part):
            if _IMAGE_FN.search(term) and "(" in term:
                out.extend(_unreadable_terms(term[term.index("(") + 1 : term.rindex(")")], tokens))
                continue
            if _paints(term, tokens):
                continue
            if term.lower() in _BACKGROUND_KEYWORD or _NUMERIC_TERM.fullmatch(term):
                continue
            out.append(term)
    return out


def _fractional_opacities(rules) -> dict[str, float]:
    """Selectors painted at a group opacity strictly between 0 and 1.

    `opacity` is the stylesheet's second compositing mechanism and the one the
    background model above cannot see. It composites the element's whole subtree
    -- the ink and the ground it sits on, together -- against whatever is behind
    the element, so a sealed stack stops deciding what the glyph lands on and the
    ratio measured from the declarations is a number no browser produces. It is
    the same defect `f4-sealed-wash` named, through the other mechanism.

    Zero is not here. At `opacity: 0` nothing is painted at all, so there is no
    ground and no text to put a number on -- that is how the toast stays hidden
    until `.toast.show` brings it back to 1, and flagging it would be a false
    failure. Between the two the ground is real and indeterminate.

    app.css already knows this: `.story.stale` dropped `opacity: .78` for an
    opaque colour demotion because "an overlay extension multiplied it a second
    time", and the accessibility block sets `opacity: 1` on the decorations it
    keeps. What neither reached was `.btn:disabled`, whose ground is one this
    check measures.
    """
    found: dict[str, float] = {}
    for _index, _condition, subject, decls in _selector_rules(rules):
        raw = decls.get("opacity")
        if raw is None:
            continue
        try:
            alpha = float(raw.strip().rstrip("%")) / (100 if "%" in raw else 1)
        except ValueError:
            continue
        if 0 < alpha < 1:
            found[subject] = min(alpha, found.get(subject, 1.0))
    return found


def _text_grounds(rules, tokens):
    """Every (rule, ground, ink) the page paints text on. The whole derivation."""
    all_rules = _selector_rules(rules)
    for index, condition, subject, decls in all_rules:
        if not _declares_background(decls):
            continue
        value = _background_value(decls)
        # Paints nothing *and* was fully understood: a cleared UA background, and
        # the only reason to drop a rule here. A value carrying a term the reader
        # cannot account for is yielded instead, so the check reports it rather
        # than mistaking "I cannot read this" for "this paints nothing".
        if not _paints_a_ground(value, tokens) and not _unreadable_terms(value, tokens):
            continue
        if not _holds_text(subject, decls, all_rules):
            continue
        yield index, condition, subject, decls, _ink_on(subject, all_rules, tokens)


def test_all_text_grounds_clear_aa_under_the_extension_overlay(rules, tokens):
    """Every ground under text is determinate, and clears AA for the reading palette."""
    # Three relations this derivation rests on, pinned before it runs, because each
    # of them decides what the gate does *not* look at -- and a gate that looks at
    # nothing passes. What `_holds_text` excludes, what `_background_value` reads as
    # the stack, and which terms of it `_unreadable_terms` cannot account for.
    assert _holds_text(".x::before", {"content": '""', "color": "var(--text)"}, []), (
        "a rule that colours text holds text whatever its content string says; ordered "
        "the other way round, `content: \"\"` took any rule out of this gate entirely"
    )
    assert not _holds_text(".x::before", {"content": '""'}, []), (
        "an empty decorative pseudo-element that colours nothing still holds no text"
    )
    assert _background_value({"background": "#000", "background-color": "rgba(255,255,255,.4)"}) == (
        "rgba(255,255,255,.4)"
    ), "a `background-color` written after the shorthand takes the colour, so it decides the seal"
    assert _background_value({"background-color": "rgba(255,255,255,.4)", "background": "#000"}) == "#000", (
        "written before it, the shorthand resets it and wins outright"
    )
    assert _unreadable_terms("#0b0b1e linear-gradient(180deg, rgba(0,0,0,.2), transparent 62%)", tokens) == []
    assert _unreadable_terms("white", tokens) == ["white"]
    assert _unreadable_terms("hsl(0 0% 97% / 30%)", tokens) == ["hsl(0 0% 97% / 30%)"]
    assert _unreadable_terms("color-mix(in srgb, #fff 3%, #0b0b1e)", tokens) == [
        "color-mix(in srgb, #fff 3%, #0b0b1e)"
    ], "an unreadable colour must be reported, never read as painting nothing"

    measured: dict[str, list[str]] = {}
    reached: set[str] = set()
    unsealed, failures, exempted, unreadable = [], [], [], []
    faded = _fractional_opacities(rules)

    for _index, condition, subject, decls, ink in _text_grounds(rules, tokens):
        reached.add(subject)
        where = f"{_where(condition)} " if condition else ""
        value = _background_value(decls)
        strange = _unreadable_terms(value, tokens)
        if strange:
            unreadable.append(f"{where}{subject} -> {value}  (cannot read: {strange})")
            continue
        grounds = _grounds(value, tokens)
        if grounds is None:
            unsealed.append(f"{where}{subject} -> {value}")
            continue
        palette = {tokens[token].strip(): token for token in PROSE_TOKENS}
        for label, ground in grounds:
            for ink_label, literal in ink.items():
                fg = _hex_rgb(literal)
                ratio = _contrast(
                    _over_black(fg, EXTENSION_BLACK_ALPHA),
                    _over_black(ground, EXTENSION_BLACK_ALPHA),
                )
                if literal not in palette:
                    # Chrome chroma. Not gated here -- see the section header for
                    # the ceiling that makes gating it unsatisfiable -- but kept
                    # rather than dropped, so the exemption is a set this check
                    # can state and guard instead of a silent `continue`.
                    exempted.append((ratio, f"{where}{subject}", ink_label, literal))
                    continue
                measured.setdefault(subject, []).append(f"{palette[literal]} {ratio:.2f}")
                if ratio < 4.5:
                    failures.append(
                        f"{where}{subject} ({label}): {ink_label} on "
                        f"rgb({', '.join(f'{c:.0f}' for c in ground)}) is {ratio:.2f}, needs 4.5"
                    )
                for fade_sel, alpha in faded.items():
                    if _covers(subject, fade_sel) or _descends_from(fade_sel, subject):
                        failures.append(
                            f"{where}{subject} ({label}): {fade_sel} paints it at opacity "
                            f"{alpha}, so the {ratio:.2f} measured for {ink_label} here is not "
                            f"what the browser paints -- group opacity composites ink and "
                            f"ground together against a backdrop this page cannot name"
                        )

    assert not unreadable, (
        "a ground under text is written in a syntax this reader does not understand, so "
        "nothing below measured it. Teach `_paints` the colour, or `_BACKGROUND_KEYWORD` "
        "the keyword -- do not leave it unread:\n" + "\n".join(sorted(unreadable))
    )
    assert not unsealed, (
        "a ground under text takes its colour from whatever is behind it, which under an "
        "overlay extension is not this page:\n" + "\n".join(sorted(unsealed))
    )
    assert not failures, "\n".join(sorted(failures))

    unreached = [anchor for anchor in GROUND_ANCHORS if anchor not in reached]
    assert not unreached, f"the ground derivation no longer reaches: {unreached}"

    unweighed = [anchor for anchor in READING_GROUND_ANCHORS if anchor not in measured]
    assert not unweighed, (
        f"these grounds are reached but no reading step is weighed against them any "
        f"more, so the 4.5 floor no longer applies to them: {unweighed}"
    )

    # The exemption is a partition, and this is the edge it can rot along. Reading
    # ink reaches the gate by resolving to one of the three palette literals; if a
    # token stopped resolving -- renamed, moved out of `:root`, misspelled in
    # PROSE_TOKENS -- its rules would keep their `var(--text-dim)` labels and fall
    # out of the gate into this set, and every assertion above would still pass
    # with the floor quietly applying to less of the page.
    assert exempted, "no chroma ink was found at all, so this partition is vacuous"
    leaked = sorted({
        f"{subject} <- {ink_label}"
        for _ratio, subject, ink_label, _literal in exempted
        if any(token in ink_label for token in PROSE_TOKENS)
    })
    assert not leaked, (
        "reading ink fell out of the gate into the chroma exemption, so the 4.5 floor "
        "stopped applying to it -- a prose token is no longer resolving to its "
        f"literal:\n" + "\n".join(leaked)
    )


# ── 8. the accessibility block has to actually change something ──────────
#
# The block promises to flatten every ground that carries text for a reader who
# has already told the browser they cannot take another composited layer. A rule
# that re-declares the value it is overriding keeps that promise in form and
# breaks it in substance: the scan above it can see a rule is present for the
# surface and cannot see that the rule does nothing.

# The three signals, as the stylesheet spells them. A rule arriving under any of
# them is in the block.
A11Y_SIGNALS = ("prefers-contrast", "prefers-reduced-transparency", "forced-colors")


def _covers(outer: str, inner: str) -> bool:
    """Whether `outer` matches every element `inner` matches.

    Compared by constraint, not by text: `.badge` covers
    `.badge.c-research`, and `.panel > h3` covers itself. Combinator strictness
    is not modelled -- a `>` in `outer` against a descendant in `inner` would be
    read as covering when it is not -- which is sound for this stylesheet, where
    the block writes `>` only where the base rule it overrides writes it too.

    Pseudo-elements must match. A pseudo-element is its own box with its own
    declaration block, so a ground set on `nav.tabs a.on` never overrides the one
    on `nav.tabs a.on::after` and is not competing with it in the cascade. Ink is
    a different question -- what `::before` paints does land on the originating
    element's ground -- and `_also_matches` stays permissive for that reason.
    """
    outer_pseudo = frozenset(s for s in _key(outer) if s.startswith("::"))
    inner_pseudo = frozenset(s for s in _key(inner) if s.startswith("::"))
    if outer_pseudo != inner_pseudo:
        return False
    return _key(outer) <= _key(inner) and _subsequence(_ancestor_keys(outer), _ancestor_keys(inner))


def _a11y_rules(rules):
    return [
        (index, condition, part, decls)
        for index, condition, part, decls in _selector_rules(rules)
        if any(signal in condition for signal in A11Y_SIGNALS)
    ]


def _base_rules(rules):
    return [
        (index, condition, part, decls)
        for index, condition, part, decls in _selector_rules(rules)
        if not condition
    ]


def _same_value(a: str, b: str, tokens: dict[str, str]) -> bool:
    """Whether two declaration values paint the same thing.

    Resolved through the palette and whitespace-normalised, so
    `background: var(--panel)` and `background: #0b0b1e` are one value and the
    restatement cannot hide behind a token name.
    """
    def norm(value: str) -> str:
        layers = [_resolve(term, tokens) for term in _split_top(value)]
        return ", ".join(" ".join(layer.split()) for layer in layers).lower()

    return norm(a) == norm(b)


def test_a11y_not_vacuous_no_block_rule_restates_what_it_overrides(rules, tokens):
    """No rule in the block is an exact restatement of what it overrides."""
    block = _a11y_rules(rules)
    base = _base_rules(rules)
    assert block, "app.css answers none of the accessibility signals"

    overriding, vacuous = 0, []
    for _bi, condition, block_sel, block_decls in block:
        for _i, _c, base_sel, base_decls in base:
            if not _covers(block_sel, base_sel):
                continue
            shared = [prop for prop in block_decls if prop in base_decls]
            if not shared or len(shared) != len(block_decls):
                continue  # the block declares something the base rule does not
            overriding += 1
            if all(_same_value(block_decls[prop], base_decls[prop], tokens) for prop in shared):
                vacuous.append(
                    f"@media ({condition.splitlines()[0].strip()} ...) {block_sel} "
                    f"restates {base_sel}: "
                    + "; ".join(f"{prop}: {block_decls[prop]}" for prop in shared)
                )

    assert overriding, "no rule in the block overrides a base rule; this scan is vacuous"
    assert not vacuous, (
        "these rules say nothing, so the block's promise to flatten the surface is "
        "kept in form only:\n" + "\n".join(sorted(set(vacuous)))
    )


# ── 9. the block's flattening has to win the cascade ─────────────────────
#
# Ticket 01's Risks 5 named this for `.badge.c-*` and understated it: the block
# flattened `.badge` at one class while nine category rules set the same property
# at two, and the same defeat hit five `.pill` variants and
# `.ingest-status.live .ingest-stage` at three. Every one of those block
# declarations was dead on arrival, and nothing said so -- a presence scan sees
# the rule and cannot see that the cascade throws it away.
#
# So specificity is compared here rather than assumed, both ways round. A
# surface whose base ground still composites must have a block rule that covers
# it and wins; and a block rule that declares a ground must win against every
# base rule that sets one on an element it covers, or it is a promise the
# cascade does not keep.

_FUNCTIONAL_PSEUDO = re.compile(r"^:(?:not|is|has|matches|any)\(", re.I)


def _specificity(selector: str) -> tuple[int, int, int]:
    """(ids, classes, types) for one complex selector, per Selectors 4.

    `:is()`, `:not()` and `:has()` take the specificity of their most specific
    argument; `:where()` takes none; `*` adds nothing.
    """
    ids = classes = types = 0
    for compound in _compounds(selector):
        for simple in _SIMPLE.findall(compound):
            lowered = simple.lower()
            if simple.startswith("::"):
                types += 1
            elif simple.startswith("#"):
                ids += 1
            elif simple.startswith((".", "[")):
                classes += 1
            elif simple.startswith(":"):
                if lowered.startswith(":where("):
                    continue
                match = _FUNCTIONAL_PSEUDO.match(simple)
                if match:
                    best = (0, 0, 0)
                    for argument in _split_top(simple[len(match.group(0)) : -1]):
                        best = max(best, _specificity(argument))
                    ids, classes, types = ids + best[0], classes + best[1], types + best[2]
                else:
                    classes += 1
            elif simple != "*":
                types += 1
    return ids, classes, types


def _important(value: str) -> bool:
    return value.rstrip().endswith("!important")


def _wins(winner, loser, prop: str) -> bool:
    """Whether `winner` beats `loser` for `prop`: importance, then specificity, then order."""
    w_index, _wc, w_sel, w_decls = winner
    l_index, _lc, l_sel, l_decls = loser
    w_bang, l_bang = _important(w_decls.get(prop, "")), _important(l_decls.get(prop, ""))
    if w_bang != l_bang:
        return w_bang
    w_spec, l_spec = _specificity(w_sel), _specificity(l_sel)
    if w_spec != l_spec:
        return w_spec > l_spec
    return w_index > l_index


def _is_one_flat_colour(value: str, tokens: dict[str, str]) -> bool:
    """Whether a background is a single opaque colour -- nothing left to flatten."""
    layers = _layers(value, tokens)
    return len(layers) == 1 and len(layers[0]) == 1 and layers[0][0][2] >= 1.0


def _background_property(decls: dict[str, str]) -> str:
    for prop in ("background", "background-color", "background-image"):
        if decls.get(prop):
            return prop
    return ""


def test_a11y_specificity_every_flattening_rule_wins_the_cascade(rules, tokens):
    """The block's grounds beat the base rules they override, and cover what composites."""
    # The comparison is only as good as the counter. A `_specificity` that
    # returned a constant would make every claim below true and every defeat
    # invisible, so the selectors this stylesheet actually turns on are pinned
    # here -- including the three shapes that caused the defeats: a variant class,
    # a descendant of a compound, and a pseudo-class on a type.
    assert _specificity(".badge") == (0, 1, 0)
    assert _specificity(".badge.c-research") == (0, 2, 0)
    assert _specificity("a.pill:hover") == (0, 2, 1)
    assert _specificity(".ingest-status.live .ingest-stage") == (0, 3, 0)
    assert _specificity("nav.tabs a.on") == (0, 2, 2)
    assert _specificity(".panel > h3") == (0, 1, 1)
    assert _specificity(".panel > h3 > span:last-child:not(:first-child)") == (0, 3, 2)
    assert _specificity("body") == (0, 0, 1)
    assert _specificity("::selection") == (0, 0, 1)
    assert _specificity(":where(.a.b) .c") == (0, 1, 0), ":where() contributes no specificity"

    # And the comparison that uses it, on the three axes in the order the cascade
    # applies them. A `_wins` that answered yes to everything would make the
    # whole check below assert nothing at all, which is the failure mode this
    # criterion is most exposed to.
    def rule(index, selector, value):
        return (index, "", selector, {"background": value})

    assert _wins(rule(2, ".badge", "red"), rule(1, ".badge", "blue"), "background"), (
        "equal specificity is decided by source order"
    )
    assert not _wins(rule(1, ".badge", "red"), rule(2, ".badge", "blue"), "background"), (
        "an earlier rule does not beat a later one at equal specificity"
    )
    assert not _wins(rule(9, ".badge", "red"), rule(1, ".badge.c-x", "blue"), "background"), (
        "source order does not rescue a less specific rule -- this is the F7 defeat"
    )
    assert _wins(rule(1, ".badge.c-x", "red"), rule(9, ".badge", "blue"), "background"), (
        "higher specificity beats later source order"
    )
    assert _wins(rule(1, ".badge", "red !important"), rule(9, ".badge.c-x", "blue"), "background"), (
        "importance outranks specificity"
    )

    # The two inputs `_wins` reads that the cases above hand it ready-made. They
    # were the rest of the same hole: the pins construct their own tuples and pass
    # `"background"` as a literal, so neither the property the comparison is run on
    # nor the base set it is run against was pinned by them, and gutting either
    # left this criterion green.
    assert _background_property({"background": "red"}) == "background"
    assert _background_property({"background-color": "red"}) == "background-color"
    assert _background_property({"background-image": "url(x)"}) == "background-image"
    assert _background_property({"color": "red"}) == "", "a rule with no background names no property"
    longhand_bang = (1, "", ".badge", {"background-color": "red !important"})
    longhand_plain = (9, "", ".badge.c-x", {"background-color": "blue"})
    assert _wins(longhand_bang, longhand_plain, _background_property(longhand_bang[3])), (
        "importance on a longhand outranks specificity too -- which it cannot do if "
        "`_background_property` reports a property the rules do not declare"
    )

    block = [r for r in _a11y_rules(rules) if _declares_background(r[3])]
    base = [r for r in _base_rules(rules) if _declares_background(r[3])]
    assert block, "the accessibility block flattens no ground at all"

    # The base cascade the comparison runs against, anchored the way
    # `GROUND_ANCHORS` anchors the derivation. `block` and `composites` each carry
    # an anti-vacuity guard and this set carried none, so a `_base_rules` narrowed
    # to type selectors -- dropping every class rule, which is every rule the 22
    # defeats lived on -- emptied the `defeated` half below in silence. These three
    # are the defeat shapes the criterion was written for.
    base_seen = {sel for _i, _c, sel, _d in base}
    missing_base = [
        sel
        for sel in (".badge.c-research", "a.pill:hover", ".ingest-status.live .ingest-stage")
        if sel not in base_seen
    ]
    assert not missing_base, (
        f"the base cascade this comparison runs against no longer contains the rules the "
        f"defeats lived on, so it is comparing the block against almost nothing: {missing_base}"
    )

    # Which surfaces still need flattening: a ground that carries text and is not
    # already one opaque colour. Derived from the same ground model as the
    # contrast check, so a surface cannot need flattening there and not here.
    composites = [
        (index, condition, subject, decls)
        for index, condition, subject, decls, _ink in _text_grounds(rules, tokens)
        if not condition and not _is_one_flat_colour(_background_value(decls), tokens)
    ]
    assert composites, "no base ground composites; this scan is vacuous"

    uncovered = []
    for rule in composites:
        _i, _c, subject, decls = rule
        prop = _background_property(decls)
        winners = [
            k for k in block if _covers(k[2], subject) and _wins(k, rule, prop)
        ]
        if not winners:
            covering = [k[2] for k in block if _covers(k[2], subject)]
            uncovered.append(
                f"{subject} {_specificity(subject)} still composites and the block "
                + (
                    f"only offers {covering} which lose on specificity"
                    if covering
                    else "flattens it with nothing"
                )
            )

    defeated = []
    for rule in block:
        _bi, condition, block_sel, block_decls = rule
        prop = _background_property(block_decls)
        for other in base:
            _i, _c, base_sel, base_decls = other
            if not _covers(block_sel, base_sel):
                continue
            if not _background_property(base_decls):
                continue
            if not _wins(rule, other, prop):
                defeated.append(
                    f"@media ({condition.splitlines()[0].strip()} ...) {block_sel} "
                    f"{_specificity(block_sel)} loses to {base_sel} "
                    f"{_specificity(base_sel)}, so its flattening never applies"
                )

    assert not uncovered, "\n".join(sorted(uncovered))
    assert not defeated, "\n".join(sorted(set(defeated)))
