"""app.css under a third-party browser overlay extension.

A CRT filter, a VHS tint or a dark-mode inverter injects a `position: fixed`
full-viewport layer at the top of the stacking order, usually painting through
`mix-blend-mode`. The page cannot out-stack it and cannot remove it, so the
only budget it controls is its own: how much darkening app.css has already
spent before the extension's lands, and whether prose contrast is carried by
opaque colour or borrowed from whatever happens to be behind.

app.css spends none of that budget: it paints no layer of its own over its own
content, which is what `test_no_crt_layers_remain` holds. The compositing model
stays anyway, because the way a layer comes back is quietly — so these tests
read app.css as a stylesheet rather than as text, resolve the `:root` palette,
composite whatever the page does stack over a reading surface the way the
cascade would, then composite a worst-case extension overlay on top and check
the surviving WCAG ratio. A layer put back, a prose token dimmed, or a ground
left taking its colour from whatever is behind it fails here rather than in
someone's browser.
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
