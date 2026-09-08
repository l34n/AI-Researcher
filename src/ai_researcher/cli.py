"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import Settings, load_sources
from .db import Database
from .enrich import ChatRouter, Judge
from .pipeline import Pipeline, sync_sources
from .research import DeepResearcher
from .trends import backfill_topics, build_clusters, generate_brief
from .web import queries as Q


def _log(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
    )


def _ctx() -> tuple[Settings, Database]:
    settings = Settings.load()
    return settings, Database(settings.db_path)


# ── commands ─────────────────────────────────────────────────────────
def cmd_run(args) -> int:
    settings, db = _ctx()
    pipeline = Pipeline(settings, db)
    only = args.source or None
    result = asyncio.run(
        pipeline.run(only=only, skip_ingest=args.no_ingest, force_brief=args.force_brief)
    )

    if result.get("status") == "busy":
        print(f"\n  {result.get('error', 'another run is active')}")
        print("  nothing was changed; try again once it finishes.\n")
        return 0

    ingest = result.get("ingest") or {}
    print()
    print(f"  run #{result.get('run_id')}  {result.get('status')}  "
          f"{result.get('elapsed_s')}s")
    if ingest:
        print(f"  sources : {ingest.get('ok', 0)} ok, "
              f"{ingest.get('failed', 0)} failed, {ingest.get('skipped', 0)} skipped")
        print(f"  items   : {ingest.get('new_items', 0)} new")
    enrich = result.get("enrich") or {}
    if enrich:
        print(f"  enriched: {enrich.get('enriched', 0)} "
              f"({enrich.get('llm', 0)} by model, {enrich.get('heuristic', 0)} heuristic), "
              f"{enrich.get('pending', 0)} pending")
    embed = result.get("embed") or {}
    if embed.get("embedded") or embed.get("total"):
        note = f", {embed['failed_batches']} batches failed" if embed.get("failed_batches") else ""
        print(f"  embedded: {embed.get('embedded', 0)} new "
              f"({embed.get('total', 0)} total{note})")
    cluster = result.get("cluster") or {}
    if cluster:
        print(f"  stories : {cluster.get('clusters', 0)} from "
              f"{cluster.get('items', 0)} items via {cluster.get('method')}")
    judge = result.get("judge") or {}
    if judge:
        print(f"  judged  : {judge.get('judged', 0)} "
              f"({judge.get('llm', 0)} by model)  "
              f"adopt={judge.get('adopt', 0)} research={judge.get('research', 0)}")
    research = result.get("research") or {}
    if research:
        print(f"  research: {research.get('researched', 0)} briefs "
              f"({research.get('llm', 0)} model, {research.get('fallback', 0)} fallback)")
    ollama = result.get("ollama") or {}
    if ollama:
        print(f"  ollama  : {'ready' if ollama.get('available') else 'unavailable'} "
              f"chat={ollama.get('chat_model') or '-'} "
              f"embed={ollama.get('embed_model') or '-'}")
    chat = result.get("chat") or {}
    if chat:
        print(f"  chat    : workhorse={chat.get('workhorse') or '-'} "
              f"premium={chat.get('premium') or '-'}")
    for err in (ingest.get("errors") or [])[:12]:
        print(f"    ! {err}")
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return 1
    return 0


def _lan_address() -> str:
    """The address other machines should use to reach this host.

    Resolving the hostname is the obvious approach and the wrong one: on
    Debian-family systems /etc/hosts maps the hostname to a loopback address,
    so it confidently prints a URL that works nowhere but here. Opening a UDP
    socket toward a routable address sends no packets but makes the kernel
    pick the real outbound interface, which is the one we want to advertise.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))  # TEST-NET-1, never actually routed to
        addr = sock.getsockname()[0]
    except OSError:
        return ""
    finally:
        sock.close()
    return "" if addr.startswith("127.") else addr


def cmd_serve(args) -> int:
    import uvicorn
    settings = Settings.load()
    host = args.host or settings.host
    port = args.port or settings.port
    print(f"  dashboard : http://{host}:{port}")
    if host in ("0.0.0.0", "::"):
        lan = _lan_address()
        if lan:
            print(f"  on LAN    : http://{lan}:{port}")
    if settings.access_token:
        # Never echo the secret: under systemd this banner lands in the journal.
        print("  token     : required — append ?k=<AIR_ACCESS_TOKEN from .env>")
    uvicorn.run(
        "ai_researcher.web.app:create_app",
        host=host, port=port, factory=True,
        reload=args.reload, log_level=args.log.lower(),
    )
    return 0


def cmd_sources(args) -> int:
    settings, db = _ctx()
    sync_sources(db, load_sources(settings))
    rows = Q.source_health(db)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    width = max((len(r["name"]) for r in rows), default=20)
    print(f"\n  {'SOURCE'.ljust(width)}  {'KIND':<16} {'STATUS':<13} "
          f"{'LAST':<10} {'7d':>5}  NOTE")
    print("  " + "─" * (width + 60))
    for r in rows:
        note = r["error"][:56] if r["error"] else ""
        mark = {"ok": "✓", "not-modified": "·", "disabled": "○"}.get(r["status"], "✗")
        print(f"  {r['name'].ljust(width)}  {r['kind']:<16} "
              f"{mark} {r['status']:<11} {r['last_fetch']:<10} "
              f"{r['week_items']:>5}  {note}")
    failing = [r for r in rows if r["status"] == "error"]
    print(f"\n  {len(rows)} sources, {len(failing)} failing\n")
    return 0


def cmd_stats(args) -> int:
    _, db = _ctx()
    print(json.dumps(Q.dashboard_stats(db), indent=2))
    return 0


def cmd_brief(args) -> int:
    settings, db = _ctx()

    async def go():
        client = ChatRouter(settings)
        try:
            if args.rebuild:
                build_clusters(db)
            return await generate_brief(db, client, day=args.day, force=True)
        finally:
            await client.aclose()

    asyncio.run(go())
    brief = Q.get_brief(db, args.day)
    if brief:
        print(f"\n─── {brief['day']} "
              f"({brief['model'] or 'no model'}) ───\n")
        print(brief["markdown"])
        print()
    return 0


def cmd_doctor(args) -> int:
    """Check the things that silently degrade the dashboard."""
    settings, db = _ctx()
    print("\n  ── environment ──")
    print(f"  data dir       : {settings.data_dir}")
    print(f"  database       : {settings.db_path} "
          f"({settings.db_path.stat().st_size // 1024 if settings.db_path.exists() else 0} KB)")
    print(f"  FTS5 search    : {'yes' if db.fts_enabled else 'no (LIKE fallback)'}")

    print("\n  ── chat ──")

    async def probe():
        client = ChatRouter(settings)
        try:
            ok = await client.probe()
            return ok, client.describe(), client.installed, client.last_error
        finally:
            await client.aclose()

    ok, chat, installed, err = asyncio.run(probe())
    print(f"  ollama host    : {settings.ollama_host}")
    print(f"  ollama models  : {', '.join(installed) or 'none'}")
    print(f"  gemini         : {'configured' if settings.gemini_api_key else 'unset'} "
          f"({settings.gemini_model} / {settings.gemini_premium_model})")
    print(f"  openrouter     : {'configured' if settings.openrouter_api_key else 'unset'} "
          f"({settings.openrouter_model} / {settings.openrouter_premium_model})")
    print(f"  workhorse      : {chat.get('workhorse') or 'NONE — summaries will be extractive'}")
    print(f"  premium        : {chat.get('premium') or 'NONE'}")
    print(f"  default chat   : {chat.get('default_chat') or settings.ollama_default_chat_model}")
    print(f"  enrich/judge   : {chat.get('enrich') or '-'} / {chat.get('judge') or '-'}")
    print(f"  research/brief : {chat.get('research') or '-'} / {chat.get('brief') or '-'}")
    print(f"  ready          : {'yes' if ok else 'NO — ' + (err or chat.get('error') or 'no backend')}")
    print(f"  premium gate   : readiness >= {settings.premium_readiness} "
          f"(research, brief, and high-readiness judgment)")
    print(f"  resource cap   : {settings.max_model_memory_gb:g} GB, "
          f"ctx {settings.max_context}, "
          f"{settings.max_concurrent_generations} concurrent, "
          f"{settings.daily_model_calls} calls/day")
    if chat.get("resource_warning"):
        print(f"  WARNING        : {chat['resource_warning']}")
    last_embed = db.get_kv("embed_model")
    if not chat.get("embed"):
        print("  embed model    : NONE — clustering falls back to hashed TF-IDF")
        print("                   TF-IDF catches near-duplicate wording, not")
        print("                   'same story, different words'. Quality drop is")
        print("                   real. Fix: ollama pull nomic-embed-text")
    else:
        print(f"  embed model    : {chat.get('embed')}")
        if last_embed and last_embed != chat.get("embed"):
            print(f"                   was {last_embed}; stored vectors will be rewritten")

    print("\n  ── credentials ──")
    print(f"  GITHUB_TOKEN   : {'set' if settings.github_token else 'NOT SET — 60 req/hr limit'}")
    print(f"  X_BEARER_TOKEN : {'set' if settings.x_bearer_token else 'not set — X connector off'}")
    print(f"  REDDIT oauth   : {'set' if settings.reddit_client_id else 'not set — public Atom feeds'}")

    stats = Q.dashboard_stats(db)
    print("\n  ── content ──")
    print(f"  items total    : {stats['items_total']}")
    print(f"  items last 24h : {stats['items_24h']}")
    print(f"  stories today  : {stats['stories_today']}")
    print(f"  pending enrich : {stats['pending_enrich']}")
    print(f"  judged         : {stats.get('judged', 0)}  "
          f"(adopt {stats.get('adopt', 0)} · research {stats.get('research_ready', 0)})")
    print(f"  lab briefs     : {stats.get('research_briefs', 0)}")
    if stats.get("judged") and not stats.get("research_briefs"):
        print("                   nothing cleared AIR_RESEARCH_THRESHOLD — "
              "briefs stay empty until a story is specific and adoptable")
    print(f"  research gate  : {settings.research_threshold}  "
          f"budget {settings.research_budget}  "
          f"time {settings.research_time_budget}s")
    print(f"  sources ok     : {stats['sources_ok']}/{stats['sources_total']}"
          f"  ({stats['sources_failing']} failing)")
    print(f"  last run       : {stats['last_run']} ({stats['last_run_status']})")
    print()
    return 0


def cmd_recluster(args) -> int:
    settings, db = _ctx()
    print(json.dumps(build_clusters(db, window_hours=args.hours), indent=2))
    backfill_topics(db, days=args.backfill)
    print(f"  topics backfilled for {args.backfill} days")
    return 0


def cmd_research(args) -> int:
    """Re-judge and (re)build Karpathy briefs without fetching."""
    settings, db = _ctx()

    async def go():
        client = ChatRouter(settings)
        try:
            await client.probe()
            judged = await Judge(settings, db, client).run()
            researcher = DeepResearcher(settings, db, client)
            researched = await researcher.run(limit=args.limit, force=args.force)
            researcher.relink_clusters()
            return {"judge": judged, "research": researched}
        finally:
            await client.aclose()

    result = asyncio.run(go())
    print(json.dumps(result, indent=2))
    return 0


def cmd_eval(args) -> int:
    """Run the offline benchmark corpus. No network, no GPU."""
    from .eval import LAYERS, run_corpus

    layers = tuple(args.layers.split(",")) if args.layers else LAYERS
    layers = tuple(layer.strip() for layer in layers if layer.strip())
    report = run_corpus(layers=layers, case_ids=args.case)
    print(json.dumps(report, indent=2))
    metrics = (report.get("layers") or {}).get(report.get("best_layer") or "", {}).get("metrics") or {}
    print(f"\n  corpus {report.get('corpus_version')}  best layer {report.get('best_layer')}")
    print(f"  format {metrics.get('format_compliance')}  "
          f"fallback {metrics.get('fallback_rate')}  "
          f"injection {metrics.get('injection_following_rate')}  "
          f"hallucinated-ready {metrics.get('hallucinated_recommendation_rate')}")
    return 0


def cmd_compare(args) -> int:
    """Compare named models against the benchmark corpus.

    Offline (default) scores fixture outputs so CI stays dark. ``--live``
    actually calls each installed model; that needs Ollama or a cloud key.
    """
    from .eval import compare_models, run_corpus

    names = [n.strip() for n in (args.models or []) if n and n.strip()]
    if not names:
        print(json.dumps(run_corpus(layers=("schema", "fallback")), indent=2))
        return 0
    if not args.live:
        def generate_for(model_name: str):
            def generate(case, **_kwargs):
                return case.get("model_output") or case.get("hostile_model_output") or ""
            return generate

        print(json.dumps(compare_models(names, generate_for), indent=2))
        return 0
    settings, _db = _ctx()
    print(json.dumps(_compare_live_sync(settings, names), indent=2))
    return 0


def _compare_live_sync(settings: Settings, names: list[str]) -> dict:
    from .eval import run_corpus
    from .eval.harness import _ready_of, _stories_of
    from .sanitize import fence
    from .trends.brief import PROMPT, SYSTEM

    client = ChatRouter(settings)

    async def setup():
        await client.probe()
        return client.available

    if not asyncio.run(setup()):
        asyncio.run(client.aclose())
        return {"error": client.last_error or "no chat backend", "models": {}}

    out: dict = {"models": {}}
    try:
        for name in names:
            def generate(case, model=name, **_k):
                prompt = PROMPT.format(
                    stories=fence("STORY", json.dumps(_stories_of(case))[:2000], limit=2000),
                    rising=fence("RISING", "none", limit=40),
                    ready=fence("READY", json.dumps(_ready_of(case))[:800], limit=800),
                )

                async def once():
                    return await client.generate_text(
                        prompt, system=SYSTEM, num_predict=850,
                        premium=True, role="brief", model=model,
                    ) or ""

                return asyncio.run(once())

            out["models"][name] = run_corpus(
                generate=generate, layers=("schema", "fallback"),
            )
    finally:
        asyncio.run(client.aclose())
    return out


def cmd_backup(args) -> int:
    from .backup import backup_database, integrity_check

    settings, db = _ctx()
    if args.check:
        ok, msg = integrity_check(settings.db_path)
        print(json.dumps({"path": str(settings.db_path), "ok": ok, "integrity": msg}))
        return 0 if ok else 1
    dest = Path(args.out) if args.out else None
    result = backup_database(db, dest)
    print(json.dumps(result, indent=2))
    return 0


def cmd_restore(args) -> int:
    from .backup import restore_database

    settings, _db = _ctx()
    src = Path(args.src)
    dest = Path(args.dest) if args.dest else settings.db_path
    if not args.yes:
        print("refusing to restore without --yes (this replaces the database file)")
        return 2
    result = restore_database(src, dest)
    print(json.dumps(result, indent=2))
    return 0


def cmd_worker(args) -> int:
    """Ingest loop for the Compose worker profile. Shares the data volume."""
    import time

    settings, db = _ctx()
    interval = max(1, int(args.interval or 60))
    pipeline = Pipeline(settings, db)
    print(f"  worker interval {interval} min  data {settings.data_dir}")
    while True:
        result = asyncio.run(pipeline.run(force_brief=args.force_brief))
        print(json.dumps({
            "status": result.get("status"),
            "run_id": result.get("run_id"),
            "elapsed_s": result.get("elapsed_s"),
        }))
        if args.once:
            return 0 if result.get("status") in ("ok", "partial", "busy") else 1
        time.sleep(interval * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-researcher", description="Local AI trends dashboard"
    )
    parser.add_argument("--log", default="INFO", help="log level")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="fetch, enrich, judge, research, cluster, and brief")
    p.add_argument("--source", action="append", help="limit to source key(s)")
    p.add_argument("--no-ingest", action="store_true", help="re-analyse without fetching")
    p.add_argument("--force-brief", action="store_true", help="regenerate today's brief")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("serve", help="run the web dashboard")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("sources", help="show per-source health")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_sources)

    p = sub.add_parser("doctor", help="diagnose degraded capabilities")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("brief", help="print (and regenerate) the daily brief")
    p.add_argument("--day")
    p.add_argument("--rebuild", action="store_true", help="recluster first")
    p.set_defaults(fn=cmd_brief)

    p = sub.add_parser("stats", help="dashboard counters as JSON")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("recluster", help="rebuild stories and topic history")
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--backfill", type=int, default=14)
    p.set_defaults(fn=cmd_recluster)

    p = sub.add_parser("research", help="judge items and write deep-research briefs")
    p.add_argument("--limit", type=int, default=None, help="max stories to research")
    p.add_argument("--force", action="store_true", help="rewrite existing briefs")
    p.set_defaults(fn=cmd_research)

    p = sub.add_parser("eval", help="run the offline quality corpus")
    p.add_argument("--layers", default="", help="comma-separated harness layers")
    p.add_argument("--case", action="append", help="limit to case id (repeatable)")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("compare", help="compare models on the quality corpus")
    p.add_argument("--models", nargs="+", default=[], help="model tags to compare")
    p.add_argument("--live", action="store_true", help="call each model (needs Ollama or a key)")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("backup", help="copy the SQLite database via the backup API")
    p.add_argument("--out", help="destination path")
    p.add_argument("--check", action="store_true", help="integrity-check the live database only")
    p.set_defaults(fn=cmd_backup)

    p = sub.add_parser("restore", help="restore a backup over the live database")
    p.add_argument("src", help="backup file")
    p.add_argument("--dest", help="destination database path")
    p.add_argument("--yes", action="store_true", help="confirm replacing the database")
    p.set_defaults(fn=cmd_restore)

    p = sub.add_parser("worker", help="run ingest on an interval (Compose worker profile)")
    p.add_argument("--interval", type=int, default=60, help="minutes between runs")
    p.add_argument("--once", action="store_true", help="run once and exit")
    p.add_argument("--force-brief", action="store_true")
    p.set_defaults(fn=cmd_worker)

    args = parser.parse_args(argv)
    _log(args.log)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
