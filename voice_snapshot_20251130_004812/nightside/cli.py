#!/usr/bin/env python3
"""
Nightside CLI - Introspection into your thought patterns.

Usage:
    nightside log          # Show recent thoughts
    nightside themes       # Show extracted themes with scores
    nightside search TEXT  # Find related thoughts
    nightside stats        # Summary of thought patterns
    nightside record TOPIC TEXT  # Manually record a thought
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from nightside import choir, reflect, suggest_themes
from nightside.silent_choir import NIGHTSIDE_LOG

console = Console()


def cmd_log(args):
    """Show recent thought events"""
    events = choir.to_list(limit=args.limit)

    if not events:
        console.print("[yellow]No thoughts recorded yet.[/yellow]")
        return

    table = Table(title=f"Recent Thoughts ({len(events)})", box=box.ROUNDED)
    table.add_column("ID", style="dim", width=4)
    table.add_column("Time", style="cyan", width=12)
    table.add_column("Kind", style="green", width=10)
    table.add_column("Topic", style="yellow", width=15)
    table.add_column("Text", style="white", max_width=60)

    for evt in reversed(events[-args.limit:]):
        ts = datetime.fromtimestamp(evt["ts"]).strftime("%H:%M:%S")
        text = evt["text"][:57] + "..." if len(evt["text"]) > 60 else evt["text"]
        text = text.replace("\n", " ")
        table.add_row(
            str(evt["id"]),
            ts,
            evt["kind"],
            evt["topic"],
            text
        )

    console.print(table)
    console.print(f"[dim]Log file: {NIGHTSIDE_LOG}[/dim]")


def cmd_themes(args):
    """Show extracted themes with time-decay scores"""
    themes = reflect(limit=args.limit, half_life_days=args.half_life)

    if not themes:
        console.print("[yellow]No themes detected yet.[/yellow]")
        return

    table = Table(title=f"Themes (half-life: {args.half_life} days)", box=box.ROUNDED)
    table.add_column("Theme", style="cyan")
    table.add_column("Score", style="green", justify="right")
    table.add_column("Count", style="yellow", justify="right")
    table.add_column("Recent", style="magenta", justify="center")

    for t in themes:
        recent = "●" if t.get("recent") else "○"
        table.add_row(
            t["theme"],
            f"{t['score']:.3f}",
            str(t["count"]),
            recent
        )

    console.print(table)


def cmd_search(args):
    """Search thoughts for a pattern"""
    query = args.query.lower()
    events = choir.to_list(limit=1000)

    matches = []
    for evt in events:
        text = evt["text"].lower()
        topic = evt["topic"].lower()
        tags = " ".join(evt.get("tags", [])).lower()

        if query in text or query in topic or query in tags:
            matches.append(evt)

    if not matches:
        console.print(f"[yellow]No thoughts matching '{args.query}'[/yellow]")
        return

    console.print(f"[green]Found {len(matches)} matching thoughts:[/green]\n")

    for evt in matches[-args.limit:]:
        ts = datetime.fromtimestamp(evt["ts"]).strftime("%Y-%m-%d %H:%M")
        console.print(Panel(
            f"[dim]{ts}[/dim] | [cyan]{evt['topic']}[/cyan]\n\n{evt['text']}",
            title=f"#{evt['id']} [{evt['kind']}]",
            border_style="dim"
        ))


def cmd_stats(args):
    """Show summary statistics"""
    events = choir.to_list(limit=5000)

    if not events:
        console.print("[yellow]No thoughts recorded yet.[/yellow]")
        return

    # Compute stats
    now = time.time()
    total = len(events)
    kinds = {}
    topics = {}
    hours = [0] * 24

    oldest = min(evt["ts"] for evt in events)
    newest = max(evt["ts"] for evt in events)

    for evt in events:
        kinds[evt["kind"]] = kinds.get(evt["kind"], 0) + 1
        topics[evt["topic"]] = topics.get(evt["topic"], 0) + 1
        hour = datetime.fromtimestamp(evt["ts"]).hour
        hours[hour] += 1

    span_days = (newest - oldest) / 86400

    # Display
    console.print(Panel.fit(
        f"[bold]Total thoughts:[/bold] {total}\n"
        f"[bold]Time span:[/bold] {span_days:.1f} days\n"
        f"[bold]Avg per day:[/bold] {total / max(span_days, 1):.1f}\n"
        f"[bold]Unique topics:[/bold] {len(topics)}\n"
        f"[bold]Unique kinds:[/bold] {len(kinds)}",
        title="Nightside Stats"
    ))

    # Top kinds
    top_kinds = sorted(kinds.items(), key=lambda x: x[1], reverse=True)[:5]
    console.print("\n[bold]Top kinds:[/bold]")
    for kind, count in top_kinds:
        bar = "█" * min(count, 30)
        console.print(f"  {kind:15} {bar} {count}")

    # Top topics
    top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
    console.print("\n[bold]Top topics:[/bold]")
    for topic, count in top_topics:
        bar = "█" * min(count, 30)
        console.print(f"  {topic:15} {bar} {count}")

    # Activity by hour
    console.print("\n[bold]Activity by hour:[/bold]")
    max_hour = max(hours) if hours else 1
    for h in range(24):
        bar_len = int(hours[h] / max_hour * 20) if max_hour > 0 else 0
        bar = "█" * bar_len
        console.print(f"  {h:02d}:00 {bar}")


def cmd_record(args):
    """Manually record a thought"""
    evt = choir.record(
        topic=args.topic,
        content=args.text,
        kind=args.kind or args.topic,
        tags=args.tags.split(",") if args.tags else []
    )
    console.print(f"[green]Recorded thought #{evt.id}[/green]")


def cmd_clear(args):
    """Clear all thoughts (with confirmation)"""
    if not args.yes:
        confirm = input("Clear all thoughts? This cannot be undone. [y/N] ")
        if confirm.lower() != 'y':
            console.print("[yellow]Cancelled[/yellow]")
            return

    choir.clear(clear_disk=True)
    console.print("[green]All thoughts cleared[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Nightside - Introspection into your thought patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # log
    p_log = subparsers.add_parser("log", help="Show recent thoughts")
    p_log.add_argument("-n", "--limit", type=int, default=20, help="Number to show")
    p_log.set_defaults(func=cmd_log)

    # themes
    p_themes = subparsers.add_parser("themes", help="Show extracted themes")
    p_themes.add_argument("-n", "--limit", type=int, default=10, help="Number to show")
    p_themes.add_argument("--half-life", type=float, default=7.0, help="Decay half-life in days")
    p_themes.set_defaults(func=cmd_themes)

    # search
    p_search = subparsers.add_parser("search", help="Search thoughts")
    p_search.add_argument("query", help="Search pattern")
    p_search.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    p_search.set_defaults(func=cmd_search)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show statistics")
    p_stats.set_defaults(func=cmd_stats)

    # record
    p_record = subparsers.add_parser("record", help="Manually record a thought")
    p_record.add_argument("topic", help="Topic/category")
    p_record.add_argument("text", help="Thought content")
    p_record.add_argument("-k", "--kind", help="Kind (defaults to topic)")
    p_record.add_argument("-t", "--tags", help="Comma-separated tags")
    p_record.set_defaults(func=cmd_record)

    # clear
    p_clear = subparsers.add_parser("clear", help="Clear all thoughts")
    p_clear.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_clear.set_defaults(func=cmd_clear)

    args = parser.parse_args()

    if not args.command:
        # Default to themes
        args.limit = 10
        args.half_life = 7.0
        cmd_themes(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
