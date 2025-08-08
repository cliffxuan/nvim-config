#!/usr/bin/env python3
"""
Script to convert sample shirt data and load it into the flow-actions API.
This handles both Daier and DASK data formats.
"""

from pathlib import Path

import typer
import urllib3
from rich.console import Console

from app.primitives import PLATFORM
from app.quota_monitors.utils import get_mixtures

PWD = Path(__file__).absolute().parent

urllib3.disable_warnings()

console = Console()


def generate_mixture_type_alias():
    """Generate type aliases (SITE, PLATFORM, MIXTURE) and write to primitives.py."""
    console.print("[bold blue]Generating type aliases...[/bold blue]")

    all_mixtures = []
    platforms: list[PLATFORM] = ["daier", "dusk"]

    for platform in platforms:
        try:
            mixtures = get_mixtures(platform)
            all_mixtures.extend(mixtures)
            console.print(
                f"[blue]Found {len(mixtures)} {platform.title()} mixtures[/blue]"
            )
        except Exception as e:
            console.print(f"[red]Error fetching {platform.title()} mixtures: {e}[/red]")

    if not all_mixtures:
        console.print("[red]No mixtures found, cannot generate type alias[/red]")
        return

    # Sort mixtures for consistent output
    all_mixtures.sort()

    # Path to primitives.py
    primitives_file = Path(__file__).parent.parent / "primitives.py"

    # Generate the content for primitives.py
    mixture_literals = []
    for i, mixture in enumerate(all_mixtures):
        if i == 0:
            mixture_literals.append(f'    "{mixture}",')
        else:
            mixture_literals.append(f'    "{mixture}",')

    content = f"""from typing import Literal

type SITE = Literal["cuda", "rocm"]
type PLATFORM = Literal["daier", "dusk"]
type MIXTURE = Literal[
{chr(10).join(mixture_literals)}
]
"""

    console.print(
        f"[green]Generated type aliases with {len(all_mixtures)} mixtures[/green]"
    )

    try:
        # Write the complete file content
        with open(primitives_file, "w") as f:
            f.write(content)

        console.print(
            f"[green]✓ Updated {primitives_file} with all type aliases[/green]"
        )
        console.print('[dim]SITE: TypeAlias = Literal["cuda", "rocm"][/dim]')
        console.print('[dim]PLATFORM: TypeAlias = Literal["daier", "dusk"][/dim]')
        console.print(
            f"[dim]MIXTURE: TypeAlias with {len(all_mixtures)} mixtures[/dim]"
        )

    except Exception as e:
        console.print(f"[red]Error writing to primitives.py: {e}[/red]")


app = typer.Typer(help="Flow Actions Data Processing Tool")


@app.command()
def generate_types():
    """Generate type aliases based on available mixtures and write to primitives.py."""
    generate_mixture_type_alias()


if __name__ == "__main__":
    app()
