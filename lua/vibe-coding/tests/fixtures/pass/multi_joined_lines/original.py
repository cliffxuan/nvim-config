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