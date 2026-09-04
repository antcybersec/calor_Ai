"""
CalorAI Interactive Terminal CLI.
Supports plain text meal logging, /image <path> <caption> vision analysis,
running totals visualization, memory view, and built-in evaluation commands.
"""

import sys
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from agent import run_agent_turn
from database import get_daily_totals, get_memories, get_meal_history, clear_user_data

console = Console()

def print_welcome_banner():
    console.print(Panel.fit(
        "[bold green]CalorAI Conversational Meal Logging Agent[/bold green]\n"
        "[dim]WhatsApp-style calorie & macro tracking with persistent memory & vision[/dim]\n\n"
        "• Type meal descriptions directly (e.g. [italic]'had 2 rotis and dal for lunch'[/italic])\n"
        "• Send photo: [bold]/image path/to/plate.jpg optional caption[/bold]\n"
        "• Commands: [bold]/totals[/bold], [bold]/history[/bold], [bold]/memories[/bold], [bold]/clear[/bold], [bold]/eval[/bold], [bold]/exit[/bold]",
        title="Welcome to CalorAI",
        border_style="green"
    ))

def display_daily_totals(user_id: str):
    totals = get_daily_totals(user_id)
    table = Table(title=f"Daily Nutrition Summary ({totals['date']})", border_style="cyan")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="bold white")

    table.add_row("Total Calories", f"{totals['total_calories']} kcal")
    table.add_row("Protein", f"{totals['total_protein_g']} g")
    table.add_row("Carbohydrates", f"{totals['total_carbs_g']} g")
    table.add_row("Fats", f"{totals['total_fat_g']} g")
    table.add_row("Meals Logged Today", f"{totals['meal_count']}")

    console.print(table)

    if totals["meals"]:
        m_table = Table(title="Logged Meals Today", border_style="dim white")
        m_table.add_column("ID", style="dim")
        m_table.add_column("Type", style="cyan")
        m_table.add_column("Items", style="white")
        m_table.add_column("Calories", style="yellow")
        m_table.add_column("Protein", style="magenta")

        for m in totals["meals"]:
            item_names = ", ".join([it.get("name", "item") for it in m["items"]])
            m_table.add_row(
                str(m["id"]),
                m["meal_type"].title(),
                item_names,
                f"{m['total_calories']} kcal",
                f"{m['total_protein_g']} g"
            )
        console.print(m_table)

def display_memories(user_id: str):
    mems = get_memories(user_id)
    if not mems:
        console.print("[yellow]No persistent memories stored yet for this user.[/yellow]")
        return

    table = Table(title="Persistent User Profile & Memories", border_style="magenta")
    table.add_column("Category", style="bold cyan")
    table.add_column("Memory Key", style="bold yellow")
    table.add_column("Stored Fact / Value", style="white")
    table.add_column("Last Updated", style="dim")

    for m in mems:
        table.add_row(
            m["category"].upper(),
            m["memory_key"].replace("_", " ").title(),
            m["memory_value"],
            str(m["updated_at"])
        )
    console.print(table)

def main():
    print_welcome_banner()
    user_id = "test_user_1"
    session_id = "cli_session"

    while True:
        try:
            user_input = console.input("\n[bold green]You > [/bold green]").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                console.print("[bold red]Exiting CalorAI CLI. Goodbye![/bold red]")
                break

            elif user_input.lower() == "/totals":
                display_daily_totals(user_id)
                continue

            elif user_input.lower() == "/memories":
                display_memories(user_id)
                continue

            elif user_input.lower() == "/history":
                history = get_meal_history(user_id, limit=5)
                console.print(f"[bold cyan]Recent Meal History ({len(history)} entries):[/bold cyan]")
                for h in history:
                    console.print(f" • [yellow]{h['date']}[/yellow] - {h['raw_input']} ({h['total_calories']} kcal, {h['total_protein_g']}g protein)")
                continue

            elif user_input.lower() == "/clear":
                clear_user_data(user_id)
                console.print(f"[bold red]Cleared all database records for user '{user_id}'.[/bold red]")
                continue

            elif user_input.lower() == "/eval":
                console.print("[bold cyan]Launching automated evals runner...[/bold cyan]")
                os.system("python evals.py")
                continue

            # Parse /image path caption command
            image_path: Optional[str] = None
            text_prompt = user_input

            if user_input.startswith("/image "):
                parts = user_input[7:].split(" ", 1)
                image_path = parts[0].strip()
                text_prompt = parts[1].strip() if len(parts) > 1 else ""

                if not os.path.exists(image_path):
                    console.print(f"[bold red]Error: Image file not found at '{image_path}'[/bold red]")
                    continue
                else:
                    console.print(f"[dim]Routing photo '{image_path}' to Vision Model...[/dim]")

            # Run Agent Turn
            with console.status("[bold green]CalorAI is thinking...[/bold green]", spinner="dots"):
                result = run_agent_turn(
                    user_id=user_id,
                    message_text=text_prompt,
                    image_path=image_path,
                    session_id=session_id
                )

            # Display Response
            response_md = Markdown(result["response"])
            console.print(Panel(response_md, title=f"CalorAI ({result['latency_seconds']}s)", border_style="blue"))

            # Display quick status bar
            totals = result["daily_totals"]
            console.print(
                f"[dim]Today's Running Totals: {totals['total_calories']} kcal | "
                f"Protein: {totals['total_protein_g']}g | Carbs: {totals['total_carbs_g']}g | Fat: {totals['total_fat_g']}g[/dim]"
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Session ended.[/bold red]")
            break

if __name__ == "__main__":
    main()
