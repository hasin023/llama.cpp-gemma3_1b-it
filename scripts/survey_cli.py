"""
Survey CLI - Interactive Chat Tool for LLM Survey Agent
=========================================================
A Typer-based CLI for turn-by-turn conversations with liama.cpp server.

Usage:
    python survey_cli.py list                    # List available surveys
    python survey_cli.py chat frozen-berry       # Chat with a pre-defined survey
    python survey_cli.py custom                  # Create your own survey

Requirements:
    - typer
    - openai
    - python-dotenv
"""

import sys
import os
import time

# Fix Bengali/Unicode display in Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# ============================================================================
# TEMPLATES & CONSTANTS
# ============================================================================
INTRO_TEMPLATES = [
    "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি {name} বলছেন?",
]

POSITIVE_ENDINGS = [
    "ধন্যবাদ, আমাকে তথ্য দিয়ে সহযোগিতা করার জন্য, ভাল থাকবেন",
]

NEGATIVE_ENDINGS = [
    "দুঃখিত। ধন্যবাদ, ভাল থাকবেন।",
]

# ============================================================================
# CONFIGURATION
# ============================================================================
app = typer.Typer(
    name="survey-cli",
    help="Interactive CLI for LLM-powered Bangla surveys.",
    add_completion=False,
)
console = Console()

# Server configuration
ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1")  ## Change this to your LLM endpoint
API_KEY = os.getenv("OPENAI_API_KEY", "sk-no-key-required")
MODEL_NAME = os.getenv("LLM_MODEL", "gemma3_1b_400S_p77s16v20-Q4_K_M")


# ============================================================================
# PRE-DEFINED BENGALI SURVEY TEMPLATES
# ============================================================================
SURVEY_TEMPLATES = {
    "frozen-berry": {
        "name": "FrozenBerry পণ্য জরিপ",
        "description": "FrozenBerry ফ্রোজেন ফ্রুট পণ্যের গ্রাহক প্রতিক্রিয়া জরিপ",
        "context": "বাংলাদেশে FrozenBerry কিনতে পারবেন Foodpanda, Daraz, Unimart, Shawpno এর মতো অনলাইন শপ ও সুপারশপ থেকে। আমাদের পণ্য ১০০% প্রাকৃতিক এবং কোনো প্রিজার্ভেটিভ নেই।",
        "questions": [
            "আপনি কি FrozenBerry ব্যবহার করেছেন?",
            "এটার স্বাদ নিয়ে কি কোনো feedback দিতে চান?",
            "সংরক্ষণ করা সহজ কি?",
            "দাম কি যুক্তিসঙ্গত মনে হয়েছে?",
            "আপনি কি আবার কিনবেন?",
        ],
    },
    "student-life": {
        "name": "ছাত্রজীবন প্রতিফলন জরিপ",
        "description": "শিক্ষার্থীদের জীবন অভিজ্ঞতা এবং মতামত সংগ্রহ",
        "context": "ছাত্রজীবন মানে হচ্ছে, যখন আমরা জীবিত থাকি, শেখার জন্য উন্মুক্ত থাকি এবং প্রতিদিন নতুন কিছু আবিষ্কার করি।",
        "questions": [
            "তুমি যদি হঠাৎ একদিন অদৃশ্য হয়ে যেতে পারো, প্রথমে কী করতে চাইবে?",
            "জীবনের কোন ছোট অভ্যাসটা তোমার মতে সবচেয়ে বেশি পরিবর্তন আনতে পারে?",
            "যদি আবার ছাত্রজীবনে ফিরে যাওয়ার সুযোগ পাও, কোন সিদ্ধান্তটা বদলাতে চাইতে?",
        ],
    },
    "healthcare": {
        "name": "স্বাস্থ্যসেবা সন্তুষ্টি জরিপ",
        "description": "হাসপাতাল ও ক্লিনিক সেবার মান যাচাই",
        "context": "আমরা বাংলাদেশের স্বাস্থ্যসেবা খাতের উন্নয়নে কাজ করছি। আপনার মতামত আমাদের সেবার মান বাড়াতে সাহায্য করবে। সব তথ্য গোপনীয় রাখা হবে।",
        "questions": [
            "আপনি সম্প্রতি কোন হাসপাতাল বা ক্লিনিকে গিয়েছিলেন?",
            "ডাক্তারের সাথে কথা বলার সময় কি যথেষ্ট পেয়েছিলেন?",
            "ওষুধের দাম কি সাশ্রয়ী মনে হয়েছে?",
            "সেবার মান নিয়ে আপনার সামগ্রিক অভিজ্ঞতা কেমন ছিল?",
            "আপনি কি এই হাসপাতাল অন্যদের সুপারিশ করবেন?",
        ],
    },
    "ecommerce": {
        "name": "ই-কমার্স কেনাকাটা অভিজ্ঞতা",
        "description": "অনলাইন শপিং অভিজ্ঞতা এবং সন্তুষ্টি পরিমাপ",
        "context": "বাংলাদেশে অনলাইন কেনাকাটা দ্রুত বাড়ছে। Daraz, Evaly, Chaldal, Foodpanda সহ অনেক প্ল্যাটফর্ম রয়েছে। আমরা গ্রাহকদের অভিজ্ঞতা জানতে চাই।",
        "questions": [
            "আপনি সাধারণত কোন অনলাইন প্ল্যাটফর্ম থেকে কেনাকাটা করেন?",
            "ডেলিভারি সময় কি সন্তোষজনক ছিল?",
            "পণ্যের গুণমান কি ছবির সাথে মিলেছে?",
            "পেমেন্ট পদ্ধতি কি সুবিধাজনক ছিল?",
            "কোন সমস্যায় পড়লে কাস্টমার সাপোর্ট কি সাহায্য করেছে?",
        ],
    },
    "banking": {
        "name": "ব্যাংকিং সেবা জরিপ",
        "description": "ব্যাংক ও মোবাইল ব্যাংকিং সেবার মান যাচাই",
        "context": "বাংলাদেশে bKash, Nagad, Rocket সহ মোবাইল ব্যাংকিং জনপ্রিয়। এছাড়াও প্রচলিত ব্যাংকগুলোর অনলাইন সেবাও বাড়ছে। আমরা আপনার অভিজ্ঞতা জানতে চাই।",
        "questions": [
            "আপনি কোন ব্যাংক বা মোবাইল ব্যাংকিং সেবা ব্যবহার করেন?",
            "লেনদেন করতে কি কোনো সমস্যা হয়?",
            "অ্যাপ বা ওয়েবসাইট ব্যবহার করা কি সহজ?",
            "কাস্টমার সার্ভিস কি দ্রুত সাড়া দেয়?",
            "নিরাপত্তা নিয়ে কি কোনো উদ্বেগ আছে?",
        ],
    },
    "food-delivery": {
        "name": "ফুড ডেলিভারি সার্ভিস জরিপ",
        "description": "খাবার ডেলিভারি সেবার সন্তুষ্টি পরিমাপ",
        "context": "Foodpanda, Pathao Food, HungryNaki বাংলাদেশের জনপ্রিয় ফুড ডেলিভারি সার্ভিস। আমরা গ্রাহকদের অভিজ্ঞতা সংগ্রহ করছি সেবার মান উন্নত করতে।",
        "questions": [
            "আপনি কোন ফুড ডেলিভারি অ্যাপ বেশি ব্যবহার করেন?",
            "খাবারের মান কি সন্তোষজনক ছিল?",
            "ডেলিভারি সময় কি ঠিক ছিল?",
            "রাইডারের ব্যবহার কেমন ছিল?",
            "অ্যাপে অর্ডার করা কি সহজ?",
        ],
    },
}


# ============================================================================
# PROMPT BUILDER
# ============================================================================
def build_prompt(context: str, questions: list[str], conversation_turns: list[dict]) -> str:
    """Build Gemma-formatted prompt with conversation history and strict instructions."""

    prompt = f"""<start_of_turn>user
You are a polite Bangla phone survey agent. ALWAYS respond in Bangla.
ONLY ask the questions from the Question List one by one, exactly as written, in order.
IF user queries about anything, respond from the Survey Context, then continue with the next question.

Survey Context:
{context}

Question List:
""" + "\n".join(["• " + q for q in questions]) + """
<end_of_turn>
"""

    # Add conversation turns
    for turn in conversation_turns:
        role = turn["role"]
        content = turn["content"]
        prompt += f"<start_of_turn>{role}\n{content}\n<end_of_turn>\n"

    # End with model turn to generate next response
    prompt += "<start_of_turn>model\n"

    return prompt


# ============================================================================
# INFERENCE CLIENT
# ============================================================================
def run_inference(context: str, questions: list[str], conversation_turns: list[dict]) -> str:
    """Send inference request to llama.cpp server."""
    client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)
    prompt_text = build_prompt(context, questions, conversation_turns)

    # Calculate dynamic max tokens based on expected output length
    # Endings are short, questions are short. 
    response = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt_text,
        max_tokens=256, # Reduced as we don't need long generations
        stream=False,
        temperature=0.1, # Lower temperature for adherence to instructions
        stop=["<end_of_turn>"] # Ensure model stops
    )

    return response.choices[0].text.strip()


# ============================================================================
# CLI COMMANDS
# ============================================================================
@app.command(name="list")
def list_surveys():
    """List all available pre-defined survey templates."""
    table = Table(title="Available Survey Templates", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="green", width=15)
    table.add_column("Name", style="yellow")
    table.add_column("Description", style="white")

    for survey_id, survey in SURVEY_TEMPLATES.items():
        table.add_row(survey_id, survey["name"], survey["description"])

    console.print(table)
    console.print("\n[dim]Usage: python survey_cli.py chat <ID>[/dim]")


@app.command()
def chat(
    survey_id: str = typer.Argument(..., help="Survey template ID (use 'list' to see options)"),
    name: str = typer.Option("মিঃ/মিসেস", "--name", "-n", help="Name of the customer to call")
):
    """Start an interactive chat session with a pre-defined survey."""
    if survey_id not in SURVEY_TEMPLATES:
        console.print(f"[red]Error:[/red] Survey '{survey_id}' not found.")
        console.print("[dim]Use 'python survey_cli.py list' to see available surveys.[/dim]")
        raise typer.Exit(1)

    survey = SURVEY_TEMPLATES[survey_id]
    _run_chat_loop(survey["context"], survey["questions"], survey["name"], customer_name=name)


@app.command()
def custom(
    name: str = typer.Option("মিঃ/মিসেস", "--name", "-n", help="Name of the customer to call")
):
    """Create and run a custom survey with your own context and questions."""
    console.print(Panel("[bold]Custom Survey Builder[/bold]\nCreate your own survey by providing context and questions."))

    # Get context
    console.print("\n[cyan]Enter Survey Context[/cyan] (background info for the LLM):")
    context = Prompt.ask("[dim]Context[/dim]")

    if not context.strip():
        console.print("[red]Error:[/red] Context cannot be empty.")
        raise typer.Exit(1)

    # Get questions
    console.print("\n[cyan]Enter Questions[/cyan] (one per line, empty line to finish):")
    questions = []
    while True:
        q = Prompt.ask(f"[dim]Q{len(questions) + 1}[/dim]", default="")
        if not q.strip():
            break
        questions.append(q.strip())

    if not questions:
        console.print("[red]Error:[/red] At least one question is required.")
        raise typer.Exit(1)

    _run_chat_loop(context, questions, "Custom Survey", customer_name=name)


def _run_chat_loop(context: str, questions: list[str], survey_name: str, customer_name: str):
    """Main chat loop for interactive conversation."""
    console.print(Panel(f"[bold green]Starting: {survey_name}[/bold green]\n[dim]Type 'exit' or 'quit' to force stop.[/dim]"))

    conversation_turns: list[dict] = []
    
    # 1. FORCE INTRO (No inference)
    # Select template and format with name
    import random
    intro_template = random.choice(INTRO_TEMPLATES)
    intro_text = intro_template.replace("{name}", customer_name)
    
    # Add to history and display
    conversation_turns.append({"role": "model", "content": intro_text})
    console.print(f"\n[bold cyan]Agent:[/bold cyan] {intro_text}\n")

    # Main conversation loop
    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")

            # Check for exit commands
            if user_input.lower().strip() in ["exit", "quit", "bye", "বিদায়"]:
                console.print("\n[yellow]Conversation ended by user.[/yellow]")
                break

            if not user_input.strip():
                continue

            # Add user turn to history
            conversation_turns.append({"role": "user", "content": user_input})

            # Get LLM response
            with console.status("[cyan]LLM is thinking...[/cyan]", spinner="dots"):
                response = run_inference(context, questions, conversation_turns)

            # Add model response to history
            conversation_turns.append({"role": "model", "content": response})
            console.print(f"\n[bold cyan]Agent:[/bold cyan] {response}\n")
            
            # CHECK FOR EXIT CONDITION (Fixed Endings)
            clean_response = response.strip()
            # Check for partial match or exact match depending on strictness
            # Since LLM might add small variations or spaces, strict check involves cleaning
            is_positive = any(end.strip() in clean_response for end in POSITIVE_ENDINGS)
            is_negative = any(end.strip() in clean_response for end in NEGATIVE_ENDINGS)
            
            if is_positive:
                console.print(Panel("[bold green]Survey Completed Successfully![/bold green]", border_style="green"))
                break
            
            if is_negative:
                console.print(Panel("[bold yellow]Survey Terminated (User Refused/Negative)[/bold yellow]", border_style="yellow"))
                break

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            console.print("[dim]Please check if the LLM server is running.[/dim]")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    app()
