"""
CalorAI LangGraph Conversational Agent Engine.
- Dual-model routing: Gemini 2.5 Flash for text/tools, Gemini 2.5 Flash for vision
- Selective memory injection (no conversation history stuffing)
- Correction-aware meal editing (context-based, not just "last meal")
- Confidence-based vision ambiguity surfacing
- LangSmith tracing via environment variable
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from database import (
    init_db,
    get_daily_totals,
    save_chat_message,
    log_meal,
    correct_last_meal,
    get_meal_history,
    get_memories,
)
from memory_manager import (
    format_memories_for_prompt,
    extract_and_save_memories,
    resolve_meal_reference
)
from vision import analyze_meal_image
from tools import ALL_TOOLS
from gemini_runner import run_gemini_agent

load_dotenv()

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    session_id: str
    image_path: Optional[str]
    image_caption: Optional[str]
    user_memories_summary: str
    daily_totals_summary: str
    vision_analysis: Optional[Dict[str, Any]]
    resolved_reference: Optional[Dict[str, Any]]
    final_output: str

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CalorAI — a friendly WhatsApp-native meal logging assistant.

CORE BEHAVIOUR:
• Log meals from plain-text descriptions or plate photos. Be fast — never ask unnecessary questions.
• For vague portions (e.g. "some rice", "a bit of dal") make a sensible estimate, log it, and tell the user what you assumed so they can correct if needed.
• When a user corrects a meal (e.g. "actually that was 3 rotis not 2"), call correct_meal_tool to REPLACE the previous entry — never double-count.
• When asked about daily totals or macros, call get_daily_totals_tool and give a friendly summary.
• When the user states a persistent fact ("i'm vegetarian", "my usual is X", "targeting 140g protein"), call save_memory_tool immediately.

TONE: Warm, concise, like texting a knowledgeable friend. No bullet-point lists in responses unless showing macros.

{memories_context}
{totals_context}"""

# ── LLM Factory ───────────────────────────────────────────────────────────────

def _get_api_key() -> Optional[str]:
    """Return a configured API key or None."""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and not gemini_key.startswith("your_"):
        return gemini_key
    return None

# ── Graph Nodes ────────────────────────────────────────────────────────────────

def load_context_node(state: AgentState) -> Dict[str, Any]:
    """Load user profile memory + current daily totals before every agent turn."""
    user_id = state.get("user_id", "default_user")
    init_db()

    memories_str = format_memories_for_prompt(user_id)
    totals = get_daily_totals(user_id)

    totals_str = (
        f"[TODAY {totals['date']}] {totals['meal_count']} meals logged — "
        f"{totals['total_calories']} kcal | {totals['total_protein_g']}g protein | "
        f"{totals['total_carbs_g']}g carbs | {totals['total_fat_g']}g fat"
    )

    return {
        "user_memories_summary": memories_str,
        "daily_totals_summary": totals_str,
    }


def process_input_node(state: AgentState) -> Dict[str, Any]:
    """
    Pre-process step: run vision model on image inputs, resolve
    history references ('same as yesterday', 'my usual') before the LLM sees the message.
    """
    user_id = state.get("user_id", "default_user")
    image_path = state.get("image_path")
    caption = state.get("image_caption")
    memories_str = state.get("user_memories_summary", "")

    vision_analysis = None
    resolved_ref = None

    if image_path and os.path.exists(image_path):
        vision_analysis = analyze_meal_image(
            image_path=image_path,
            caption=caption,
            user_memories_summary=memories_str,
        )

    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], HumanMessage):
        text_content = str(messages[-1].content)
        resolved_ref = resolve_meal_reference(user_id, text_content)

    return {"vision_analysis": vision_analysis, "resolved_reference": resolved_ref}


def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Core agent node. Uses native google-genai SDK to run the full
    tool-calling loop in one shot. Thought signatures are preserved
    internally by the SDK — no langchain serialization stripping.
    """
    api_key = _get_api_key()
    if not api_key:
        return {"messages": [AIMessage(content="Please set GEMINI_API_KEY in your .env file.")]}

    model = os.getenv("DEFAULT_TEXT_MODEL", "gemini-3.5-flash-lite")
    user_id = state.get("user_id", "default_user")
    memories_str = state.get("user_memories_summary", "")
    totals_str = state.get("daily_totals_summary", "")
    vision_data = state.get("vision_analysis")
    ref_data = state.get("resolved_reference")

    mem_ctx = f"\n{memories_str}" if memories_str else ""
    tot_ctx = f"\n{totals_str}" if totals_str else ""
    system_content = (
        f"{SYSTEM_PROMPT.format(memories_context=mem_ctx, totals_context=tot_ctx)}\n\n"
        f"CURRENT USER ID: '{user_id}'. You must pass user_id='{user_id}' in every tool call."
    )

    # Build injected context (vision output, resolved references)
    injected_parts = []

    if vision_data:
        confidence = vision_data.get("confidence", 1.0)
        ambiguity = vision_data.get("ambiguity_notes", "")
        if confidence < 0.4:
            injected_parts.append(
                f"[VISION — LOW CONFIDENCE: {confidence}]\n"
                f"Could not clearly identify the food. Description: {vision_data.get('description', 'unknown')}\n"
                f"Ambiguity: {ambiguity}\n"
                "INSTRUCTION: Ask user to describe what they ate instead. Do NOT log."
            )
        else:
            items_str = json.dumps(vision_data.get("items", []))
            qualifier = "HIGH" if confidence >= 0.7 else "MODERATE"
            injected_parts.append(
                f"[VISION — {qualifier} CONFIDENCE: {confidence}]\n"
                f"Description: {vision_data.get('description')}\n"
                f"Items: {items_str}\n"
                f"Totals: {vision_data.get('total_calories')} kcal | {vision_data.get('total_protein_g')}g protein\n"
                f"INSTRUCTION: Log using log_meal_tool for user_id '{user_id}'." +
                ("\nTell user what was estimated and they can correct." if confidence < 0.7 else "")
            )

    if ref_data:
        if ref_data.get("total_calories") is None:
            injected_parts.append(
                f"[RESOLVED REFERENCE]\nUser's usual meal is '{ref_data.get('description')}' but no nutrition history found.\n"
                "INSTRUCTION: Estimate and log using log_meal_tool."
            )
        else:
            items_str = json.dumps(ref_data.get("items", []))
            injected_parts.append(
                f"[RESOLVED REFERENCE — {ref_data.get('source')}]\n"
                f"Items: {items_str}\n"
                f"Totals: {ref_data.get('total_calories')} kcal | {ref_data.get('total_protein_g')}g protein\n"
                f"INSTRUCTION: Log using log_meal_tool for user_id '{user_id}'."
            )

    injected_context = "\n\n---\n".join(injected_parts)

    # Get last user message
    messages = state.get("messages", [])
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content)
            break
    if not user_text:
        user_text = "Hi"

    try:
        response_text = run_gemini_agent(
            api_key=api_key,
            model=model,
            system_prompt=system_content,
            user_message=user_text,
            lc_tools=ALL_TOOLS,
            injected_context=injected_context,
            user_id=user_id,
        )
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            response_text = "I'm at my API rate limit — please try again in 30 seconds."
        else:
            response_text = f"Something went wrong: {err[:120]}"

    return {"messages": [AIMessage(content=response_text)]}


def memory_extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    Post-turn async memory extraction.
    Scans the user's message for persistent facts and stores them in the DB.
    This runs AFTER the agent responds so it doesn't add latency to the turn.
    """
    user_id = state.get("user_id", "default_user")
    messages = state.get("messages", [])

    human_text = ""
    ai_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not human_text:
            human_text = str(msg.content)
        elif isinstance(msg, AIMessage) and not ai_text:
            ai_text = str(msg.content)

    if human_text:
        extract_and_save_memories(user_id, human_text, ai_text)

    # Find final AI response for output
    final_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = str(msg.content)
            break

    return {"final_output": final_text}


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("load_context", load_context_node)
    builder.add_node("process_input", process_input_node)
    builder.add_node("agent", agent_node)
    builder.add_node("memory_extractor", memory_extractor_node)

    builder.set_entry_point("load_context")
    builder.add_edge("load_context", "process_input")
    builder.add_edge("process_input", "agent")
    builder.add_edge("agent", "memory_extractor")
    builder.add_edge("memory_extractor", END)
    return builder.compile()


agent_app = build_agent_graph()


# ── Public Entrypoint ─────────────────────────────────────────────────────────

def run_agent_turn(
    user_id: str,
    message_text: str,
    image_path: Optional[str] = None,
    session_id: str = "default_session",
) -> Dict[str, Any]:
    """
    Process a single conversational turn.
    Returns response text, latency, and updated daily totals.
    """
    start = time.time()

    input_content = message_text.strip() or "[Meal photo sent]"
    save_chat_message(
        user_id=user_id, session_id=session_id,
        role="user", content=input_content, image_path=image_path
    )

    initial_state: AgentState = {
        "messages": [HumanMessage(content=input_content)],
        "user_id": user_id,
        "session_id": session_id,
        "image_path": image_path,
        "image_caption": message_text if image_path else None,
        "user_memories_summary": "",
        "daily_totals_summary": "",
        "vision_analysis": None,
        "resolved_reference": None,
        "final_output": "",
    }

    response_text = ""
    try:
        final_state = agent_app.invoke(initial_state)
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                response_text = str(msg.content)
                break
        if not response_text:
            response_text = final_state.get("final_output", "Got it!")
    except Exception as e:
        response_text = "Sorry, something went wrong. Please try again."
        print(f"[Agent Error] {e}")

    elapsed = round(time.time() - start, 3)
    save_chat_message(user_id=user_id, session_id=session_id, role="assistant", content=response_text)

    return {
        "user_id": user_id,
        "session_id": session_id,
        "response": response_text,
        "latency_seconds": elapsed,
        "daily_totals": get_daily_totals(user_id),
        "is_image_turn": bool(image_path),
    }
