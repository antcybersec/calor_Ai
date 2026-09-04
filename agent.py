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

def get_llm(bind_tools: bool = True):
    """Return the primary conversation LLM with optional tool binding."""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if gemini_key and not gemini_key.startswith("your_"):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=0.3,
        )
        return llm.bind_tools(ALL_TOOLS) if bind_tools else llm

    if openai_key and not openai_key.startswith("your_") and openai_key != "mock_key":
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.3)
        return llm.bind_tools(ALL_TOOLS) if bind_tools else llm

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
    Core LLM agent node. Builds full context prompt and calls Gemini with tools.
    Surfaces vision uncertainty to user instead of silently logging low-confidence meals.
    """
    llm = get_llm(bind_tools=True)
    if not llm:
        # No API key configured at all
        return {"messages": [AIMessage(content="Please configure a GEMINI_API_KEY or OPENAI_API_KEY in your .env file to use CalorAI.")]}

    user_id = state.get("user_id", "default_user")
    memories_str = state.get("user_memories_summary", "")
    totals_str = state.get("daily_totals_summary", "")
    vision_data = state.get("vision_analysis")
    ref_data = state.get("resolved_reference")

    system_content = SYSTEM_PROMPT.format(
        memories_context=f"\n{memories_str}" if memories_str else "",
        totals_context=f"\n{totals_str}" if totals_str else "",
    )

    injected_context_parts = []

    # Inject vision model output (separate model's results handed off here)
    if vision_data:
        confidence = vision_data.get("confidence", 1.0)
        ambiguity = vision_data.get("ambiguity_notes", "")

        if confidence < 0.4:
            # Low confidence — surface to user instead of guessing
            injected_context_parts.append(
                f"[VISION MODEL RESULT — LOW CONFIDENCE: {confidence}]\n"
                f"The photo was unclear. Description: {vision_data.get('description', 'unknown')}\n"
                f"Ambiguity: {ambiguity}\n"
                "INSTRUCTION: Tell the user you couldn't clearly identify the food in the photo and ask them to describe what they ate instead. Do NOT log a meal."
            )
        elif confidence < 0.7:
            # Medium confidence — log but be transparent
            items_str = json.dumps(vision_data.get("items", []))
            injected_context_parts.append(
                f"[VISION MODEL RESULT — MODERATE CONFIDENCE: {confidence}]\n"
                f"Description: {vision_data.get('description')}\n"
                f"Caption adjustment: {vision_data.get('caption_applied', 'none')}\n"
                f"Detected items: {items_str}\n"
                f"Estimated totals: {vision_data.get('total_calories')} kcal | {vision_data.get('total_protein_g')}g protein\n"
                "INSTRUCTION: Log this meal using log_meal_tool, but tell the user what you estimated and that they can correct if wrong."
            )
        else:
            # High confidence — log normally
            items_str = json.dumps(vision_data.get("items", []))
            injected_context_parts.append(
                f"[VISION MODEL RESULT — HIGH CONFIDENCE: {confidence}]\n"
                f"Description: {vision_data.get('description')}\n"
                f"Caption adjustment: {vision_data.get('caption_applied', 'none')}\n"
                f"Detected items: {items_str}\n"
                f"Total: {vision_data.get('total_calories')} kcal | {vision_data.get('total_protein_g')}g protein\n"
                f"INSTRUCTION: Log this as a single meal using log_meal_tool for user_id '{user_id}'. Do NOT ask for confirmation."
            )

    # Inject resolved meal reference
    if ref_data:
        if ref_data.get("total_calories") is None:
            # Needs estimation — let LLM handle
            injected_context_parts.append(
                f"[RESOLVED MEAL REFERENCE]\n"
                f"User's '{ref_data.get('description')}' is their saved usual meal but no nutritional history found.\n"
                f"INSTRUCTION: Estimate calories/macros for '{ref_data.get('description')}' from your knowledge and log it with log_meal_tool."
            )
        else:
            items_str = json.dumps(ref_data.get("items", []))
            injected_context_parts.append(
                f"[RESOLVED MEAL REFERENCE — source: {ref_data.get('source')}]\n"
                f"Items: {items_str}\n"
                f"Total: {ref_data.get('total_calories')} kcal | {ref_data.get('total_protein_g')}g protein\n"
                f"INSTRUCTION: Log this resolved meal using log_meal_tool for user_id '{user_id}'."
            )

    full_messages = [SystemMessage(content=system_content)]
    if injected_context_parts:
        full_messages.append(SystemMessage(content="\n\n---\n".join(injected_context_parts)))
    full_messages.extend(state["messages"])

    try:
        response = llm.invoke(full_messages)
        return {"messages": [response]}
    except Exception as e:
        error_str = str(e)
        # Rate limit — give a clear message rather than silently falling back
        if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
            return {"messages": [AIMessage(content="I'm at my API rate limit for this minute — please try again in 30 seconds.")]}
        return {"messages": [AIMessage(content=f"Something went wrong — please try again. (Error: {error_str[:80]})")]}


def execute_tools_node(state: AgentState) -> Dict[str, Any]:
    """Execute tool calls requested by the LLM in the previous step."""
    user_id = state.get("user_id", "default_user")
    last_message = state["messages"][-1]
    tool_name_map = {t.name: t for t in ALL_TOOLS}
    tool_outputs = []

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for call in last_message.tool_calls:
            t_name = call["name"]
            t_args = call["args"]
            t_id = call["id"]

            # Always inject the correct user_id — prevents cross-user data leaks
            t_args["user_id"] = user_id

            target_tool = tool_name_map.get(t_name)
            if target_tool:
                try:
                    result_str = target_tool.invoke(t_args)
                except Exception as ex:
                    result_str = json.dumps({"status": "error", "message": str(ex)})
            else:
                result_str = json.dumps({"status": "error", "message": f"Unknown tool: {t_name}"})

            tool_outputs.append(ToolMessage(content=result_str, tool_call_id=t_id))

    return {"messages": tool_outputs}


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


# ── Routing ───────────────────────────────────────────────────────────────────

def should_execute_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tools"
    return "memory_extractor"


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("load_context", load_context_node)
    builder.add_node("process_input", process_input_node)
    builder.add_node("agent", agent_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_node("memory_extractor", memory_extractor_node)

    builder.set_entry_point("load_context")
    builder.add_edge("load_context", "process_input")
    builder.add_edge("process_input", "agent")
    builder.add_conditional_edges(
        "agent",
        should_execute_tools,
        {"execute_tools": "execute_tools", "memory_extractor": "memory_extractor"}
    )
    builder.add_edge("execute_tools", "agent")
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
