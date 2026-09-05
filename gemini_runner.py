"""
Native Google GenAI SDK runner for tool-calling agent loops.

Replaces LangChain's Gemini integration for the core chat loop because
langchain-google-genai strips thought_signatures from function call history,
causing 400 errors on multi-turn tool calling with all current Gemini models.

The native SDK preserves thought_signatures by keeping the raw Content objects
in conversation history — no serialization/deserialization that drops fields.
"""

import os
import json
from typing import List, Dict, Any, Optional

import google.genai as genai
from google.genai import types


def _build_tool_config(lc_tools: list) -> Optional[types.Tool]:
    """Convert LangChain @tool functions to Gemini FunctionDeclarations."""
    if not lc_tools:
        return None

    declarations = []
    for lc_tool in lc_tools:
        # Extract JSON schema from the LangChain tool
        raw_schema = {}
        if hasattr(lc_tool, "args_schema") and lc_tool.args_schema:
            raw_schema = lc_tool.args_schema.model_json_schema()

        properties = {}
        required = []
        for field_name, field_info in raw_schema.get("properties", {}).items():
            prop = {"type": _json_type_to_gemini(field_info.get("type", "string"))}
            if "description" in field_info:
                prop["description"] = field_info["description"]
            properties[field_name] = prop

        required = raw_schema.get("required", [])

        declarations.append(
            types.FunctionDeclaration(
                name=lc_tool.name,
                description=lc_tool.description or "",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={k: types.Schema(**v) for k, v in properties.items()},
                    required=required,
                ),
            )
        )

    return types.Tool(function_declarations=declarations)


def _json_type_to_gemini(json_type: str) -> str:
    mapping = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }
    return mapping.get(json_type, "STRING")


def run_gemini_agent(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    lc_tools: list,
    injected_context: str = "",
    user_id: Optional[str] = None,
    max_tool_turns: int = 6,
) -> str:
    """
    Run a full agent loop (user message → tool calls → final response) using
    the native google-genai SDK. Thought signatures are preserved automatically
    because we pass the raw Content objects back, never serialising them.

    Returns the final text response string.
    """
    client = genai.Client(api_key=api_key)
    tool_map = {t.name: t for t in lc_tools}

    gemini_tool = _build_tool_config(lc_tools)

    full_system = system_prompt
    if injected_context:
        full_system = f"{system_prompt}\n\n---\n{injected_context}"

    config = types.GenerateContentConfig(
        system_instruction=full_system,
        tools=[gemini_tool] if gemini_tool else None,
        temperature=0.7,
    )

    # Model fallback list to protect against quota exhaustion (e.g. 429) or deprecation (404)
    candidate_models = [model]
    for fallback in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    last_error = None
    for current_model in candidate_models:
        try:
            # Conversation history — raw Gemini Content objects (thought_signatures intact)
            contents: List[types.Content] = [
                types.Content(role="user", parts=[types.Part(text=user_message)])
            ]

            for _turn in range(max_tool_turns):
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config,
                )

                candidate = response.candidates[0]
                assistant_content = candidate.content

                # Append raw assistant content (preserves thought_signatures!)
                contents.append(assistant_content)

                # Collect function calls from this turn
                fn_calls = [
                    part.function_call
                    for part in assistant_content.parts
                    if part.function_call is not None
                ]

                if not fn_calls:
                    # No tool calls → done, extract text
                    text_parts = [
                        part.text
                        for part in assistant_content.parts
                        if part.text
                    ]
                    return "\n".join(text_parts).strip() or "Done!"

                # Execute each tool and collect results
                fn_response_parts = []
                for fc in fn_calls:
                    lc_tool = tool_map.get(fc.name)
                    if lc_tool:
                        tool_args = dict(fc.args)
                        if user_id and ("user_id" not in tool_args or tool_args.get("user_id") in ("default_user", "", None)):
                            tool_args["user_id"] = user_id
                        try:
                            result = lc_tool.invoke(tool_args)
                        except Exception as ex:
                            result = json.dumps({"status": "error", "message": str(ex)})
                    else:
                        result = json.dumps({"status": "error", "message": f"Unknown tool: {fc.name}"})

                    fn_response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": result},
                            )
                        )
                    )

                # Add tool results as a user turn (native format, no stripping)
                contents.append(types.Content(role="user", parts=fn_response_parts))

            return "Got it! I've processed your update and adjusted your log."

        except Exception as e:
            err_str = str(e)
            # If 429 (quota) or 404 (model not found), try next fallback model
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str or "NOT_FOUND" in err_str:
                last_error = e
                continue
            raise e

    if last_error:
        raise last_error
    return "Got it! I've processed your update and adjusted your log."

