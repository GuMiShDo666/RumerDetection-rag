import json
import re
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from core.execution_logger import log_chat_end, log_chat_start, log_error
from core.image_claim_extractor import ImageClaimExtractor, is_supported_image

SYSTEM_NODES = {"summarize_history", "rewrite_query"}
FINAL_RESPONSE_NODES = {"aggregate_answers"}

SYSTEM_NODE_CONFIG = {
    "rewrite_query":     {"title": "🔍 Query Analysis & Rewriting"},
    "summarize_history": {"title": "📋 Chat History Summary"},
}

# --- Helpers ---

def make_message(content, *, title=None, node=None):
    msg = {"role": "assistant", "content": content}
    if title or node:
        msg["metadata"] = {k: v for k, v in {"title": title, "node": node}.items() if v}
    return msg


def find_msg_idx(messages, node):
    return next(
        (i for i, m in enumerate(messages) if m.get("metadata", {}).get("node") == node),
        None,
    )


def parse_rewrite_json(buffer):
    match = re.search(r"\{.*\}", buffer, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def format_rewrite_content(buffer):
    data = parse_rewrite_json(buffer)
    if not data:
        return "⏳ Analyzing query..."
    if data.get("is_clear"):
        lines = ["✅ **Query is clear**"]
        if data.get("questions"):
            lines += ["\n**Rewritten queries:**"] + [f"- {q}" for q in data["questions"]]
    else:
        lines = ["❓ **Query is unclear**"]
        clarification = data.get("clarification_needed", "")
        if clarification and clarification.strip().lower() != "no":
            lines.append(f"\nClarification needed: *{clarification}*")
    return "\n".join(lines)


def compact_json(value, max_chars=180):
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")


def add_trace_event(response_messages, trace_events, trace_keys, key, event):
    if key in trace_keys:
        return
    trace_keys.add(key)
    trace_events.append(event)
    content = "\n".join(f"{index}. {item}" for index, item in enumerate(trace_events, start=1))
    idx = find_msg_idx(response_messages, "agent_trace")
    if idx is None:
        response_messages.append(make_message(content, title="🧭 Agent Trace", node="agent_trace"))
    else:
        response_messages[idx]["content"] = content

# --- End of Helpers ---

class ChatInterface:

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.image_extractor = ImageClaimExtractor()

    def _path_from_file_value(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("path") or value.get("name")
        return getattr(value, "path", None) or getattr(value, "name", None)

    def _normalize_message(self, message):
        if isinstance(message, dict):
            user_text = str(message.get("text") or "").strip()
            files = message.get("files") or []
        else:
            user_text = str(message or "").strip()
            files = []

        image_paths = []
        for file_value in files:
            path = self._path_from_file_value(file_value)
            if path and is_supported_image(path):
                image_paths.append(path)

        if not user_text and not image_paths:
            raise ValueError("请输入文本，或上传一张包含待检测信息的图片。")

        image_results = [self.image_extractor.extract(path) for path in image_paths]
        if not image_results:
            return user_text, []

        prompt_parts = []
        if user_text:
            prompt_parts.extend(["用户补充文本:", user_text, ""])
        prompt_parts.append("请判断下列图片中呈现的主张是否为谣言。优先使用 OCR 识别文本；如果 OCR 不完整，再参考 BLIP 图片说明。")
        prompt_parts.extend(
            self.image_extractor.to_prompt_section(result, index)
            for index, result in enumerate(image_results, start=1)
        )

        image_summaries = [
            self.image_extractor.to_summary_markdown(result, index)
            for index, result in enumerate(image_results, start=1)
        ]
        return "\n\n".join(prompt_parts).strip(), image_summaries

    def _handle_system_node(self, chunk, node, response_messages, system_node_buffer, trace_events, trace_keys):
        """Update (or create) the collapsible system-node message and surface any clarification."""
        system_node_buffer[node] = system_node_buffer.get(node, "") + chunk.content
        buffer = system_node_buffer[node]
        title  = SYSTEM_NODE_CONFIG[node]["title"]
        content = format_rewrite_content(buffer) if node == "rewrite_query" else buffer

        idx = find_msg_idx(response_messages, node)
        if idx is None:
            response_messages.append(make_message(content, title=title, node=node))
        else:
            response_messages[idx]["content"] = content

        if node == "rewrite_query":
            self._surface_clarification(buffer, response_messages)
            data = parse_rewrite_json(buffer)
            if data:
                if data.get("is_clear") and data.get("questions"):
                    rewritten = "; ".join(data["questions"])
                    add_trace_event(
                        response_messages,
                        trace_events,
                        trace_keys,
                        f"rewrite:{rewritten}",
                        f"Rewritten query: {rewritten}",
                    )
                elif data.get("clarification_needed"):
                    add_trace_event(
                        response_messages,
                        trace_events,
                        trace_keys,
                        f"clarification:{data['clarification_needed']}",
                        f"Clarification requested: {data['clarification_needed']}",
                    )

    def _surface_clarification(self, buffer, response_messages):
        """If the query is unclear, add/update a plain clarification message."""
        data          = parse_rewrite_json(buffer) or {}
        clarification = data.get("clarification_needed", "")
        if not data.get("is_clear") and clarification.strip().lower() not in ("", "no"):
            cidx = find_msg_idx(response_messages, "clarification")
            if cidx is None:
                response_messages.append(make_message(clarification, node="clarification"))
            else:
                response_messages[cidx]["content"] = clarification

    def _handle_tool_call(self, chunk, response_messages, active_tool_calls, trace_events, trace_keys):
        """Register new tool calls as collapsible messages."""
        for tc in chunk.tool_calls:
            if tc.get("id") and tc["id"] not in active_tool_calls:
                response_messages.append(
                    make_message(f"Running `{tc['name']}`...", title=f"🛠️ {tc['name']}")
                )
                active_tool_calls[tc["id"]] = len(response_messages) - 1
                add_trace_event(
                    response_messages,
                    trace_events,
                    trace_keys,
                    f"tool_call:{tc['id']}",
                    f"Tool call: `{tc['name']}` with `{compact_json(tc.get('args', {}))}`",
                )

    def _handle_tool_result(self, chunk, response_messages, active_tool_calls, trace_events, trace_keys):
        """Fill in the tool result inside the matching collapsible message."""
        idx = active_tool_calls.get(chunk.tool_call_id)
        if idx is not None:
            preview = str(chunk.content)[:300]
            suffix  = "\n..." if len(str(chunk.content)) > 300 else ""
            response_messages[idx]["content"] = f"```\n{preview}{suffix}\n```"
            add_trace_event(
                response_messages,
                trace_events,
                trace_keys,
                f"tool_result:{chunk.tool_call_id}",
                f"Tool result: `{getattr(chunk, 'name', 'tool')}` returned {len(str(chunk.content))} characters",
            )

    def _handle_llm_token(self, chunk, node, response_messages):
        """Append streaming LLM tokens to the last plain assistant message."""
        last = response_messages[-1] if response_messages else None
        if not (last and last.get("role") == "assistant" and "metadata" not in last):
            response_messages.append(make_message(""))
        response_messages[-1]["content"] += chunk.content

    def chat(self, message, history):
        """Generator that streams Gradio chat message dicts."""
        if not self.rag_system.agent_graph:
            yield "⚠️ System not initialized!"
            return

        try:
            normalized_message, image_summaries = self._normalize_message(message)
        except Exception as e:
            log_error("image_message_preprocessing", e)
            yield f"❌ Error: {str(e)}"
            return

        config        = self.rag_system.get_config()
        current_state = self.rag_system.agent_graph.get_state(config)
        log_chat_start(normalized_message, self.rag_system.thread_id, bool(current_state.next))

        try:
            if current_state.next:
                self.rag_system.agent_graph.update_state(config, {"messages": [HumanMessage(content=normalized_message)]})
                stream_input = None
            else:
                stream_input = {"messages": [HumanMessage(content=normalized_message)]}

            response_messages  = []
            active_tool_calls  = {}
            system_node_buffer = {}
            trace_events       = []
            trace_keys         = set()
            if image_summaries:
                response_messages.append(
                    make_message(
                        "\n\n---\n\n".join(image_summaries),
                        title="🖼️ Image Understanding",
                        node="image_understanding",
                    )
                )
                yield response_messages
            add_trace_event(
                response_messages,
                trace_events,
                trace_keys,
                "original_query",
                f"Original query: {normalized_message}",
            )

            for chunk, metadata in self.rag_system.agent_graph.stream(stream_input, config=config, stream_mode="messages"):
                node = metadata.get("langgraph_node", "")

                if node in SYSTEM_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                    self._handle_system_node(chunk, node, response_messages, system_node_buffer, trace_events, trace_keys)

                elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    self._handle_tool_call(chunk, response_messages, active_tool_calls, trace_events, trace_keys)

                elif isinstance(chunk, ToolMessage):
                    self._handle_tool_result(chunk, response_messages, active_tool_calls, trace_events, trace_keys)

                elif isinstance(chunk, AIMessageChunk) and chunk.content and node in FINAL_RESPONSE_NODES:
                    self._handle_llm_token(chunk, node, response_messages)

                else:
                    continue

                yield response_messages

            final_state = self.rag_system.agent_graph.get_state(config)
            log_chat_end(getattr(final_state, "values", final_state))
            add_trace_event(
                response_messages,
                trace_events,
                trace_keys,
                "final_answer",
                "Final answer streamed to the user.",
            )
            yield response_messages

        except Exception as e:
            log_error("chat", e)
            yield f"❌ Error: {str(e)}"

    def clear_session(self):
        self.rag_system.reset_thread()
        self.rag_system.observability.flush()
