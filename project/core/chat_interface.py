import json
import re
import config
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from core.execution_logger import log_chat_end, log_chat_start, log_error
from core.image_claim_extractor import ImageClaimExtractor, is_supported_image

SYSTEM_NODES = {"summarize_history", "rewrite_query"}
FINAL_RESPONSE_NODES = {"aggregate_answers"}

SYSTEM_NODE_CONFIG = {
    "rewrite_query":     {"title": "查询分析与改写"},
    "summarize_history": {"title": "对话历史摘要"},
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
        return "正在分析问题..."
    if data.get("is_clear"):
        lines = ["**问题已明确**"]
        if data.get("questions"):
            lines += ["\n**改写后的检索问题：**"] + [f"- {q}" for q in data["questions"]]
    else:
        lines = ["**问题需要补充信息**"]
        clarification = data.get("clarification_needed", "")
        if clarification and clarification.strip().lower() != "no":
            lines.append(f"\n需要补充：*{clarification}*")
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
        response_messages.append(make_message(content, title="Agent 执行轨迹", node="agent_trace"))
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

    def _fallback_query(self, normalized_message):
        text = normalized_message
        if "OCR 识别文本:" in text:
            text = text.split("OCR 识别文本:", 1)[1]
        text = re.sub(r"BLIP 图片说明:.*", "", text, flags=re.DOTALL)
        text = re.sub(r"图片 \d+:.*", "", text)
        text = re.sub(r"用户补充文本:", "", text)
        text = re.sub(r"请判断.*", "", text)
        return re.sub(r"\s+", " ", text).strip() or normalized_message

    @staticmethod
    def _field_from_content(content, field):
        match = re.search(rf"^{field}:\s*(.+)$", content, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _title_from_metadata(metadata):
        candidates = [
            metadata.get("H2"),
            metadata.get("Header 2"),
            metadata.get("h2"),
        ]
        for value in candidates:
            if not value:
                continue
            for part in reversed(str(value).split(" -> ")):
                part = part.strip()
                match = re.search(r"Article\s+[^:]+:\s*(.+)", part)
                if match:
                    return match.group(1).strip()
                if part and part.lower() != "articles":
                    return part
        return ""

    @staticmethod
    def _title_from_claim_text(content):
        compact = re.sub(r"\s+", " ", content)
        for pattern in (
            r"【([^】]{4,90})】",
            r"认为[“\"]([^”\"]{4,90})[”\"]",
            r"(喝汤[^，。；;]{0,60}[？?])",
        ):
            match = re.search(pattern, compact)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _clip_text(text, limit=420):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text[:limit].rstrip() + ("..." if len(text) > limit else "")

    def _content_excerpt(self, content, limit=520):
        content = re.sub(r"^.*?### Content", "", content, flags=re.DOTALL)
        content = re.sub(r"## Article\s+[^#]+$", "", content, flags=re.DOTALL)
        return self._clip_text(content, limit=limit)

    def _retrieve_fallback_evidence(self, query):
        collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
        docs = collection.similarity_search(
            query,
            k=max(3, config.DEFAULT_RETRIEVAL_K),
            score_threshold=max(0.1, config.RETRIEVAL_SCORE_THRESHOLD - 0.15),
        )

        evidence = []
        seen_signatures = set()
        for doc in docs:
            content = doc.page_content
            metadata = dict(doc.metadata)
            signature = self._clip_text(content, 160)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            evidence.append(
                {
                    "title": (
                        self._title_from_claim_text(content)
                        or self._field_from_content(content, "Title")
                        or self._title_from_metadata(metadata)
                        or "相关资料"
                    ),
                    "source": self._field_from_content(content, "Source") or metadata.get("source", "rumor_database.csv"),
                    "date": self._field_from_content(content, "Date") or "N/A",
                    "url": self._field_from_content(content, "URL") or "",
                    "content": self._content_excerpt(content),
                }
            )
            if len(evidence) >= 4:
                break
        return evidence

    def _verdict_from_evidence(self, evidence):
        if not evidence:
            return "判定：证据不足"

        joined = "\n".join(f"{item['title']}\n{item['content']}" for item in evidence)
        rumor_markers = ("谣言", "辟谣", "流言", "不实", "假的", "并非如此", "并不是", "并不代表", "误区", "别信")
        non_rumor_markers = ("是真的", "属实", "确有", "可以", "建议", "专家表示")
        rumor_hits = sum(1 for marker in rumor_markers if marker in joined)
        non_rumor_hits = sum(1 for marker in non_rumor_markers if marker in joined)

        if rumor_hits >= 2 and rumor_hits >= non_rumor_hits:
            return "判定：谣言"
        if non_rumor_hits >= 3 and rumor_hits == 0:
            return "判定：非谣言"
        return "判定：证据不足"

    def _fallback_response(self, normalized_message, original_error):
        query = self._fallback_query(normalized_message)
        try:
            evidence = self._retrieve_fallback_evidence(query)
        except Exception as retrieval_error:
            log_error("fallback_retrieval", retrieval_error)
            return None

        reason = str(original_error)
        forced_fallback = "快速检索兜底模式" in reason
        status_text = (
            "服务器已启用快速检索兜底模式，系统直接使用知识库相似证据生成保守结果。"
            if forced_fallback
            else f"原始 Agent 生成失败，已自动切换检索兜底。错误摘要：{self._clip_text(reason, 220)}"
        )
        answer_note = (
            "当前服务器使用快速检索兜底模式；下面结论只基于知识库相似资料，换回可用生成模型后会恢复完整 Agent 回答。"
            if forced_fallback
            else "当前生成模型不可用，系统已自动切换为检索证据模式；下面结论只基于知识库相似资料，建议以完整 Agent 回答为准。"
        )

        verdict = self._verdict_from_evidence(evidence)
        if evidence:
            bullets = [
                f"- 《{item['title']}》（{item['source']}，{item['date']}）：{item['content']}"
                for item in evidence
            ]
            sources = sorted({item["source"] for item in evidence if item.get("source")})
            answer = "\n".join(
                [
                    verdict,
                    answer_note,
                    "",
                    *bullets,
                    "",
                    "Sources:",
                    *[f"- {source}" for source in sources],
                ]
            )
        else:
            answer = "\n".join(
                [
                    "判定：证据不足",
                    "当前生成模型不可用，且知识库没有检索到足够接近的参考资料。",
                ]
            )

        return [
            make_message(
                status_text,
                title="运行状态",
                node="fallback_status",
            ),
            make_message(
                "\n\n".join(
                    f"证据 {index}\n标题：{item['title']}\n来源：{item['source']}\n日期：{item['date']}\n摘要：{item['content']}"
                    for index, item in enumerate(evidence, start=1)
                )
                or "未找到可用证据。",
                title="检索证据",
                node="fallback_evidence",
            ),
            make_message(answer),
        ]

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
                        f"改写检索问题：{rewritten}",
                    )
                elif data.get("clarification_needed"):
                    add_trace_event(
                        response_messages,
                        trace_events,
                        trace_keys,
                        f"clarification:{data['clarification_needed']}",
                        f"请求补充信息：{data['clarification_needed']}",
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
                    make_message(f"正在运行 `{tc['name']}`...", title=f"工具调用：{tc['name']}")
                )
                active_tool_calls[tc["id"]] = len(response_messages) - 1
                add_trace_event(
                    response_messages,
                    trace_events,
                    trace_keys,
                    f"tool_call:{tc['id']}",
                    f"调用工具：`{tc['name']}`，参数：`{compact_json(tc.get('args', {}))}`",
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
                f"工具结果：`{getattr(chunk, 'name', 'tool')}` 返回 {len(str(chunk.content))} 个字符",
            )

    def _handle_llm_token(self, chunk, node, response_messages):
        """Append streaming LLM tokens to the last plain assistant message."""
        last = response_messages[-1] if response_messages else None
        if not (last and last.get("role") == "assistant" and "metadata" not in last):
            response_messages.append(make_message(""))
        response_messages[-1]["content"] += chunk.content

    def chat(self, message, history):
        """Generator that streams chat message dicts."""
        if not self.rag_system.agent_graph:
            yield "系统尚未初始化。"
            return

        try:
            normalized_message, image_summaries = self._normalize_message(message)
        except Exception as e:
            log_error("image_message_preprocessing", e)
            yield f"错误：{str(e)}"
            return

        if config.RAG_FORCE_RETRIEVAL_FALLBACK:
            fallback = self._fallback_response(normalized_message, "服务器已启用快速检索兜底模式。")
            yield fallback if fallback else "错误：检索兜底模式不可用。"
            return

        runtime_config = self.rag_system.get_config()
        current_state = self.rag_system.agent_graph.get_state(runtime_config)
        log_chat_start(normalized_message, self.rag_system.thread_id, bool(current_state.next))

        try:
            if current_state.next:
                self.rag_system.agent_graph.update_state(runtime_config, {"messages": [HumanMessage(content=normalized_message)]})
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
                        title="图片解析结果",
                        node="image_understanding",
                    )
                )
                yield response_messages
            add_trace_event(
                response_messages,
                trace_events,
                trace_keys,
                "original_query",
                f"原始问题：{normalized_message}",
            )

            for chunk, metadata in self.rag_system.agent_graph.stream(stream_input, config=runtime_config, stream_mode="messages"):
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

            final_state = self.rag_system.agent_graph.get_state(runtime_config)
            log_chat_end(getattr(final_state, "values", final_state))
            add_trace_event(
                response_messages,
                trace_events,
                trace_keys,
                "final_answer",
                "已生成最终回答。",
            )
            yield response_messages

        except Exception as e:
            log_error("chat", e)
            fallback = self._fallback_response(normalized_message, e)
            yield fallback if fallback else f"错误：{str(e)}"

    def clear_session(self):
        self.rag_system.reset_thread()
        self.rag_system.observability.flush()
