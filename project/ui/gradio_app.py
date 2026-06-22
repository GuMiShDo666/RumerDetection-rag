import gradio as gr
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

def create_gradio_ui():
    rag_system = RAGSystem()
    rag_system.initialize()
    
    doc_manager = DocumentManager(rag_system)
    chat_interface = ChatInterface(rag_system)
    
    def format_database_summary():
        return doc_manager.get_database_summary()

    def build_database_handler(progress=gr.Progress()):
        result, parent_count, child_count = doc_manager.build_rumor_database(
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        gr.Info(f"Indexed {result['rows']} cases into {parent_count} parent chunks and {child_count} child chunks")
        return format_database_summary()
    
    def clear_handler():
        try:
            doc_manager.clear_all()
            gr.Info("🗑️ Removed all documents")
        except Exception as exc:
            gr.Error(f"Unable to clear documents: {exc}")
        return format_database_summary()
    
    def chat_handler(msg, hist):
        for chunk in chat_interface.chat(msg, hist):
            yield chunk
    
    def clear_chat_handler():
        chat_interface.clear_session()
    
    with gr.Blocks(title="RumorDetection RAG") as demo:
        
        with gr.Tab("Rumor Database", elem_id="doc-management-tab"):
            gr.Markdown("## RumorDetection RAG Database")
            gr.Markdown("Build a retrieval database from labeled Chinese rumor-detection cases. Label `1` means rumor, and label `0` means non-rumor.")

            build_btn = gr.Button("Build / Rebuild Rumor RAG Database", variant="primary", size="md")

            gr.Markdown("## Database Status")
            database_summary = gr.Textbox(
                value=format_database_summary(),
                interactive=False,
                lines=8,
                max_lines=12,
                elem_id="file-list-box",
                show_label=False
            )
            
            with gr.Row():
                refresh_btn = gr.Button("Refresh", size="md")
                clear_btn = gr.Button("Clear All", variant="stop", size="md")
            
            build_btn.click(build_database_handler, None, database_summary, show_progress="corner")
            refresh_btn.click(format_database_summary, None, database_summary)
            clear_btn.click(clear_handler, None, database_summary)
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=720, 
                placeholder="<strong>Enter a claim or upload an image for rumor detection.</strong><br><em>Images are parsed with OCR and BLIP before retrieval.</em>",
                show_label=False,
                avatar_images=(None, os.path.join(ASSETS_DIR, "chatbot_avatar.png")),
                layout="bubble"
            )
            chatbot.clear(clear_chat_handler)

            chat_input = gr.MultimodalTextbox(
                file_types=["image"],
                file_count="single",
                placeholder="Type a claim or upload an image...",
                show_label=False,
                sources=["upload"],
            )
            
            gr.ChatInterface(
                fn=chat_handler,
                chatbot=chatbot,
                textbox=chat_input,
                multimodal=True,
            )
    
    return demo
