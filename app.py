import gradio as gr
from simple_rag import RAGOrchestrator, DataExtractor
import os

# Initialize RAG system
rag = RAGOrchestrator()
collection_name = rag.setup_vector_store()

# Get supported file extensions
data_extractor = DataExtractor()
supported_extensions = list(data_extractor.processors.keys())

def process_uploaded_files(files):
    """Process uploaded files and add them to the vector store"""
    if not files:
        return "No files uploaded."
    
    # Limit the number of files to 5
    if len(files) > 5:
        return "Error: Maximum 5 files can be processed at once."
    
    status_messages = []
    for file in files:
        try:
            success = rag.process_file_if_needed(file.name, collection_name)
            if success:
                status_messages.append(f"✓ Successfully processed: {os.path.basename(file.name)}")
            else:
                status_messages.append(f"✗ Failed to process: {os.path.basename(file.name)}")
        except Exception as e:
            status_messages.append(f"✗ Error processing {os.path.basename(file.name)}: {str(e)}")
    
    return "\n".join(status_messages)

def process_message(message: str, history: list) -> list:
    """Process incoming messages and return response"""
    try:
        result = rag.query(message, collection_name)
        sources = "\n\nSources:\n" + "\n".join(
            f"- {source['content'][:200]}..."
            for source in result['sources']
        )
        response = f"{result['answer']}{sources}"
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
    except Exception as e:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Error: {str(e)}"}
        ]

def clear_chat_history():
    rag.clear_chat_history()
    return []

# Create Gradio interface with updated components
with gr.Blocks() as demo:
    gr.Markdown("# Document RAG Chatbot")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### Supported file types: {', '.join(supported_extensions)}")
            file_output = gr.Textbox(label="Upload Status", lines=5)
            files = gr.File(
                label="Upload Documents (max 5 files)",
                file_count="multiple",  # Changed from 5 to "multiple"
                file_types=supported_extensions,
                type="filepath"
            )
            upload_btn = gr.Button("Process Uploaded Files")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox(label="Message", placeholder="Type your question here...")
            clear_btn = gr.Button("Clear Chat History")
    
    upload_btn.click(
        fn=process_uploaded_files,
        inputs=[files],
        outputs=[file_output]
    )
    
    msg.submit(
        fn=process_message, 
        inputs=[msg, chatbot], 
        outputs=[chatbot]
    ).then(
        fn=lambda: "", 
        outputs=msg
    )
    
    clear_btn.click(
        fn=clear_chat_history,
        outputs=chatbot
    )

if __name__ == "__main__":
    demo.launch(share=True)