import gradio as gr
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tamil-llama"

# System prompt for Tamil grammar and writing assistant
SYSTEM_PROMPT = """You are a Tamil grammar and writing assistant. Your task is to check Tamil input sentences for errors and correct them. You should identify and fix all grammatical, spelling, punctuation, and syntactic mistakes without changing the original meaning or tone. For example, correct verb tense, agreement, case-suffixes, gender forms, word order, and diacritics as needed. Always preserve the input's script and style. If the input is already grammatically correct, return it verbatim (unchanged). Do NOT hallucinate or add information not present in the input; focus only on grammar. Output only the corrected Tamil sentence (no quotes or extra explanation).

Tone: Polite, precise, and helpful.
Constraints: No translation (stay in Tamil), no creative rewriting. If the input contains harmful or disallowed content, refuse safely (e.g. say "மன்னிக்கவும், உதவி செய்ய முடியவில்லை.").

Example: User: "நான் நேற்று பள்ளிக்கு போகிறேன்." -> Assistant: "நான் நேற்று பள்ளிக்குப் போனேன்."""


def correct_tamil_text(user_text):
    """Send text to Ollama for Tamil text correction."""
    if not user_text.strip():
        return "தயவுசெய்து தமிழ் உரையை உள்ளிடவும்."

    prompt = f"திருத்தப்பட்ட உரை:\n{user_text}"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "பதில் கிடைக்கவில்லை.")
    except requests.ConnectionError:
        return "Ollama இயங்கவில்லை. Ollama-ஐ முதலில் தொடங்கவும்:\n\n```\nollama serve\n```"
    except requests.Timeout:
        return "காலாவதியானது. மீண்டும் முயற்சிக்கவும்."
    except Exception as e:
        return f"பிழை: {str(e)}"


# Build Gradio UI
with gr.Blocks(title="Tamil Text Corrector") as demo:
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1>தமிழ் உரை பிழை திருத்தம்</h1>
            <p>Tamil Text Error Correction</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="தவறான உரை (Input Text with Errors)",
                placeholder="இங்கே தமிழ் உரையை உள்ளிடவும்...",
                lines=6,
            )
            correct_btn = gr.Button("பிழையை திருத்து (Correct)", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="சரியான உரை (Corrected Text)",
                lines=6,
                interactive=False,
            )

    with gr.Accordion("எடுத்துக்காட்டுகள் (Examples)", open=False):
        gr.Examples(
            examples=[
                ["நான் நேற்று பள்ளிக்கு போகிறேன்."],
                ["நான் தமிழ் பேசுகிறது."],
                ["அவள் நல்ல மாணவன்."],
                ["எனக்கு ஒரு புத்தகம் இருக்கிறான்."],
            ],
            inputs=input_text,
            outputs=output_text,
            fn=correct_tamil_text,
            cache_examples=False,
        )

    gr.Markdown(
        """
        ---
        **Instructions:**
        1. Make sure Ollama is running: `ollama serve`
        2. Type or paste Tamil text with errors and click **Correct**
        """
    )

    correct_btn.click(fn=correct_tamil_text, inputs=input_text, outputs=output_text)
    input_text.submit(fn=correct_tamil_text, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
