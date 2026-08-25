import gradio as gr
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tamil-llama"

# Improved system prompt with Tamil grammar rules
SYSTEM_PROMPT = """நீங்கள் ஒரு தமிழ் இலக்கண நிபுணர். பயனர் அனுப்பும் தமிழ் உரையில் உள்ள இலக்கணப் பிழைகளை மட்டும் சரிசெய்து, சரியான உரையை மட்டும் திருப்பி அளிக்கவும்.

முக்கிய தமிழ் இலக்கண விதிகள்:
1. கருத்துரு முரண்: "நான் ... கிறேன்" (முதல் நபர்), "அவன் ... கிறான்" (ஆண்), "அவள் ... கிறாள்" (பெண்), "அவர்கள் ... கிறார்கள்" (பன்மை)
2. கால முரண்: "நேற்று" என்றால் கடந்த காலம் பயன்படுத்த வேண்டும் (போனேன், சென்றான், வந்தாள்)
3. பால் முரண்: ஆண் நபருக்கு "அவன்/பையன்/மாணவன்", பெண் நபருக்கு "அவள்/பெண்/மாணவி"
4. உரிமை முரண்: "எனக்கு" பதில் "என்னிடம்" பயன்படுத்த வேண்டும் (பொருள் குறிப்பிடும் போது)
5. இணைப்பு முரண்: வன்மை முடிவு: "பள்ளிக்குப் போனேன்" (இணைப்பு "ப்" பயன்படுத்த வேண்டும்)

விதிகள்:
- சரியான உரையை மட்டும் திருப்பி அளிக்கவும்
- விளக்கம் வேண்டாம், திருத்தப்பட்ட வாக்கியம் மட்டும் போதும்
- உரையின் அர்த்தத்தை மாற்றாதீர்கள்"""


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
                    "temperature": 0.1,
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
