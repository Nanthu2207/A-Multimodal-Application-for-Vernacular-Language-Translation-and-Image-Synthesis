
import torch
import gradio as gr
from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer
import requests
from PIL import Image
import io
import  os

# Model and device configuration for Whisper
MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
device = 0 if torch.cuda.is_available() else "cpu"

# Initialize Whisper pipeline
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

# Define transcription function
def transcribe(inputs, task):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    result = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)
    return result["text"]


API_URL = "https://api-inference.huggingface.co/models/XLabs-AI/flux-RealismLora"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

# Define image generation function
def generate_image(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes
    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))
    return image

# Initialize GPT-Neo model and tokenizer
text_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
text_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Define text generation function
def generate_text(prompt, temperature=0.9, max_length=100):
    inputs = text_tokenizer(prompt, return_tensors="pt")
    gen_tokens = text_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=text_tokenizer.eos_token_id,
    )
    gen_text = text_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    return gen_text

# Gradio app with multiple functionalities in tabs
with gr.Blocks() as demo:
    with gr.Tab("Microphone Transcription"):
        gr.Markdown("### Whisper Large V3: Microphone Transcription")
        gr.Markdown(
            "Transcribe long-form microphone inputs with the click of a button! This demo uses OpenAI Whisper to process microphone inputs."
        )
        mic_input = gr.Audio(sources="microphone", type="filepath", label="Microphone Input")
        mic_task = gr.Radio(["transcribe", "translate"], label="Task", value="transcribe")
        mic_output = gr.Textbox(label="Transcribed Text")
        gr.Button("Submit").click(transcribe, inputs=[mic_input, mic_task], outputs=mic_output)

    with gr.Tab("File Upload Transcription"):
        gr.Markdown("### Whisper Large V3: File Transcription")
        gr.Markdown(
            "Transcribe long-form audio files with the click of a button! This demo uses OpenAI Whisper to process uploaded audio files."
        )
        file_input = gr.Audio(sources="upload", type="filepath", label="Upload Audio File")
        file_task = gr.Radio(["transcribe", "translate"], label="Task", value="transcribe")
        file_output = gr.Textbox(label="Transcribed Text")
        gr.Button("Submit").click(transcribe, inputs=[file_input, file_task], outputs=file_output)

    with gr.Tab("Image Generation"):
        gr.Markdown("### Image Generation with Flux-RealismLora")
        gr.Markdown(
            "Generate images from text prompts using the Flux-RealismLora model on Hugging Face."
        )
        img_prompt = gr.Textbox(lines=2, placeholder="Enter your image generation prompt here...")
        img_output = gr.Image(label="Generated Image")
        gr.Button("Generate").click(generate_image, inputs=img_prompt, outputs=img_output)

    with gr.Tab("Text Generation"):
        gr.Markdown("### GPT-Neo Text Generator")
        gr.Markdown(
            "Generate text using GPT-Neo. Adjust the temperature for creativity and max length for output size."
        )
        txt_prompt = gr.Textbox(label="Enter your prompt:", placeholder="Start typing...")
        txt_temperature = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="Temperature")
        txt_max_length = gr.Slider(10, 200, value=100, step=10, label="Max Length")
        txt_output = gr.Textbox(label="Generated Text")
        gr.Button("Generate").click(
            generate_text, inputs=[txt_prompt, txt_temperature, txt_max_length], outputs=txt_output
        )

# Launch the app
demo.launch()