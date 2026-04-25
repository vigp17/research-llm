import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from model import ResearchLLM
from inference.generate import generate_text
from training.checkpointing import load_checkpoint

MODEL_CONFIGS = {
    "100M": dict(hidden_size=768, num_layers=8, num_heads=12, repo="Vigp17/researchllm-100m", ckpt="checkpoints_100m/ckpt_0100000.pt"),
    "300M": dict(hidden_size=1024, num_layers=16, num_heads=16, repo="Vigp17/researchllm-300m", ckpt="checkpoints/ckpt_0100000.pt"),
}

loaded_models = {}


def load_model(size: str):
    if size in loaded_models:
        return loaded_models[size]

    cfg = MODEL_CONFIGS[size]
    ckpt_path = hf_hub_download(repo_id=cfg["repo"], filename=cfg["ckpt"])
    tokenizer_path = hf_hub_download(repo_id=cfg["repo"], filename="tokenizer/researchllm_bpe.model")

    model = ResearchLLM(
        vocab_size=32000,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        max_seq_len=1024,
    )
    load_checkpoint(ckpt_path, model)
    model.eval()

    loaded_models[size] = (model, tokenizer_path)
    return model, tokenizer_path


def generate(prompt, model_size, max_new_tokens, temperature, top_k, top_p):
    if not prompt.strip():
        return "Please enter a prompt."
    model, tokenizer_path = load_model(model_size)
    output = generate_text(
        model,
        tokenizer_path,
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    return output


with gr.Blocks(title="ResearchLLM") as demo:
    gr.Markdown("# ResearchLLM\nA GPT-style language model trained from scratch on C4.")

    with gr.Row():
        model_size = gr.Radio(["100M", "300M"], value="300M", label="Model Size")

    prompt = gr.Textbox(label="Prompt", placeholder="Once upon a time...", lines=3)

    with gr.Row():
        max_new_tokens = gr.Slider(10, 300, value=100, step=10, label="Max New Tokens")
        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
        top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-P")

    output = gr.Textbox(label="Output", lines=8)
    btn = gr.Button("Generate", variant="primary")
    btn.click(generate, inputs=[prompt, model_size, max_new_tokens, temperature, top_k, top_p], outputs=output)

    gr.Examples(
        examples=[
            ["The transformer architecture", "300M", 100, 0.8, 50, 0.95],
            ["Once upon a time", "300M", 100, 0.9, 50, 0.95],
            ["In the field of machine learning", "300M", 100, 0.8, 50, 0.95],
            ["The scientific method", "300M", 100, 0.8, 50, 0.95],
        ],
        inputs=[prompt, model_size, max_new_tokens, temperature, top_k, top_p],
    )

if __name__ == "__main__":
    demo.launch()
