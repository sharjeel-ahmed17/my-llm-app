import streamlit as st
import torch
import pickle
import os
from model import TinyLLM

st.set_page_config(page_title="My Tiny LLM", page_icon="🤖", layout="centered")

st.title("🤖 My Tiny LLM")
st.caption("A GPT-style model trained from scratch on children's stories.")

@st.cache_resource
def load_model():
    if not os.path.exists("tinyllm.pt"):
        st.error("❌ tinyllm.pt not found!")
        st.stop()
    if not os.path.exists("vocab.pkl"):
        st.error("❌ vocab.pkl not found!")
        st.stop()

    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    model = TinyLLM(vocab["vocab_size"])
    model.load_state_dict(
        torch.load("tinyllm.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, vocab

with st.spinner("Loading model..."):
    model, vocab = load_model()

st.success("✅ Model loaded!")

stoi = vocab["stoi"]
itos = vocab["itos"]
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

def generate(prompt, max_tokens, temperature):
    encoded = encode(prompt)
    if len(encoded) == 0:
        return "⚠️ Prompt has unknown characters. Try: 'Once upon a time'"
    context = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)
    return decode(output[0].tolist())

st.divider()
prompt      = st.text_area("✏️ Enter your prompt:", value="Once upon a time", height=80)
col1, col2  = st.columns(2)
with col1:
    max_tokens  = st.slider("📏 Max tokens", 50, 300, 150, step=10)
with col2:
    temperature = st.slider("🌡️ Temperature", 0.1, 1.5, 0.8, step=0.1)

if st.button("🚀 Generate Text", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Generating..."):
            result = generate(prompt, max_tokens, temperature)
        st.divider()
        st.subheader("📝 Output:")
        st.write(result)
        st.download_button("⬇️ Download Output", result, file_name="output.txt")
