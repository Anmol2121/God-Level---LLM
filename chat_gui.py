from model import (Transformer, Embeddings, MultiHeadAttention, FeedForward, EncoderLayer, DecoderLayer,
                   LossWithLS, AdamWarmup, evaluate, tokenizer, device)

import torch
import streamlit as st
import torch.nn.functional as F


# Load model with caching
@st.cache_resource(show_spinner=True)
def load_model():
    checkpoint = torch.load(
        r"C:\Users\rawat\Downloads\checkpoint_13.pth.tar",
        map_location=device,weights_only=False,
    )
    return checkpoint['transformer']


# Top-p sampling
def top_p_sampling(logits, p=0.9, temperature=0.7):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_idx = (cumulative_probs > p).nonzero(as_tuple=True)[0][0].item()
    filtered_probs = sorted_probs.clone()
    filtered_probs[cutoff_idx + 1:] = 0
    filtered_probs /= filtered_probs.sum()
    sampled_idx = torch.multinomial(filtered_probs, 1).item()
    return sorted_indices[sampled_idx].item()


# Format special patterns like ** data collection **:
def format_special_phrases(text):
    import re
    return re.sub(r"\*\*\s*(.*?)\s*\*\*:", r"<h4 style='margin-top:10px; color:#00adb5;'>\1:</h4>", text)


# Streaming generator
def evaluate_stream(transformer, question, question_mask, max_len, tokenizer, top_p=0.9, temperature=0.7):
    transformer.eval()
    start_token = tokenizer.token_to_id("<start>")
    end_token = tokenizer.token_to_id("<end>")

    with torch.no_grad():
        encoded = transformer.encode(question, question_mask)
        words = torch.LongTensor([[start_token]]).to(device)
        generated = []

        for _ in range(max_len - 1):
            size = words.shape[1]
            target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).to(torch.uint8).to(device)
            target_mask = target_mask.unsqueeze(0).unsqueeze(0)

            decoded = transformer.decode(words, target_mask, encoded, question_mask)
            logits = transformer.logit(decoded[:, -1])
            next_word = top_p_sampling(logits.squeeze(0), p=top_p, temperature=temperature)

            if next_word == end_token:
                break

            words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim=1)
            word_str = tokenizer.decode([next_word])

            if word_str not in ["<start>", "<end>", "<pad>"]:
                generated.append(word_str + " ")
                yield "".join(generated)


# Main Streamlit App
def main():
    if "history" not in st.session_state:
        st.session_state.history = []

    transformer = load_model()

    st.markdown("""<style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212;
            color: #ffffff;
        }
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        .chat-container {
            max-height: 70vh;
            overflow-y: auto;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .message {
            display: flex;
            gap: 12px;
            max-width: 80%;
            padding: 14px 20px;
            margin-bottom: 18px;
            border-radius: 16px;
            font-size: 1rem;
            animation: fadeIn 0.3s ease-in;
        }
        .user-message {
            background: linear-gradient(135deg, #00ADB5, #00CED1);
            align-self: flex-end;
            margin-left: auto;
            box-shadow: 0 4px 15px rgba(0, 205, 209, 0.5);
        }
        .bot-message {
            background: #1f1f1f;
            align-self: flex-start;
            margin-right: auto;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
        .message::before {
            font-size: 1.2rem;
            margin-right: 6px;
        }
        .user-message::before { content: "🧑"; }
        .bot-message::before { content: "🤖"; }
        .typing-indicator {
            display: flex;
            gap: 6px;
            margin-bottom: 15px;
            margin-left: 10px;
        }
        .typing-dot {
            width: 10px;
            height: 10px;
            background: #00ADB5;
            border-radius: 50%;
            animation: bounce 1.5s infinite;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #00adb5, #0098a3);
            color: #ffffff;
            font-weight: 600;
            border-radius: 10px;
        }
    </style>""", unsafe_allow_html=True)

    st.markdown('<h1 style="text-align:center;">✨ Quantum Chat</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings ⚙️")
        temperature = st.slider("Creativity (Temperature)", 0.1, 1.0, 0.7, 0.05)
        top_p = st.slider("Response Focus (Top-p)", 0.5, 1.0, 0.9, 0.05)
        max_len = st.slider("Max Response Length", 50, 500, 150)

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.history:
            role = message["role"]
            content = message["content"]
            css_class = "user-message" if role == "user" else "bot-message"
            content = format_special_phrases(content)
            st.markdown(f'<div class="message {css_class}">{content}</div>', unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([8, 2])
        with col1:
            user_input = st.text_input("You:", placeholder="Type your message here...", label_visibility="collapsed")
        with col2:
            send_button = st.button("Send")

    if send_button and user_input.strip() != "":
        st.session_state.history.append({"role": "user", "content": user_input.strip()})
        encoding = tokenizer.encode(user_input.strip())
        question_tensor = torch.LongTensor(encoding.ids).to(device).unsqueeze(0)
        question_mask = (question_tensor != 0).to(device).unsqueeze(1).unsqueeze(1)

        with chat_container:
            typing_placeholder = st.empty()
            typing_placeholder.markdown(
                '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>',
                unsafe_allow_html=True
            )

        full_response = ""
        for partial_response in evaluate_stream(transformer, question_tensor, question_mask, max_len,
                                                tokenizer, top_p=top_p, temperature=temperature):
            full_response = partial_response
            formatted = format_special_phrases(full_response)
            with chat_container:
                typing_placeholder.markdown(
                    f'<div class="message bot-message">{formatted}</div>', unsafe_allow_html=True
                )

        typing_placeholder.empty()
        st.session_state.history.append({"role": "assistant", "content": full_response})
        st.rerun()


if __name__ == "__main__":
    main()
