from model import (Transformer,Embeddings,MultiHeadAttention,FeedForward,EncoderLayer,DecoderLayer,
                   LossWithLS,AdamWarmup,Transformer,evaluate)
import torch 
from model import tokenizer,device
from warnings import filterwarnings


checkpoint = torch.load(r"C:\pytorch\chat_bot\new_bot\train_data\transformer_model\checkpoint_19.pth.tar",weights_only=False,map_location=torch.device("cpu"))
transformer = checkpoint['transformer']



def top_p_sampling(logits, p=0.9, temperature=1.0):
    import torch.nn.functional as F

    logits = logits / temperature  

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cumulative_probs > p
    cutoff_idx = torch.argmax(cutoff.int()).item()

    filtered_probs = sorted_probs.clone()
    filtered_probs[cutoff_idx+1:] = 0
    filtered_probs /= filtered_probs.sum()  

    sampled_idx = torch.multinomial(filtered_probs, 1).item()
    return sorted_indices[sampled_idx].item()



def evaluate(transformer, question, question_mask, max_len, tokenizer, top_p=0.9, temperature=1.0):
    import torch.nn.functional as F

    transformer.eval()

    start_token = tokenizer.token_to_id("<start>")
    end_token = tokenizer.token_to_id("<end>")

    encoded = transformer.encode(question, question_mask)
    words = torch.LongTensor([[start_token]]).to(device)

    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)

        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        logits = transformer.logit(decoded[:, -1])  # Last token logits

        next_word = top_p_sampling(logits.squeeze(0), p=top_p, temperature=temperature)

        if next_word == end_token:
            break

        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim=1)

    words = words.squeeze(0).tolist()
    sen_idx = [w for w in words if w not in {start_token, end_token}]
    sentence = tokenizer.decode(sen_idx)
    return sentence


print("+===================================================================================================================================+")
import os
import shutil
print("\n\n")
terminal_size = shutil.get_terminal_size((80, 20)) 
terminal_width = terminal_size.columns
name = "🧠 Anmino SLM 🧠"
print(name.center(terminal_width)) 
print("\t\t\t\t\t\t\t ========================") 

while True:
    

    print("")
    question = input("Question: ")
    print("________________________________________________________________________________________________________________________________")
    
    if question.lower() == 'quit':
        break

    encoding = tokenizer.encode(question)
    enc_qus = encoding.ids
    question_tensor = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
    question_mask = (question_tensor != 0).to(device).unsqueeze(1).unsqueeze(1)

    temperature = 0.7 
    sentence = evaluate(transformer, question_tensor, question_mask, max_len=500, tokenizer=tokenizer, top_p=0.9, temperature=temperature)
    print("Ans: ", end="")
    for x in sentence:
        print(x, end="", flush=True)
        import time
        time.sleep(0.03)
    print()
    print("===============================================================================================================================")
