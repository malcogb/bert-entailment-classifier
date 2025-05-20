import gradio as gr
import torch
from transformers import AutoTokenizer
from model import BertClassifier

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertClassifier(model_name, num_labels=3)
model.load_state_dict(torch.load("C:/Users/MSI/Desktop/Deep Learning_2/bert-entailment-classifier/saved_model.pt"))
model.eval()

labels_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

def predict(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True, max_length=259) # c'est le max_length du dataset utilisé (train_6.csv)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs, dim=1).item()
    return labels_map[pred]

demo = gr.Interface(fn=predict, inputs=["text", "text"], outputs="text", title="BERT Entailment Classifier")
demo.launch()

