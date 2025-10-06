from huggingface_hub import login
import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoModel, ViTModel, ViTImageProcessor
import os
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")  
login(hf_token)

# ---------------------------
# Config
# ---------------------------
CONFIG = {
    "vit_name": "google/vit-base-patch16-224-in21k",
    "text_name": "vinai/phobert-base",
    "answer_space": "./model/answer_space.txt",  
    "save_path": "./model/best_model.pth",       
    "max_len": 20,
    "num_question_types": 3,     
    "type_embedding_dim": 128,
    "dropout": 0.3,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ---------------------------
# Utils
# ---------------------------
def normalize_answer_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


# ---------------------------
# Model
# ---------------------------
class VQAModel(nn.Module):
    def __init__(self, num_answers: int, vit_name: str, text_name: str,
                 max_question_type: int, type_embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_name)
        self.text_encoder = AutoModel.from_pretrained(text_name)
        vit_dim = self.vit.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size
        self.type_embedding = nn.Embedding(max_question_type + 1, type_embedding_dim)
        combined_feat_dim = vit_dim + text_dim + type_embedding_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_feat_dim),
            nn.Linear(combined_feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_answers)
        )

    def forward(self, images, input_ids, attention_mask, question_type):
        img_feat = self.vit(images).pooler_output
        q_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        type_feat = self.type_embedding(question_type)
        combined_feat = torch.cat([img_feat, q_feat, type_feat], dim=1)
        return self.classifier(combined_feat)


# ---------------------------
# Load tokenizer, processor, answer space
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_name"])
image_processor = ViTImageProcessor.from_pretrained(CONFIG["vit_name"])
with open(CONFIG["answer_space"], "r", encoding="utf-8") as f:
    answer_space = [normalize_answer_text(x) for x in f.read().splitlines()]
num_answers = len(answer_space)


# ---------------------------
# Load model + checkpoint
# ---------------------------
model = VQAModel(
    num_answers=num_answers,
    vit_name=CONFIG["vit_name"],
    text_name=CONFIG["text_name"],
    max_question_type=CONFIG["num_question_types"],
    type_embedding_dim=CONFIG["type_embedding_dim"],
    dropout=CONFIG["dropout"]
).to(CONFIG["device"])

checkpoint = torch.load(CONFIG["save_path"], map_location=CONFIG["device"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ---------------------------
# Inference function
# ---------------------------
def predict(image_path: str, question: str, question_type: int = 0):
    device = CONFIG["device"]

    # Load image
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    # Tokenize question
    enc = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=CONFIG["max_len"],
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    question_type_tensor = torch.tensor([question_type], dtype=torch.long).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values, input_ids, attention_mask, question_type_tensor)
        pred_idx = outputs.argmax(dim=1).item()

    return answer_space[pred_idx]


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    image_path = "./image.png" 
    question = "Có mấy con mèo?"
    answer = predict(image_path, question, question_type=1)
    print("Predicted answer:", answer)

