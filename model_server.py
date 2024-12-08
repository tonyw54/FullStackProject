import os
import torch
import torch.nn as nn
from transformers import BertTokenizer
import flask
from flask import request, jsonify, render_template

vocab_size = 30522  # Number of tokens in BERT vocabulary
embedding_dim = 128
hidden_dim = 512
output_dim = 1  # Binary classification (0 or 1)

class RNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, vocab_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=10, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        final_output = hn[-1]
        out = self.fc(final_output)
        return self.sigmoid(out)

class ModelPredictor:
    def __init__(self, model_path, tokenizer_name, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
        # Define the model architecture
        self.model = RNNClassifier(embedding_dim, hidden_dim, output_dim, vocab_size)
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def predict(self, text):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'])
            
        # Process and return predictions (modify based on your model's output)
        predictions = outputs.squeeze().cpu().numpy().tolist()
        return predictions

# Flask app setup
app = flask.Flask(__name__)
predictor = ModelPredictor(
    model_path='model.pth',  # Path inside the Docker container
    tokenizer_name='bert-base-uncased'  # Replace with your tokenizer
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        predictions = predictor.predict(text)
        return jsonify({
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)