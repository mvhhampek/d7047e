import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from data import data_loading_code
from torch.utils.data import DataLoader, TensorDataset

device="cpu"

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

def prep_user_input(user_input, vectorizer):
        # Preprocess user input
    user_input = user_input.lower()
    user_input = re.sub(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', user_input)  # remove emails
    user_input = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', user_input)  # remove IP address
    user_input = re.sub(r'[^\w\s]', '', user_input)  # remove special characters
    user_input = re.sub('\d', '', user_input)  # remove numbers
    word_tokens = word_tokenize(user_input)
    filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
    processed_input = " ".join(filtered_sent)

    # Vectorize user input using the provided TF-IDF vectorizer
    user_tfidf = vectorizer.transform([processed_input])

    # Convert to PyTorch tensor
    user_tensor = torch.tensor(user_tfidf.toarray(), dtype=torch.float32)

    return user_tensor

def chatbot_response(prediction):
    positive = 1
    negative = 0

    chatbot_responses: dict[int, tuple[str, str]] = {
        positive: ("Positive",
                          "Good review"),

        negative: ("Negative",
                         "Bad review")
    }  


def main():

    
    train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor, vocab_size, word_vectorizer, test_x_tensor, test_y_tensor =  data_loading_code.get_data_test()

    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    

    train_loader = DataLoader(TensorDataset(train_x_tensor, train_y_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(validation_x_tensor, validation_y_tensor), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x_tensor, test_y_tensor), batch_size=BATCH_SIZE, shuffle=False)


    ntokens = vocab_size  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train_model(NUM_EPOCHS, optimizer, criterion, train_loader, val_loader)
    model.test_model(criterion, test_loader)

    model.eval()

    print(f"\nType 'exit' to end the conversation.",
          sep="\n")

    print("Bot: Give a reivew.")

    while True:
        text = input("User: ")
        if text == "exit":
            break
        user_prompt = prep_user_input(text, word_vectorizer)

        output = model(user_prompt)
        _, predicted = torch.max(output, 1)

        print("Bot:", chatbot_response(predicted), sep=" ")

if __name__ == "__main__":
    main()

