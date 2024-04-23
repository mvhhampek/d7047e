import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import random
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import copy


import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from data import data_loading_code
from torch.utils.data import DataLoader, TensorDataset, Dataset

device="cpu"

class TextDataset(Dataset):
    """Dataset class for text data and labels."""
    def __init__(self, text_data, labels):
        """
        Args:
            text_data (Tensor): Padded sequence of token indices.
            labels (Tensor): Labels corresponding to the text data.
        """
        self.text_data = text_data
        self.labels = labels

    def __len__(self):
        """Return the number of items in the dataset."""
        return self.text_data.size(0)

    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        return self.text_data[idx, :], self.labels[idx]

def create_dataloaders(train_data, train_labels, val_data, val_labels, test_data, test_labels, batch_size=32):
    """
    Creates DataLoader instances for training, validation, and testing datasets.
    
    Args:
        train_data, val_data, test_data: Tensors of padded sequences of token indices.
        train_labels, val_labels, test_labels: Tensors of labels.
        batch_size (int): Number of samples per batch.
        
    Returns:
        Three DataLoader instances for the training, validation, and test datasets.
    """
    # Create Dataset instances

    train_dataset = TextDataset(train_data, train_labels)
    val_dataset = TextDataset(val_data, val_labels)
    test_dataset = TextDataset(test_data, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


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
            output Tensor of shape ``[batch_size, ntoken]``
        """
                
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0)).to(src.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        output = output.mean(dim=1)

        return output





    
    def train_model(self,num_epochs, optimizer, criterion, train_loader, val_loader, patience = 10):
        best_val_loss = float('inf')
        stop_count = 0
        best_model_state = None
        print("Beginning training & validation")
        
        # Training loop
        for epoch in range(num_epochs):
            train_loss = 0.0
            self.train()
            for i, (data, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.forward(data)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if i % 1 == 0:  # Print training loss every 20 mini-batches
                    print(f'\rEpoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {train_loss / (i+1):.4f}', end="")
            print()
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, (data, labels) in enumerate(val_loader):
                    outputs = self.forward(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)


            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.state_dict())  # Deep copy the model state
                stop_count = 0  # Reset the early stopping counter
                print(f'New best model found at epoch {epoch + 1} with validation loss: {best_val_loss}')
            else:
                stop_count += 1  # Increment the counter if no improvement
                print(f'No improvement in validation loss for epoch {epoch+1}. Early stopping counter: {stop_count}/{patience}')
                
                if stop_count >= patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}. No improvement in validation loss for {patience} consecutive epochs.')
                    break  # Break out of the loop to stop training

        self.load_state_dict(best_model_state)  
        return
    def test_model(self, criterion, test_loader):
        self.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():  # Disable gradient computation for efficiency
            for data, labels in test_loader:
                outputs = self.forward(data)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
                correct_predictions += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader.dataset)
        accuracy = correct_predictions / (len(test_loader.dataset)) * 100
        print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}')

    
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
    

def prep_user_input(user_input, vocab, tokenizer):
    # Preprocess user input
    user_input = user_input.lower()
    user_input = re.sub(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', user_input)  # remove emails
    user_input = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', user_input)  # remove IP address
    user_input = re.sub(r'[^\w\s]', '', user_input)  # remove special characters
    user_input = re.sub('\d', '', user_input)  # remove numbers
    word_tokens = word_tokenize(user_input)
    filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
    processed_input = " ".join(filtered_sent)

    # # Vectorize user input using the provided TF-IDF vectorizer
    # user_tfidf = vectorizer.transform([processed_input])

    # # Convert to PyTorch tensor
    # user_tensor = torch.tensor(user_tfidf.toarray(), dtype=torch.float32)
    user_tensor = data_process_single(processed_input,vocab, tokenizer)

    return user_tensor

def data_process_single(raw_text, vocab, tokenizer):
    # Tokenize the raw text
    #tokenized_text = vocab(tokenizer(raw_text))
    
    # Convert tokens to tensor and add padding
    #tensor_text = torch.tensor(tokenized_text, dtype=torch.long)
    #max_length = len(tensor_text)
    #padded_text = torch.nn.functional.pad(tensor_text, (0, max_length - len(tensor_text)), value=vocab["<pad>"])
    
    tokenized_text = [vocab[token] for token in tokenizer(raw_text)]
    tensor_text = torch.tensor([tokenized_text], dtype=torch.long)
    return tensor_text

def chatbot_response(prediction):
    positive = 1
    negative = 0

    chatbot_responses: dict[int, tuple[str, str]] = {
        positive: ("Positive","Good review"),
        negative: ("Negative","Bad review")
    }
    return chatbot_responses[prediction.item()][random.randint(0, 1)]



def main():

    train_x, train_y, val_x, val_y, test_x, test_y, vocab, tokenizer = data_loading_code.get_data_transformer()
 


    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32

    train_loader, val_loader, test_loader = create_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y, batch_size=BATCH_SIZE)


    ntokens = len(vocab)  # size of vocabulary
    emsize = 256  # embedding dimension
    d_hid = 512  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 8  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.5  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    print("Model initialized")
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
        user_prompt = prep_user_input(text, vocab, tokenizer)

        output = model(user_prompt)
        _, predicted = torch.max(output, 1)
        print(predicted)
        print("Bot:", chatbot_response(predicted), sep=" ")

if __name__ == "__main__":
    main()

