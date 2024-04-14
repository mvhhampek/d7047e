import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from data import data_loading_code
from torch.utils.data import DataLoader, TensorDataset
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTM, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_dim,  # Input dimension is the size of the TF-IDF feature vector
                            hidden_dim,  # Hidden layer dimension
                            num_layers=n_layers,  # Number of LSTM layers
                            bidirectional=bidirectional,  # Whether the LSTM is bidirectional
                            dropout=dropout if n_layers > 1 else 0,  # Dropout (only applied if num_layers > 1)
                            batch_first=True)  # Input and output tensors are provided as (batch, seq, feature)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 
                            output_dim)  # Adjusting for bidirectional

    def forward(self, x):
        # No need to embed since we are directly using TF-IDF features
        # x: [batch_size, seq_len, feature_size] - seq_len is artificially considered as 1 here

        # Passing the inputs to the LSTM layer
        x = x.unsqueeze(1)  # Add a sequence length dimension (seq_len=1)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Concatenate the final forward and backward hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Passing the output of the LSTM to the fully connected layer
        out = self.fc(hidden)
        return out
    
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
                total_predictions += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions * 100
        print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}')


def chatbot_response(prediction):
    positive = 1
    negative = 0

    chatbot_responses: dict[int, tuple[str, str]] = {
        positive: ("Positive",
                          "Good review"),

        negative: ("Negative",
                         "Bad review")
    }  

    return chatbot_responses[prediction.item()][random.randint(0, 1)]

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


def main():

    
    train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor, vocab_size, word_vectorizer, test_x_tensor, test_y_tensor =  data_loading_code.get_data_test()

    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    

    train_loader = DataLoader(TensorDataset(train_x_tensor, train_y_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(validation_x_tensor, validation_y_tensor), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x_tensor, test_y_tensor), batch_size=BATCH_SIZE, shuffle=False)


    input_dim = vocab_size  # From your dataset
    hidden_dim = 250  # Example, adjust based on your needs and model capacity
    output_dim = 2  # Assuming binary classification, adjust as necessary
    n_layers = 2  # Starting with 2 layers is common, adjust based on complexity
    bidirectional = True  # Depending on your need for capturing context in both directions
    dropout = 0.5  # Helps prevent overfitting, especially important with smaller datasets

    model = LSTM(input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

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

    


