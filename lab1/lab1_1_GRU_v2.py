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

class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        #print("h0_shape:", h0.shape)
        #print("x_shape:", x.shape)
        #print("x:", x)
        x = x.unsqueeze(1)
       # print("x_shape_v2:", x.shape)
        # Forward pass through GRU
        out, _ = self.gru(x, h0)

        # Only take the output from the final time step
        out = out[:, -1, :]

        # Pass the output through the fully-connected layer
        out = self.fc(out)
       
        out = self.sigmoid(out)
        return out

    def train_model(self, num_epochs, optimizer, criterion, train_loader, val_loader, patience=10):
        best_val_loss = float('inf')
        stop_count = 0
        best_model_state = None
        print("Beginning training & validation")

        for epoch in range(num_epochs):
            train_loss = 0.0
            self.train()
            for i, (data, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output.squeeze(1), labels.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, (data, labels) in enumerate(val_loader):
                    outputs = self.forward(data)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.state_dict())
                stop_count = 0
                print(f'New best model found at epoch {epoch + 1} with validation loss: {best_val_loss}')
            else:
                stop_count += 1
                print(f'No improvement in validation loss for epoch {epoch+1}. Early stopping counter: {stop_count}/{patience}')
                
                if stop_count >= patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}. No improvement in validation loss for {patience} consecutive epochs.')
                    break

        self.load_state_dict(best_model_state)
        return

    def test_model(self, criterion, test_loader):
        self.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for data, labels in test_loader:
                outputs = self.forward(data)
                loss = criterion(outputs.squeeze(), labels.float())
                test_loss += loss.item()

                predicted = (outputs > 0.5).float()
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions * 100
        print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}')

def chatbot_response(prediction):
    positive = 1
    negative = 0

    chatbot_responses = {
        positive: ("Positive", "Good review"),
        negative: ("Negative", "Bad review")
    }

    return chatbot_responses[prediction.item()][random.randint(0, 1)]

def prep_user_input(user_input, vectorizer):
    user_input = user_input.lower()
    user_input = re.sub(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', user_input)  # remove emails
    user_input = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', user_input)  # remove IP address
    user_input = re.sub(r'[^\w\s]', '', user_input)  # remove special characters
    user_input = re.sub('\d', '', user_input)  # remove numbers
    word_tokens = word_tokenize(user_input)
    filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
    processed_input = " ".join(filtered_sent)

    user_tfidf = vectorizer.transform([processed_input])
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

    input_dim = vocab_size
    hidden_dim = 250
    output_dim = 1  # Binary classification
    model = SentimentClassifier(input_dim, hidden_dim, output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCELoss()
    model.train_model(NUM_EPOCHS, optimizer, criterion, train_loader, val_loader)
    model.test_model(criterion, test_loader)

    print(f"\nType 'exit' to end the conversation.")
    print("Bot: Give a review.")

    while True:
        text = input("User: ")
        if text == "exit":
            break
        user_prompt = prep_user_input(text, word_vectorizer)

        output = model(user_prompt.unsqueeze(0))
        prediction = (output > 0.5).float()
        
        print("Bot:", chatbot_response(prediction))

if __name__ == "__main__":
    main()
