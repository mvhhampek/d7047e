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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # Initialize hidden state with device-aware code
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        x = x.unsqueeze(1)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]  # Only take the output from the final time step
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

    def train_model(self, num_epochs, optimizer, criterion, train_loader, val_loader, patience=10):
        best_val_loss = float('inf')
        stop_count = 0
        best_model_state = None
        print("Beginning training & validation")

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = self.forward(data)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.state_dict())
                stop_count = 0
            else:
                stop_count += 1

            print(f'Epoch {epoch + 1}: Validation loss = {avg_val_loss:.4f}', end="")
            if stop_count>0:
                print(f" Early stopping counter = {stop_count}/{patience}", end="")
            print()
            if stop_count >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

        if best_model_state:
            self.load_state_dict(best_model_state)

    def test_model(self, criterion, test_loader):
        self.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.forward(data)
                loss = criterion(outputs.squeeze(), labels.float())
                test_loss += loss.item()

                predicted = (outputs.squeeze() >= 0.5).float()
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

        accuracy = (correct_predictions / total_predictions) * 100
        print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%')

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

    # Set parameters
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 32

    train_loader = DataLoader(TensorDataset(train_x_tensor, train_y_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(validation_x_tensor, validation_y_tensor), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x_tensor, test_y_tensor), batch_size=BATCH_SIZE, shuffle=False)

    input_dim = vocab_size
    hidden_dim = 200
    output_dim = 1
    model = SentimentClassifier(input_dim, hidden_dim, output_dim)
    model.to(model.device)


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
        
        user_prompt = user_prompt.to(model.device)

        output = model(user_prompt)
        prediction = (output > 0.5).float()
        
        print("Bot:", chatbot_response(prediction))

if __name__ == "__main__":
    main()
