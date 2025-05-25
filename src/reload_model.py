import torch
import joblib
import numpy as np
import socket
import ast
import time
import keyboard
from collections import deque

# Define the LSTM model for reload prediction
class LSTMReloadNet(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # LSTM layer with 32 hidden units
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True)
        # Fully connected layers for classification
        self.fc = torch.nn.Sequential( 
            torch.nn.ReLU(), # Activation function
            torch.nn.Linear(32, 16), # Reduce to 16 units
            torch.nn.ReLU(), # Activation function
            torch.nn.Linear(16,1), # Output a single value
            torch.nn.Sigmoid() # Sigmoid for binary
        )
    
    def forward(self, x):
        # Forward pass through the LSTM and fully connected layers
        _, (hh, _) = self.lstm(x) # Get the hidden state from the LSTM
        return self.fc(hh.squeeze(0)) # Pass the hidden state through the fully connected layers

# Class to handle reload prediction logic
class ReloadPredictor:
    def __init__(self, udp_port=5005, seq_len=3):
        self.seq_len = seq_len # Length of the action history sequence
        self.action_history = deque([0]*seq_len, maxlen=seq_len) # Initialize action history with zeros
        self.model = LSTMReloadNet(input_size=seq_len + 7) # Initialize the LSTM model
        self.model.load_state_dict(torch.load("reload_model.pt")) # Load the trained model weight
        self.model.eval() # Set the model to evaluation mode

        self.scaler = joblib.load("scaler.save") # Load the scaler for feature normalization
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create a UDP socket to communicate with the x86 component
        self.sock.bind(("127.0.0.1", udp_port)) # Bind the socket to the specified port
        self.sock.settimeout(0.1) # Set a timeout for receiving data

        self.last_data = None # Store the last received memory data

    # Function to receive the latest memory data from the UDP socket
    def get_latest_memory(self):
        try:
            data, _ = self.sock.recvfrom(1024) # Receive data from the socket
            parsed = ast.literal_eval(data.decode()) # Parse the received data
            self.last_data = parsed # Update the last received data``
        except socket.timeout:
            pass # Ignore timeout errors
        return self.last_data # Return the last received data
    
    # Function to predict whether a reload is needed
    def predict_reload(self, enemy_visible: bool = False):
        mem = self.get_latest_memory() # Get the latest memory data
        if not mem: 
            return None # Return None if no data is received


        features = [
            mem[1],  # health
            mem[2],  # primary_ammo
            mem[3],  # primary_in_mag
            mem[4],  # secondary_ammo
            mem[5],  # secondary_in_mag
            mem[1],  # prev_health_3s
            int(enemy_visible)
        ]

        # Combine action history and features into a single input vector
        input_vec = np.array(list(self.action_history) + features).reshape(1, -1)
        # Normalize the feature part of the input vector
        input_vec[:, self.seq_len:] = self.scaler.transform(input_vec[:, self.seq_len:])
        # Convert the input vector to a PyTorch tensor
        input_tensor = torch.tensor(input_vec, dtype=torch.float32).reshape(1, 1, -1)

        # Perform a forward pass through the model to get the prediction
        with torch.no_grad():
            prediction = self.model(input_tensor).item()

        # Update the action history with the predicted action (1 for reload, 0 for no reload)
        predicted_action = 1 if prediction > 0.5 else 0
        self.action_history.append(predicted_action)
        return prediction # Return the prediction score
    
    # Function to decide whether to reload based on the prediction
    def should_reload(self, enemy_visible: bool = False, auto_press=True):
        prediction = self.predict_reload(enemy_visible)

        if prediction is None:
            # If no prediction is available, assume no reload is needed
            print(f"[+] LSTM N/a NO RELOAD {prediction:.2f}")
            return False
        if prediction > 0.71:
            # If the prediction score is above 55%, suggest a reload
            print(f"[+] LSTM suggested RELOAD {prediction:.2f}")

            if auto_press:
                # Automatically press the reload key if enabled
                keyboard.press('r')
                time.sleep(0.2)
                keyboard.release('r')
            return True
        
        # If the prediction score is below the threshold, no reload
        print(f"[+] LSTM suggested NO RELOAD {enemy_visible} {prediction:.2f}")
        return False