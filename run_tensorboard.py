import os
import webbrowser
import time

# Path to your logs directory
LOGDIR = "logs"

# Start TensorBoard
os.system(f"start cmd /k tensorboard --logdir={LOGDIR}")

# Wait a moment for TensorBoard to start
time.sleep(20)

# Open TensorBoard in the default web browser
webbrowser.open("http://localhost:6006")