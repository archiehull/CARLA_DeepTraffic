from keras.utils import plot_model
from CarlaClient_5 import DQNAgent, IMG_HEIGHT, IMG_WIDTH

# Instantiate agent and models
agent = DQNAgent()

# List of model creation methods and names
models = [
    ("Xception", agent.create_model_x),
    ("64x3", agent.create_model_64),
    ("CNN1", agent.cnn_1),
    ("DeepTraffic", agent.create_model_dt),
]

for name, create_fn in models:
    model = create_fn()
    plot_model(
        model,
        to_file=f"{name}_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        dpi=100
    )
    print(f"Saved {name}_architecture.png")