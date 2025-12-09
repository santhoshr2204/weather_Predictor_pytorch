import torch
from data_loader_large import load_data
from model import WeatherNN
from train import train_model
from utils import save_output

def main():
    # 1. Load dataset
    X, y = load_data()

    # 2. Build model
    model = WeatherNN()

    # 3. Train model
    final_loss = train_model(model, X, y)
    print("\nFinal Training Loss:", final_loss)

    # 4. Predictions on training data
    with torch.no_grad():
        pred = model(X)
    print("Predictions on training data:\n", pred)

    # 5. Save predictions to file
    save_output(pred)

    # 6. ---- Take a random/custom input from user and predict ----
    print("\n--- Test the model with your own weather values ---")
    try:
        temp = float(input("Enter Temperature (°C): "))
        hum = float(input("Enter Humidity (%): "))
        wind = float(input("Enter Wind Speed (km/h): "))

        # Create a tensor for the single input: shape [1, 3]
        sample = torch.tensor([[temp, hum, wind]], dtype=torch.float32)

        # Disable gradient calculations for inference
        with torch.no_grad():
            pred_single = model(sample)

        prob = pred_single.item()   # value between 0 and 1
        label = "Rainy" if prob >= 0.5 else "Sunny"
        print(prob)

        print(f"\nModel output probability (Rainy): {prob:.4f}")
        print("Predicted Weather:", label)

    except ValueError:
        print("Invalid input. Please enter numeric values only.")

if __name__ == "__main__":
    main()
