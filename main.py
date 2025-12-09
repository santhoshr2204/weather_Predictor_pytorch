from data_loader import load_data
from model import WeatherNN
from train import train_model
from utils import save_output

def main():
    X, y = load_data()
    model = WeatherNN()

    final_loss = train_model(model, X, y)
    print("Final Training Loss:", final_loss)

    predictions = model(X)
    print("Predictions:", predictions)

    save_output(predictions)

if __name__ == "__main__":
    main()


