from data_loader_large import load_data
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

x=input("Temp,Humidity,WindSpeed : ").split(',')

def predict(x):
    y=[float(q) for q in x]
    model=WeatherNN()
    X=torch.tensor([y],dtype=torch.float32)
    prediction = model(X)
    print('Prediction: ',prediction)
    if prediction<=0.15:
        print('Sunny')
    elif prediction>=0.95:
        print('Rainy')
    else:
        print('unable to say')

predict(x)
