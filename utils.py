import json

def save_output(predictions, filename="result.json"):
    predictions = predictions.detach().numpy().tolist()
    with open(filename, "w") as file:
        json.dump({"Predictions": predictions}, file)

    print("Predictions saved to", filename)
