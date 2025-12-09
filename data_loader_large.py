import torch

def load_data():
    # Temperature, Humidity, WindSpeed
    # Expanded to 20 samples
    X = torch.tensor([
        # --- Original Data ---
        [30, 70, 10],  # Sunny
        [28, 65, 12],  # Sunny
        [25, 80, 5],   # Rainy
        [22, 90, 4],   # Rainy
        [27, 75, 7],   # Rainy
        
        # --- New Sunny Data (High Temp, Low Humidity, High Wind) ---
        [32, 60, 14],  # Sunny
        [29, 68, 11],  # Sunny
        [31, 62, 13],  # Sunny
        [33, 55, 15],  # Sunny
        [29, 66, 10],  # Sunny
        [30, 69, 12],  # Sunny
        [28, 63, 11],  # Sunny
        [31, 58, 14],  # Sunny

        # --- New Rainy Data (Low Temp, High Humidity, Low Wind) ---
        [24, 85, 6],   # Rainy
        [23, 88, 3],   # Rainy
        [26, 82, 5],   # Rainy
        [21, 92, 2],   # Rainy
        [25, 78, 6],   # Rainy
        [22, 89, 4],   # Rainy
        [24, 84, 5]    # Rainy
    ], dtype=torch.float32)

    # 0 = Sunny, 1 = Rainy
    y = torch.tensor([
      [0], 
      [0], 
      [1], 
      [1], 
      [1],
      [0], 
      [0], 
      [0], 
      [0], 
      [0], 
      [0], 
      [0], 
      [0],
      [1], 
      [1], 
      [1], 
      [1], 
      [1], 
      [1], 
      [1]
    ], dtype=torch.float32)

    return X, y
