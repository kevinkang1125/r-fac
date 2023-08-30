import pandas as pd
import random
import os
class TargetModel:
    def __init__(self, file_name):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, f"Target_Info/{file_name}.csv")
        self.probability_matrix = self._read_probability_matrix_from_csv(file_path)

    def _read_probability_matrix_from_csv(self, file_path):
        probability_df = pd.read_csv(file_path)
        probability_matrix = {}
        for _, row in probability_df.iterrows():
            room, connected_room, probability = int(row["Room"]), int(row["Connected_Room"]), row["Probability"]
            if room not in probability_matrix:
                probability_matrix[room] = {}
            probability_matrix[room][connected_room] = probability
        return probability_matrix

    def next_position(self, current_room):
        room_probabilities = self.probability_matrix[current_room]
        rooms = list(room_probabilities.keys())
        probabilities = list(room_probabilities.values())
        return random.choices(rooms, probabilities)[0]

# Example usage
if __name__ == "__main__":
    target_model = TargetModel("MUSEUM_Random")
    current_room = 1
    next_room = target_model.next_position(current_room)
    print(f"Next room from {current_room}: {next_room}")
