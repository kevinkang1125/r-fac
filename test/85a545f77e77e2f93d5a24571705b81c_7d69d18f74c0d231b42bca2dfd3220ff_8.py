import pandas as pd
import random


class Target:
    def __init__(self, data):
        self.data = data

    def get_probability_matrix(self):
        probability_matrix = {}
        for room, connected_rooms in self.data.items():
            num_connected_rooms = len(connected_rooms)
            probabilities = [1 / (num_connected_rooms + 1) for _ in range(num_connected_rooms)]
            probability_matrix[room] = dict(zip(connected_rooms, probabilities))
            probability_matrix[room][room] = 1 / (num_connected_rooms + 1)  # Add stay probability
        return probability_matrix

    def next_room(self, current_room):
        room_probabilities = self.get_probability_matrix()[current_room]
        rooms = list(room_probabilities.keys())
        probabilities = list(room_probabilities.values())
        return random.choices(rooms, probabilities)[0]


def read_data_from_csv(file_path):
    map_df = pd.read_csv(file_path)
    map_data = {}
    for _, row in map_df.iterrows():
        room, connected_room = int(row["Room"]), int(row["Connected_Room"])
        if room not in map_data:
            map_data[room] = []
        map_data[room].append(connected_room)
    return map_data


def save_probability_matrix_to_csv(probability_matrix, file_path):
    probability_data = []
    for room, connected_rooms in probability_matrix.items():
        for connected_room, probability in connected_rooms.items():
            probability_data.append((room, connected_room, probability))

    probability_df = pd.DataFrame(probability_data, columns=["Room", "Connected_Room", "Probability"])
    probability_df.to_csv(file_path, index=False)
    print(probability_df)

# Example usage
if __name__ == "__main__":
    # file_path = "Map_Info/MUSEUM.csv"
    # data_from_csv = read_data_from_csv(file_path)
    # target = Target(data_from_csv)
    # probability_matrix = target.get_probability_matrix()
    # save_probability_matrix_to_csv(probability_matrix, "Target_Info/MUSEUM_Random.csv")
    #
    # current_room = 1
    # next_room = target.next_room(current_room)
    #
    # file_path = "Map_Info/Grid_10.csv"
    # data_from_csv = read_data_from_csv(file_path)
    # target = Target(data_from_csv)
    # probability_matrix = target.get_probability_matrix()
    # save_probability_matrix_to_csv(probability_matrix, "Target_Info/Grid_10_Random.csv")
    file_path = "Map_Info/OFFICE.csv"
    data_from_csv = read_data_from_csv(file_path)
    target = Target(data_from_csv)
    probability_matrix = target.get_probability_matrix()
    save_probability_matrix_to_csv(probability_matrix, "Target_Info/OFFICE_Random.csv")
