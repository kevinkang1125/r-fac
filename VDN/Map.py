# map class 只提供在position pt 使用action a 到达下一个position pt+1
import pandas as pd
import os
class Map:
    def __init__(self, map_name):
        self.map_name = map_name
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, f"Map_Info/{self.map_name}.csv")
        self.map = pd.read_csv(file_path)
        self.map_position_num = self._max_room_number_df()
        self.map_action_num = self._max_connected_rooms_df()

    def step(self, position, action):
        next_possible_position = self._find_connected_rooms(position)
        action %= len(next_possible_position)
        return next_possible_position[action]

    def next_total_action(self, position):
        next_possible_position = self._find_connected_rooms(position)
        return len(next_possible_position)

    def _find_connected_rooms(self, room):
        connected_room = self.map[self.map["Room"] == room]["Connected_Room"].tolist()
        connected_room.append(room)
        return connected_room

    def _max_connected_rooms_df(self):
        connected_max = self.map.groupby("Room").size().max()
        return connected_max + 1

    def _max_room_number_df(self):
        return max(self.map["Room"].max(), self.map["Connected_Room"].max())