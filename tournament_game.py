


class Game(object):
    def __init__(self, id_string, team_lower, team_higher, pre_binary, pre_proba):
        self.id_str = id_string
        self.predicted_binary = pre_binary
        self.predicted_probability = pre_proba[0][1]        
        self.lower_team = team_lower
        self.higher_team = team_higher
        self.result = -1.
        self.played = False
        
    def to_prediction(self):
        prediction_string = self.id_str + ',' + str(self.predicted_probability) + '\n'
        return prediction_string
    
    def actual_result(self, result):
        self.result = result
        self.played = True
        
    def dump(self, out):
        out.write('Game id: ', self.id_str, self.lower_team.name, 'vs', self.higher_team.name, '\n')        
        
    @staticmethod
    def create_game_str(season, lower_team, opponent):
        game_str = season + "_" + lower_team + "_" + opponent
        return game_str
        
     