from Queue import *

class Team(object):
    games_learned = 8

    def __init__(self, id_num, name):
        self.name = name
        self.id_num = id_num
        self.games = []
        self.season_end_list = {}
        self.wlp = {}
        self.wins = {}
        self.losses = {}
        self.game_queue = {}

    def initSeason(self, season):
        self.game_queue[season] = Queue()
        self.wlp[season] = 0.
        self.wins[season] = 0
        self.losses[season] = 0

    def addLoss(self, season, game):
        self.losses[season] += 1
        self.games.append(game)
        
        home_str = 1. if game[6] == 'H' else .5 if game[6] == 'N' else 0.
        game_str = []
        game_str.append(float(game[5]) / 100.)
        game_str.append(float(game[3]) / 100.)
        game_str.append(home_str)
        self.insertGame(season, game_str)
        self.calcWP(season)

    def addWin(self, season, game):
        self.wins[season] += 1
        self.games.append(game)
        
        home_str = 1. if game[6] == 'H' else .5 if game[6] == 'N' else 0.
        game_str = []
        game_str.append(float(game[3]) / 100.)
        game_str.append(float(game[5]) / 100.)
        game_str.append(home_str)
        self.insertGame(season, game_str)
        self.calcWP(season)

    def checkSeason(self, season):
        if season in self.game_queue:
            pass
        else :
            #print self.id_num, season, "Init season"
            self.initSeason(season)

    def calcWP(self, season):
        wins = self.wins[season]
        losses = self.losses[season]
        if wins == 0 and losses > 0 :
            self.wlp[season] = 0.
        elif wins > 0 and losses == 0 :
            self.wlp[season] = 1.00
        else : 
            self.wlp[season] = float(wins) / (float(wins) + float(losses))

    def insertGame(self, season, game_str):
        games_l = 8
        if self.game_queue[season].qsize() == games_l:
            self.game_queue[season].get()

        self.game_queue[season].put(game_str)
