from college_team import Team
from tournament_game import Game
from Queue import *
from math import log
import pylab as pl
import time

from sklearn import svm, preprocessing, metrics

def vectorize(q_w, q_l, wp_1, wp_2):
    ret_q = []
    for item in list(q_w.queue):
        ret_q.extend(item)
    for item in list(q_l.queue):
        ret_q.extend(item)
    ret_q.append(wp_1)
    ret_q.append(wp_2)
    #print "Check", wp_1, wp_2
    return ret_q

def load_teams(fn):
    teamDict = {}
    fd = open(fn)
    # Dictionary of classes based on index - 
    for line in fd:
        lst = line.split(',')
        id = lst[0]
        teamDict[id] = Team(lst[0].strip(), lst[1].strip())
       
    fd.close()
    return teamDict
    
def build_clf(results_fn, training_seasons):
    results_file = open(results_fn)
    
    training_set_games = []
    training_set_wl = []
    
    for ind_game in results_file:
        # iterate over each season
        cur_game = ind_game.split(',')
        season = cur_game[0]
        # Skip header
        if season == 'season' : 
            continue;
        elif season not in training_seasons:
            continue
    
        # Pull winning team, losing team IDs  

        wteam = teams[cur_game[2]]
        lteam = teams[cur_game[4]]

        wteam.checkSeason(season)
        lteam.checkSeason(season)

        w_q = wteam.game_queue[season]
        l_q = lteam.game_queue[season]

        # If game # greater or equal to number we need to learn, start adding to training set
        # Also note, add values to team regarldless - only add to training set if in appropriate season
        if season in training_seasons and w_q.qsize() >= games_learn_num and l_q.qsize() >= games_learn_num :
            # Add winning train
            training_set_games.append(vectorize(w_q, l_q, wteam.wlp[season], lteam.wlp[season]))
            training_set_wl.append(1.)

            # Add losing data point
            training_set_games.append(vectorize(l_q, w_q, lteam.wlp[season], wteam.wlp[season]))
            training_set_wl.append(0.)

        # Add both games to each teams games list
        wteam.addWin(season, cur_game)
        lteam.addLoss(season, cur_game)
        
    results_file.close()
    
    # Do classification here
    start_time = time.time()
    print "Begin Classification", start_time

    #print normal_games
    clf = svm.SVC(gamma=0.001, probability=True)
    normal_games = preprocessing.normalize(training_set_games)
    #classifier.fit(normal_games, training_set_wl)
    clf.fit(training_set_games, training_set_wl)
    print "Classification took", time.time() - start_time, "seconds"
    
    return clf

def load_tournament_teams(teams_fn, predicting_seasons):
    t_teams = {}
    
    tournament_fd = open(teams_fn, 'r')
    
    for line in tournament_fd :
        seed = line.split(',')
        if seed[0] == 'season':
            continue
        season = seed[0].strip()
        
        # Skip any game in a non-predicting season
        if season not in predicting_seasons :
            continue
        
        team_id = seed[2].strip()
        if season in t_teams :
            pass;
        else :
            t_teams[season] = []
        t_teams[season].append(team_id)
        
    tournament_fd.close()
    
    return t_teams

# Create a prediction for all game possibilities of a given tournament
def predict_games(t_teams, clf, outfile_fn, out):
    pred_fd = open(outfile_fn, 'w')    
    header = "id,pred\n"        
    pred_fd.write(header)
                   
    prediction_dict = {}
   
    # For producing submissions
    for season in t_teams:
        #print "Key : ",key
        t_teams[season].sort()
        
        game_vect = []
        
        # Begin with first team of season - iterate over all remaining
        while len(t_teams[season]) > 1 :
            lower_team = t_teams[season].pop(0)
            for opponent in t_teams[season] :
                # Predict percentage for 'lower team', or team with 
                this_game = vectorize(teams[lower_team].game_queue[season], teams[opponent].game_queue[season], 
                        teams[lower_team].wlp[season], teams[opponent].wlp[season])

                game_key = Game.create_game_str(season, lower_team, opponent)

                predict_probability = clf.predict_proba(this_game)
                predict_binary = clf.predict(this_game)

                prediction_dict[game_key] = Game(game_key, lower_team, opponent, predict_binary, predict_probability)
                
                pred_fd.write(prediction_dict[game_key].to_prediction())
                game_vect.append(this_game)
        print "Predictions for tournament season", season

    pred_fd.close()
    
    return prediction_dict

def get_results(results_fn, game_dict):
    results_fd = open(results_fn)

    # For testing
    for result in results_fd:
        split_res = result.split(',')
        season = split_res[0].strip()
        if season not in predict_season :
            continue
        winner = split_res[2].strip()
        loser = split_res[4].strip()
        if winner < loser :
            game_str = Game.create_game_str(season, winner, loser)

            # Game considered win            
            game_dict[game_str].actual_result(1.)

        else :
            game_str = Game.create_game_str(season, loser, winner)

            # Game considered loss  
            game_dict[game_str].actual_result(0.)

    results_fd.close()

def evaluate_clf(clf, game_dict, out):
    predicted_bin = []
    actual_bin = []
    
    num_wrong = 0.
    num_correct = 0.
        
    for game in game_dict:
        if not game_dict[game].played :
            continue
        predicted = game_dict[game].predicted_binary[0]
        result = game_dict[game].result
        proba = game_dict[game].predicted_probability
        id_str = game_dict[game].id_str
        predicted_bin.append(predicted)
        actual_bin.append(result)
    
        if result == predicted :
            result_str =  "Correct!"
            num_correct += 1.
        else :
            result_str = "Wrong!" 
            num_wrong += 1.
        
        split_games = game_dict[game].id_str.split('_')
        game_str = teams[split_games[1]].name + " vs " + teams[split_games[2]].name    
        log_str = id_str + ' ' + game_str + ", Outcome: " + str(result) + ", Probability: " + str(proba) + ", Prediction: " + str(predicted) + ', ' + result_str + '\n'
        out.write(log_str) 
    
    correct_perct = num_correct / (num_correct + num_wrong)
    print "Results, Num Correct:", num_correct, "Num wrong:", num_wrong, ",% :", correct_perct    
        
    print("Classification report for classifier %s:\n%s\n"
    % (classifier, metrics.classification_report(actual_bin, predicted_bin)))
    
def calc_log_loss(game_dict):
    loss = 0.
    n = 0
    # - 1/n sum(1:n)[(yilog(yi^) + (1 - yi)log(1-yi^)]
    # n is number of games played
    # yi^ is the predicted probability
    # yi is outcome
    # log is base e
    for game in game_dict :
        if not game_dict[game].played :
            continue
        yi = game_dict[game].result
        yihat = game_dict[game].predicted_probability
        loss += yi * log(yihat) + (1. - yi) * log(1 - yihat)
        n += 1
    loss = -(loss)/n
    return loss

teams_filename = "teams.csv"
season_results = "regular_season_results.csv"
t_teams_fn = "tourney_seeds.csv"
prediction_fn = "prediction.csv"
results_fn = "tourney_results.csv"
log_file = "log.txt"

games_learn_num = 8

train_seasons = ['R']
#train_seasons = ['N', 'O', 'P', 'Q','R']

predict_season = ['R']
#predict_season = ['N', 'O', 'P', 'Q','R']

log_fd = open(log_file, 'w')

start_time = time.time()

teams = load_teams(teams_filename)

classifier = build_clf(season_results, train_seasons);

t_teams = load_tournament_teams(t_teams_fn, predict_season)

prediction_dict = predict_games(t_teams, classifier, prediction_fn, log_fd);

get_results(results_fn, prediction_dict)

evaluate_clf(classifier, prediction_dict, log_fd)

log_loss = calc_log_loss(prediction_dict)

print "*******************"
print "** Log Loss :", log_loss
print "*******************"
log_fd.write("*******************\n**  Log Loss:" + str(log_loss) + "\n*******************\n")

log_fd.close()

print "Total time taken :", time.time()-start_time, "seconds"

