"""
Filename: lobby_sim.py
Name: terracubist
Date: 04/06/26
Purpose: monte carlo how lobbies can shake out
         w/ various strength profiles
"""

## imports
import pandas as pd
import numpy as np
import random
import functools
from scipy.stats import norm
from scipy.stats import bootstrap
from scipy import special as sp
import networkx as nx
import math
import timeit
import code
# import js

## settings
NUM_SIMS = int(1e2)
NUM_PLAYERS = 8
START_HP = {"p"+str(x): 100 for x in range(1,9)}
USING_CSV = False
INPUT_FILENAME = "./str_series_tempo.csv"
STAGE_DMG = {2: 2, 3: 5, 4: 8, 5: 10, 6: 12, 7: 17}
PVE_ROUND_NUMS = [5, 11, 17, 23, 29, 35] # hardcoded for last_x speedup
COMP_MATCHUPS = pd.DataFrame([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.5, 2.0], [1.0, 2.0, 1.0, 0.5], [1.0, 0.5, 2.0, 1.0]], index=["Vanilla", "Rock", "Paper", "Scissors"],columns=["Vanilla", "Rock", "Paper", "Scissors"])
COMP_TYPES = {'p1': 'Vanilla', 'p2': 'Vanilla', 'p3': 'Vanilla', 'p4': 'Vanilla', 'p5': 'Vanilla', 'p6': 'Vanilla', 'p7': 'Vanilla', 'p8': 'Vanilla'}

# NUM_SIMS = js.numSims
# NUM_PLAYERS = 8
# START_HP = {"p"+str(x["Player"]):x["HP"] for x in js.startingHP.to_py()}
# USING_CSV = False
# # INPUT_FILENAME = "./str_series_tempo.csv"
# STAGE_DMG = {int(x["stage"]):x["damage"] for x in js.stageDamage.to_py()}
# PVE_ROUND_NUMS = [5, 11, 17, 23, 29, 35] # hardcoded for last_x speedup

# # have to do a little massaging to get matchups table (triangle matrix) into useful form (filled w/ inverse vals)
# COMP_MATCHUPS = pd.DataFrame(js.matchups.to_py()).set_index('row')
# COMP_MATCHUPS.index.name = None
# COMP_MATCHUPS = COMP_MATCHUPS.replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)
# COMP_MATCHUPS = COMP_MATCHUPS.fillna(1/COMP_MATCHUPS.T)
# COMP_TYPES = {}
# js_playerProfilesMetadata = js.playerProfilesMetadata.to_py()
# for profile in js_playerProfilesMetadata.keys():
#     if "Player " in profile:
#         COMP_TYPES["p"+profile.replace("Player ","")] = js_playerProfilesMetadata[profile]["comp_type"]

## debug tools
DEBUG_PRINT = False
random.seed(10) # fix RNG for debug
GHOST_FIGHT_TRACKER = {'p1': 0,
                       'p2': 0,
                       'p3': 0,
                       'p4': 0,
                       'p5': 0,
                       'p6': 0,
                       'p7': 0,
                       'p8': 0}
p_names = [f"p{x}" for x in list(range(1, NUM_PLAYERS+1))]
OPP_FIGHT_TRACKER = {fighter: {opponent: 0 for opponent in p_names} for fighter in p_names} #faster as dict of dict than df
CALC_UNIT_LOSS_CALLS = 0
MATCH_MAKE_CALLS = [0]*8
MATCH_MAKE_TIMES = []
LAST_X_CALLS = 0
CALC_PLACEMENT_TIMES = []

# # control debug printing
# import contextlib
# import io
# import sys
# def suppress_print(func):
#     def wrapper(*args, **kwargs):
#         if not DEBUG_PRINT:
#             with contextlib.redirect_stdout(io.StringIO()):
#                 return func(*args, **kwargs)
#         else:
#             return func(*args, **kwargs)
#     return wrapper

def print_wrapper(print_str):
    if DEBUG_PRINT:
        print(print_str)

def timer_wrapper(print_str):
    """
    handles multiple calls to timer (similar to a stopwatch's lap function)
    """
    timer_wrapper.start_times.append(timeit.default_timer())
    elapsed = timer_wrapper.start_times[-1] - timer_wrapper.start_times[-2]
    print("Time to {}: {:.2f} seconds".format(print_str, elapsed))
    return elapsed

## class defs
class Player:
    def __init__(self, str_series, comp_type, p_start_hp):
        self.str_series = str_series
        self.comp_type = comp_type
        self.health = p_start_hp
        self.match_history = pd.Series([np.nan]*len(str_series), copy=False)

    def __str__(self):
        return f"{self.health} HP, str = {list(self.str_series)}"

    def is_alive(self):
        return self.health > 0

    def last_x_opponents(self, round_num, x):
        global LAST_X_CALLS
        if x == 0 or round_num == 0:
            result = []
        else:
            # # turns out that the naive code below is slow, pulling and looking for NaN is expensive
            # result = self.match_history[max(0, round_num-x):round_num]
            # if result.isna().any(): #if pve in last_x, add 1 more to return
            #     result = self.match_history[max(0, round_num-(x+1)):round_num]

            # so instead determine whether a pve round is in range pulled, and if so pull 1 more
            start_index = round_num-x
            round_range = range(start_index, round_num)
            if any(x in round_range for x in PVE_ROUND_NUMS):
                start_index += -1 # add 1 more round for pve
            start_index = max(0, start_index)

            result = self.match_history[start_index:round_num]

            # if round_num >= 5: #debugging pve
            #     print("debugging pve")
            #     code.interact(local=dict(globals(), **locals()))

        LAST_X_CALLS += 1 # looks like 177681 for 1e3 lobbies

        # >>> timeit.timeit(stmt="lobby.players['p1'].last_x_opponents(3,1)",globals=globals(),number=int(2e5))
        # 3.0166236999994

        return list(result)



class Lobby:
    def __init__(self, df_in):

        # set up players
        self.players = {}
        for i in range(1, NUM_PLAYERS+1):
            p_name = f"p{i}"
            p_comp_type = COMP_TYPES[p_name]
            p_start_hp = START_HP[p_name]
            self.players[p_name] = Player(df_in[p_name], p_comp_type, p_start_hp)

        # read in round translation table
        df_in["round_name"] = df_in["round_name"].apply(lambda x: x.replace("'", "")) # strip excel date stupidity
        self.round_trans_table = df_in.set_index("round_num")["round_name"].to_dict()

        self.round_num = 0
        max_rounds = len(df_in["p1"])
        self.history = pd.DataFrame(np.nan, index=range(max_rounds), columns=[f"p{x}" for x in list(range(1, NUM_PLAYERS+1))])

        # print(self)

    def __str__(self):
        str_out = ""
        str_out = f"Round: {self.round_name()}\n"
        str_out += "Players:\n"
        for i in range(1, NUM_PLAYERS+1):
            str_out += str(self.players[f"p{i}"]) + "\n"
        return str_out

    def stage(self):
        return int(self.round_trans_table[self.round_num][0])

    def round_name(self):
        return self.round_trans_table[self.round_num]

    def is_pve(self):
        return self.round_name()[-1] == "7"

    def num_players_alive(self):
        return len(self.alive_players())

    def alive_players(self):
        alive = []
        for player in self.players:
            if self.players[player].health > 0:
                alive += [player]
        return alive

    # @suppress_print # comment out for debug, will make code.interact REPL weird
    def sim_round(self):
        try:
            print_wrapper(f"Simulating round {self.round_name()}...")
        except:
            print("error in sim_round")
            # code.interact(local=dict(globals(), **locals()))

        if self.is_pve():
            print_wrapper("Skipping PVE round...")

        elif self.round_num >= 34:
            # "randomly" assign winners by killing everyone off on 7-6, if for some reason it goes long
            for player in self.players:
                self.players[player].health -= 200

        else:
            # calc matchups
            matchups = self.match_make()
            # matchups = {}
            # num_attempts = 0
            # while matchups == {} and num_attempts < 20:
            #     if num_attempts >= 10:
            #         matchups = self.match_make(avoid_last=1) # override
            #         if self.num_players_alive() > 3:
            #             print_wrapper("debugging matchmaking errors")
            #             for player in self.alive_players():
            #                 print_wrapper(self.players[player].last_x_opponents(self.round_num, 4))
            #             code.interact(local=dict(globals(), **locals()))
            #     else:
            #         matchups = self.match_make()
            #     num_attempts += 1
            # if num_attempts >= 20:
            #     print_wrapper("debugging matchmaking errors")
            #     code.interact(local=dict(globals(), **locals()))
            print_wrapper(f"Matchups are: {str(matchups)}")

            # calc player dmg
            results = self.sim_fights(matchups)
            print_wrapper(f"Results are: {str(results)}")

            # apply dmg
            for result in results:
                unit_loss = results[result]
                if unit_loss <= 0:
                    total_dmg = abs(unit_loss) + STAGE_DMG[self.stage()]
                    self.players[result].health -= total_dmg

            # debugging print
            hp_str = ""
            for player in self.players:
                hp_str += f"{player}: {self.players[player].health} HP, "
            hp_str = hp_str[:-2] # strip off trailing comma
            print_wrapper(hp_str)

        # record history
        for player in self.players:
            # stop recording hp after initial death
            if self.round_num == 0 or self.history.at[max(0, self.round_num - 1), player] > 0:
                self.history.at[self.round_num, player] = self.players[player].health

        self.round_num += 1

        print_wrapper("")

    def match_make(self):
        global MATCH_MAKE_CALLS, MATCH_MAKE_TIMES
        start_time = timeit.default_timer()

        player_count = self.num_players_alive()
        
        # using matchmaking 
        if player_count == 8:
            avoid_last = 4
        elif player_count == 7:
            avoid_last = 3
        elif (player_count == 4) or (player_count == 5) or (player_count == 6):
            avoid_last = 2
        else:
            avoid_last = 0 # TODO: figure out if you're guaranteed ghost fights by watching vods

        # figure out valid opponents
        alive_players_list = self.alive_players()
        prefs = {key: [] for key in alive_players_list}
        for player in alive_players_list:
            # get list of last x matches, remove those from options
            last_x = self.players[player].last_x_opponents(self.round_num, avoid_last)
            options = [x for x in alive_players_list if x not in last_x and x != player]
            random.shuffle(options) # to fix deterministic ghost fight results
            prefs[player] = options

            if options == []:
                print("empty options")
                # code.interact(local=dict(globals(), **locals()))

        # code.interact(local=dict(globals(), **locals()))
        # >>> stmt="""alive_players_list = self.alive_players()
        # ... prefs = {key: [] for key in alive_players_list}
        # ... for player in alive_players_list:
        # ...     # get list of last x matches, remove those from options
        # ...     last_x = self.players[player].last_x_opponents(self.round_num, avoid_last)
        # ...     options = [x for x in alive_players_list if x not in last_x and x != player]
        # ...     random.shuffle(options) # to fix deterministic ghost fight results
        # ...     prefs[player] = options
        # ...
        # ...     if options == []:
        # ...         print("empty options")
        # ...         code.interact(local=dict(globals(), **locals()))"""
        # >>> timeit.timeit(stmt=stmt, globals=globals(), number=int(25297))
        # 0.4252393000006123


        # construct graph, find pairs
        keys = list(prefs.keys()) # shuffle keys to fix deterministic ghost fights
        random.shuffle(keys)
        graph_dict = {key: prefs[key] for key in keys}
        edges = nx.max_weight_matching(nx.Graph(graph_dict), maxcardinality=True)
        # code.interact(local=dict(globals(), **locals()))

        # >>> stmt="""keys = list(prefs.keys()) # shuffle keys to fix deterministic ghost fights
        # ... random.shuffle(keys)
        # ... graph_dict = {key: prefs[key] for key in keys}
        # ... edges = nx.max_weight_matching(nx.Graph(graph_dict), maxcardinality=True)"""
        # >>> timeit.timeit(stmt=stmt, globals=globals(), number=int(25297))
        # 3.602003699999841

        # translate from graph pairs to dict
        matchups = {}
        for edge in edges:
            fighter = edge[0]
            opponent = edge[1]
            matchups[fighter] = opponent
            matchups[opponent] = fighter
            OPP_FIGHT_TRACKER[fighter][opponent] += 1
            OPP_FIGHT_TRACKER[opponent][fighter] += 1

        # code.interact(local=dict(globals(), **locals()))
        # >>> stmt="""matchups = {}
        # ... for edge in edges:
        # ...     fighter = edge[0]
        # ...     opponent = edge[1]
        # ...     matchups[fighter] = opponent
        # ...     matchups[opponent] = fighter
        # ...     OPP_FIGHT_TRACKER[fighter][opponent] += 1
        # ...     OPP_FIGHT_TRACKER[opponent][fighter] += 1"""
        # >>> timeit.timeit(stmt=stmt, globals=globals(), number=int(25297))
        # 0.01939469999888388

        # assign ghost fights
        if len(matchups) != player_count:
            for player in self.alive_players():
                if player not in matchups:
                    ghost_opp = random.choice(prefs[player])
                    matchups[player] = ghost_opp
                    GHOST_FIGHT_TRACKER[player] += 1
                    print_wrapper(f"Adding ghost fight for {player}: {ghost_opp}")
                    # code.interact(local=dict(globals(), **locals()))
                    break

        # >>> stmt="""for player in self.alive_players():
        # ...     if player not in matchups:
        # ...         ghost_opp = random.choice(prefs[player])
        # ...         matchups[player] = ghost_opp
        # ...         GHOST_FIGHT_TRACKER[player] += 1
        # ...         print_wrapper(f"Adding ghost fight for {player}: {ghost_opp}")
        # ...         # code.interact(local=dict(globals(), **locals()))
        # ...         break"""
        # >>> timeit.timeit(stmt=stmt, globals=globals(), number=int(25297))
        # 0.018572999999832973

        MATCH_MAKE_CALLS[player_count-1] += 1
        elapsed = timeit.default_timer() - start_time
        MATCH_MAKE_TIMES.append(elapsed)

        return matchups


    def calc_unit_loss(self, fighter_str, opponent_str, max_units):
        win_percent = fighter_str / (fighter_str + opponent_str)
        possibilities, weights = calc_fight_weights(win_percent, max_units)
        unit_loss = random.choices(possibilities, weights=weights, k=1)[0]

        # # debug
        # import matplotlib.pyplot as plt
        # debug_hist = random.choices(possibilities, weights=weights, k=int(1e5))
        # plt.hist(debug_hist, bins=len(possibilities), align='left') # Specify number of bins (optional)
        # plt.xlabel("Number of units remaining")
        # plt.ylabel("Count")
        # empirical_fight_percent = sum(i > 0 for i in debug_hist)/len(debug_hist)
        # plt.title("Debug Histogram, expected %: {:.2f}%, empirical %: {:.2f}%".format(win_percent*100, empirical_fight_percent*100))
        # plt.axis(xmin=-max_units*1.5,xmax=max_units*1.5)
        # plt.xticks(possibilities)
        # plt.show()

        return unit_loss

    def sim_fights(self, matchups):
        # assumes no real fights tie, only double open fort
        # fight winners determined by weighted odds of relative strength

        # figure out how many units are likely fielded. very rough estimates
        if self.round_num < 3: # before 2-5
            max_units = 4
        elif 3 <= self.round_num < 7:   # 2-5 to 3-1
            max_units = 5
        elif 7 <= self.round_num < 10:  # 3-2 to 3-5
            max_units = 6
        elif 10 <= self.round_num < 13: # 3-6 to 4-1
            max_units = 7
        elif 13 <= self.round_num < 21: # 4-2 to 5-3
            max_units = 8
        elif 21 <= self.round_num < 28: # 5-5 to 6-5
            max_units = 9
        else:                           # 6-6 onwards
            max_units = 10

        # process fights in random order to avoid giving p1 non-ghost preference
        player_list = list(matchups.keys())
        random.shuffle(player_list) #TODO check if this is necessary after other shuffles already

        results = {}
        for fighter in player_list:
            if fighter not in results:
                # get strengths
                fighter_str = self.players[fighter].str_series[self.round_num]
                opponent = matchups[fighter]
                opponent_str = self.players[opponent].str_series[self.round_num]

                # apply matchup multiplier
                fighter_comp_type = self.players[fighter].comp_type
                opponent_comp_type = self.players[opponent].comp_type
                multiplier = COMP_MATCHUPS[fighter_comp_type][opponent_comp_type]
                fighter_str *= multiplier

                # handle double open fort
                if opponent_str == 0 and fighter_str == 0:
                    unit_loss = 0
                elif opponent_str == 0:
                    unit_loss = max_units
                elif fighter_str == 0:
                    unit_loss = -max_units
                else:
                    # non-empty boards
                    unit_loss = self.calc_unit_loss(fighter_str, opponent_str, max_units)

                    # code.interact(local=dict(globals(), **locals()))

                results[fighter] = unit_loss
                self.players[fighter].match_history[self.round_num] = opponent
                if matchups[opponent] == fighter: # if this is non-ghost, copy result
                    results[opponent] = -1*unit_loss
                    self.players[opponent].match_history[self.round_num] = fighter

        return results

    def pretty_history(self):
        pretty_df = self.history.copy(deep=True)
        pretty_df['round'] = pd.Series(self.round_trans_table)
        pretty_df = pretty_df.set_index('round')
        pretty_df = pretty_df.dropna(how='all')
        return pretty_df

    def calc_placements(self):
        global CALC_PLACEMENT_TIMES
        start_time = timeit.default_timer()

        placements_by_player = [0] * 8

        if self.num_players_alive() > 1:
            print(f"Error in calc_placements, {self.num_players_alive()} players still alive")
        else:
            # process: iterate thru history, assigning placements as ppl die
            player_alive = [True] * NUM_PLAYERS
            current_placement = 8

            for index, row in lobby.history.iterrows():
                if player_alive.count(True) == 1:
                    placements_by_player[player_alive.index(True)] = current_placement
                    break
                else:
                    died_this_round = []
                    for player_num, hp in enumerate(row):
                        if hp <= 0 and player_alive[player_num]:
                            # player was alive, just died
                            died_this_round.append(player_num)
                            player_alive[player_num] = False

                    if len(died_this_round) == 1:
                        placements_by_player[died_this_round[0]] = current_placement
                        current_placement += -1
                    elif len(died_this_round) > 1:
                        # apply tiebreakers

                        player_processed = [False]*len(died_this_round)

                        num_loops = 0
                        while False in player_processed and num_loops < 100:
                            num_loops += 1

                            # first tiebreaker is final health
                            hp_comp = []
                            hp_comp_static = [] #this is ugly but idk how to make it cleaner
                            player_comp = []
                            for death_index, player in enumerate(died_this_round):
                                if not player_processed[death_index]:
                                    hp_comp.append(row[player])
                                    hp_comp_static.append(row[player])
                                    player_comp.append(player)
                            # print(hp_comp)
                            # if hp_comp == [-5] and died_this_round == [0, 3, 6]:
                            #     code.interact(local=dict(globals(), **locals()))
                            # loop thru lowest hp, if it's unique
                            while (len(hp_comp) > 0) and (hp_comp.count(min(hp_comp)) == 1):
                                player_num = player_comp[hp_comp.index(min(hp_comp))]
                                hp_index = died_this_round.index(player_num)
                                placements_by_player[player_num] = current_placement
                                current_placement += -1
                                player_processed[hp_index] = True
                                hp_comp.remove(min(hp_comp))
                                player_comp.remove(player_num)

                            # second tiebreaker is who had more hp previously
                            if len(hp_comp) > 0:
                                tied_players = []
                                for player in died_this_round:
                                    if row[player] == min(hp_comp):
                                        tied_players.append(player)
                                prev_hp = []
                                prev_hp_static = []
                                for player in tied_players:
                                    prev_hp.append(self.history.loc[index-1][player])
                                    prev_hp_static.append(self.history.loc[index-1][player])

                                # again, loop thru lowest hp, if it's unique
                                while (len(prev_hp) > 0) and (prev_hp.count(min(prev_hp)) == 1):
                                    player_num = tied_players[prev_hp_static.index(min(prev_hp))]
                                    placements_by_player[player_num] = current_placement
                                    current_placement += -1
                                    player_processed[died_this_round.index(player_num)] = True
                                    prev_hp.remove(min(prev_hp))

                                # third tiebreaker, who won last fight
                                    # not implementing this shit, just jump to random choice
                                if len(prev_hp) > 0:
                                    tied_players_3 = []
                                    for player in died_this_round:
                                        if self.history.loc[index-1][player] == min(prev_hp):
                                            tied_players_3.append(player)

                                    while len(tied_players_3) > 0:
                                        player_num = random.choice(tied_players_3)
                                        placements_by_player[player_num] = current_placement
                                        current_placement += -1
                                        player_processed[died_this_round.index(player_num)] = True
                                        tied_players_3.remove(player_num)

                                    # last_round_won = []
                                    # last_round_won_static = []
                                    # for player in tied_players_3:
                                    #     player_history = self.history.iloc[:index, player].diff()
                                    #     last_win = player_history.at[player_history == 0].last_valid_index()
                                    #     if not last_win:
                                    #         last_win = 0 # default if never won
                                    #     last_round_won.append(last_win)
                                    #     last_round_won_static.append(last_win)

                                    #     print("tiebreaker 3 debug")
                                    #     code.interact(local=dict(globals(), **locals()))

                            if num_loops > 50:
                                print("Error in tiebreakers...")
                                # code.interact(local=dict(globals(), **locals()))

        elapsed = timeit.default_timer() - start_time
        CALC_PLACEMENT_TIMES.append(elapsed)

        return placements_by_player

    def sim_lobby(self):
        while self.num_players_alive() > 1:
            self.sim_round()

            # if self.num_players_alive() == 8:
            #     print("timing match_make")

            #     code.interact(local=dict(globals(), **locals()))
            # >>> timeit.timeit(stmt="self.match_make()", globals=globals(), number=int(3e4))

            # old results below
            # >>> 15.856263700001364 # w/ 8 players
            # >>> 14.711585400000331 # w/ 7 players
            # >>> 12.176962800000183 # w/ 6 players
            # >>> 10.504564900000332 # w/ 5 players
            # >>> 9.349594700001035 # w/ 4 players
            # >>> 4.310208599999896 # w/ 3 players
            # >>> 3.4805369000005157 # w/ 2 players. realistically more like 0.1s w/ below distribution
            # >>> MATCH_MAKE_CALLS # for 1e3 sims
            # >>> 25232
            # distribution something like: [0, 1302, 1167, 939, 949, 1084, 1517, 18271]

        # self.history.loc[1] = [10,  1,  2,  10,  3,  4,  5,  1] #hardset to test tiebreakers
        # self.history.loc[2] = [10,  1,  2,  3,  3,  4,  5,  1] #hardset to test tiebreakers
        # self.history.loc[3] = [-1, -1, -2, -3, -3, -3, -5,  1] #hardset to test tiebreakers
        # print(self.pretty_history())

        return self.calc_placements()


# moved this func out of lobby object for functools to work better
@functools.lru_cache(maxsize=1000)
def calc_fight_weights(win_percent, max_units):
    global CALC_UNIT_LOSS_CALLS
    # figure out where to put the mean of the normal. if strengths equal, mean is zero
    # use inverted qfunc to translate fight win % into unit win numbers
    # mean = -3 + 6*fighter_str / (fighter_str + opponent_str)
    mean = -1 * np.sqrt(2)*sp.erfinv(1-2*win_percent)

    # print("win percent is: {:.2f}, mean is {:.2f}".format(win_percent, mean))

    # calculate probabilities on normal distribution
    possibilities = list(range(-max_units, max_units+1))
    possibilities.remove(0)
    weights = norm.pdf(np.linspace(-3, 3, len(possibilities)), loc=mean, scale=1)
    weights /= np.sum(weights)

    CALC_UNIT_LOSS_CALLS += 1

    return possibilities, weights

    # w/o functools:
    # >>> timeit.timeit(stmt="lobby.calc_unit_loss(1,random.choice(list(range(1,100))),8)", globals=globals(), number=int(1e5))
    # 7.25996650000161
    # >>> CALC_UNIT_LOSS_CALLS
    # 92185

    # w/ functools.lru_cache(maxsize=1000), also @functools.lru_cache(maxsize=10000)
    # >>> timeit.timeit(stmt="lobby.calc_unit_loss(1,random.choice(list(range(1,100))),8)", globals=globals(), number=int(1e5))
    # 0.32586409999930765
    # >>> CALC_UNIT_LOSS_CALLS
    # 25699

    # w/ functools.lru_cache(maxsize=10)
    # timeit.timeit(stmt="lobby.calc_unit_loss(1,random.choice(list(range(1,100))),8)", globals=globals(), number=int(1e5))
    # 5.546505299993441
    # >>> CALC_UNIT_LOSS_CALLS
    # 25747

    # also moving this function outside of lobby object doesn't appear to make it any faster? but does massively decrease CALC_UNIT_LOSS_CALLS, so something's off

# statistics wrapper functions for bootstrap ci
def top4_rate(x, axis=-1):
    return np.count_nonzero(x <= 4, axis=axis) / NUM_SIMS * 100

def top1_rate(x, axis=-1):
    return np.count_nonzero(x == 1, axis=axis) / NUM_SIMS * 100

def top8_rate(x, axis=-1):
    return np.count_nonzero(x == 8, axis=axis) / NUM_SIMS * 100

if __name__ == "__main__":
    
    scaling_results = []
    scaling_results_str = []
    scale_factors = [0, 0.1111111111, 0.25, 0.4285714286, 0.6666666667, 1, 1.5, 2.333333333, 4, 9, 19]
    # scale_factors = [4] # debugging

    timer_wrapper.start_times = [timeit.default_timer()]

    for scale_factor in scale_factors:

        if USING_CSV:
            print(f"Using {INPUT_FILENAME} as input...")
            with open(INPUT_FILENAME, "r", encoding="utf-8") as f_in:
                df_in = pd.read_csv(f_in)
        else:
            print("Using hardcoded df_in as input, scale_factor: {:.2f}".format(scale_factor))
            data = {'round_num': list(range(36)), 
                    'round_name': ["'2-1", "'2-2", "'2-3", "'2-5", "'2-6", "'2-7", "'3-1", "'3-2", "'3-3", "'3-5", "'3-6", "'3-7", "'4-1", "'4-2", "'4-3", "'4-5", "'4-6", "'4-7", "'5-1", "'5-2", "'5-3", "'5-5", "'5-6", "'5-7", "'6-1", "'6-2", "'6-3", "'6-5", "'6-6", "'6-7", "'7-1", "'7-2", "'7-3", "'7-5", "'7-6", "'7-7"],
                    
                    # # full random
                    # 'p1': list(range(1, 37)), 
                    # 'p2': list(range(1, 37)), 
                    # 'p3': list(range(1, 37)), 
                    # 'p4': list(range(1, 37)), 
                    # 'p5': list(range(1, 37)), 
                    # 'p6': list(range(1, 37)), 
                    # 'p7': list(range(1, 37)), 
                    # 'p8': list(range(1, 37))

                    # #debugging matchmaking
                    # 'p1': list(range(1, 37)), 
                    # 'p2': list(range(1, 37)), 
                    # 'p3': list(range(1, 37)), 
                    # 'p4': list(range(1, 37)), 
                    # 'p5': [0]*36, 
                    # 'p6': [0]*36, 
                    # 'p7': [0]*36, 
                    # 'p8': [0]*36

                    # real player profiles
                    'p1': list(range(1, 37)), 
                    'p2': list(range(1, 37)), 
                    'p3': list(range(1, 37)), 
                    'p4': list(range(1, 37)), 
                    'p5': list(range(1, 37)), 
                    'p6': list(range(1, 37)), 
                    # 'p7': list(range(1, 37)), 
                    'p7': [scale_factor*x for x in list(range(1, 13))] + list(range(13, 37)), # winstreak early 
                    # 'p7': [0 if x < 6 else x*0.25 for x in list(range(1, 14))] + [scale_factor*x for x in list(range(14, 37))], # losestreak early
                    # 'p7': [3*x if x < 19 else 2*x for x in list(range(1, 37))], # highrolling streaker up until 5-1, then moderately strong
                    'p8': [scale_factor*x for x in list(range(1, 13))] + list(range(13, 37)) # winstreak early
                    # 'p8': [0 if x < 6 else x*0.25 for x in list(range(1, 14))] + [scale_factor*x for x in list(range(14, 37))] # losestreak early
                    }
            df_in = pd.DataFrame(data)

        timer_wrapper("initialize input data")

        # code.interact(local=dict(globals(), **locals()))

        # simulate lobbies
        lobby_results = [None]*NUM_SIMS
        sim_count = 0
        for i in range(NUM_SIMS):
            lobby = Lobby(df_in.copy(deep=True))
            # print(lobby)
            # code.interact(local=dict(globals(), **locals()))
            # >>> timeit.timeit(stmt="Lobby(df_in.copy(deep=True))", globals=globals(), number=int(NUM_SIMS))
            # 0.7410564000001614

            placements = lobby.sim_lobby()
            lobby_results[i] = placements
            # print(placements)

            sim_count += 1
            if sim_count % (NUM_SIMS / 10) == 0:
                print("{} / {:.1E} lobbies simulated".format(sim_count, NUM_SIMS))

        timer_wrapper("simulate {:.1E} lobbies".format(NUM_SIMS))

        # calculate AVPs:
        results_df = pd.DataFrame(lobby_results, columns=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])
        print("AVP / Top 4 % / Win % / 8th %")
        for i in range(1, NUM_PLAYERS+1):

            if DEBUG_PRINT or i == 8:

                p_name = f"p{i}"
                x = results_df[p_name]

                # code.interact(local=dict(globals(), **locals()))

                avp = x.mean()
                avp_res = bootstrap((x,), np.mean, confidence_level=0.95, vectorized=True)
                avp_ci = (avp_res.confidence_interval[1] - avp_res.confidence_interval[0])/2 # high minus low

                top4 = np.count_nonzero(x <= 4) / NUM_SIMS * 100
                top4_res = bootstrap((x,), top4_rate, confidence_level=0.95, vectorized=True)
                top4_ci = (top4_res.confidence_interval[1] - top4_res.confidence_interval[0])/2 # high minus low

                top1 = np.count_nonzero(x == 1) / NUM_SIMS * 100
                top1_res = bootstrap((x,), top1_rate, confidence_level=0.95, vectorized=True)
                top1_ci = (top1_res.confidence_interval[1] - top1_res.confidence_interval[0])/2 # high minus low

                top8 = np.count_nonzero(x == 8) / NUM_SIMS * 100
                top8_res = bootstrap((x,), top8_rate, confidence_level=0.95, vectorized=True)
                top8_ci = (top8_res.confidence_interval[1] - top8_res.confidence_interval[0])/2 # high minus low

                str_out = ("{}: {:.2f} ± {:.2f} / {:.1f} ± {:.1f} % / {:.1f} ± {:.1f} % / {:.1f} ± {:.1f} %"
                          .format(p_name, avp, avp_ci, top4, top4_ci, top1, top1_ci, top8, top8_ci))
                print(str_out)

                if i == 8:
                    scaling_results.append(avp)
                    scaling_results_str.append(str_out)

            # timer_wrapper(f"calculate statistics for p{i}")
        timer_wrapper("calculate statistics")

    for i in range(len(scale_factors)):
        print("{:.1f}%, {:.2f}, {}".format(scale_factors[i]/(scale_factors[i]+1)*100, scaling_results[i], scaling_results_str[i]))

    stop = timeit.default_timer()
    total_runtime = stop - timer_wrapper.start_times[0]
    print('Total runtime: {:.2f} seconds'.format(total_runtime))
    print("")

    # example timing debug
    # timeit.timeit(stmt="bootstrap((x,), top8_rate, confidence_level=0.95, vectorized=True)",number=150,globals={**globals(), **locals()})

    # code.interact(local=dict(globals(), **locals()))
