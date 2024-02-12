from cmath import nan
import requests
from bs4 import BeautifulSoup
import pandas as pd
import xlrd
import numpy as np
import json

#TODO: must use sportsbetting_venv interpreter to run this code.

# getting all of the players that will be starting in the upcoming season.
#COMMENT--> base url: https://www.espn.com/nfl/scoreboard/_/week/1/year/2002/seasontype/2

'''This gets the statistics of the specific year'''
# def GetTeamStats(year, url):
#     headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
#     page = requests.get(url, headers=headers, timeout=10)
#     soup = BeautifulSoup(page.content.decode(), "html.parser")
    
#     mydivs = soup.find_all("div", {"class": "Table__Title"})
#     print(mydivs[0])
#     print("done")

'''
gets the game stats of all the games in a given week through webscraping with BeautifulSoup
url: url of the current week's games (e.g. https://www.espn.com/nfl/scoreboard/_/week/1/year/2002) 
'''
def GetCurrentWeekStats(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    page = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(page.content.decode(), "html.parser")
    
    #getting all of the "GAMECAST" links
    mydivs = soup.find_all("div", {"class": "Scoreboard__Callouts flex items-center mv4 flex-column"})
    week = url.split('/')[-3]
    year = url.split('/')[-1]
    weekly_game_stats = None #start with nothing
    
    gamenum = 1
    #going through all of the games in the week and getting the stats from each game
    for div in mydivs:
        print(f'Year {year} Week {week} game {gamenum}')
        gamenum+=1
        if(div == mydivs[len(mydivs) - 1]):
            print('here')
        game_link = "https://www.espn.com" + str(div).split("href")[1].split("\"")[1]
        game_ID = game_link.split("/")[-2]
        # print(game_ID)
        game_page = requests.get(game_link, headers=headers, timeout=10)
        game_soup = BeautifulSoup(page.content.decode(), "html.parser")
        gamestats = get_game_stats(game_page, game_soup, game_ID, headers)
        
        keys = list(gamestats.keys())
        #if we have no stats yet, then define the column names first
        if(not isinstance(weekly_game_stats, pd.DataFrame)):
            #TODO add home or away as well (later)
            stat_names = ['Year' , 'Week', 'Team name', 'Opponent name'] + gamestats['stat names']
            weekly_game_stats = pd.DataFrame(columns=stat_names)
        
        #adding year, week, and team names
        team1_stats = [year, week, keys[1], keys[2]] + gamestats[keys[1]]
        team2_stats = [year, week, keys[2], keys[1]] + gamestats[keys[2]]
        
        #adding this game's stats to the weekly_game_stats
        weekly_game_stats.loc[len(weekly_game_stats)] = team1_stats
        weekly_game_stats.loc[len(weekly_game_stats)] = team2_stats
        
    return weekly_game_stats

'''
helper function for GetWeekly_stats_ALLYEARS(). Just gets a single year's stats.
'''
def GetYearGameStats(generic_url, year):
    Weekly_stats_for_whole_year = None
    for week in range(1,18):
        print(f'*** Week {week} ***')
        url = generic_url + f"week/{week}/year/{year}"
        try:
            week_game_stats = GetCurrentWeekStats(url)
            if(not isinstance(Weekly_stats_for_whole_year, pd.DataFrame)):
                Weekly_stats_for_whole_year = pd.DataFrame(columns=list(week_game_stats.columns))
        
            #adding the rows of week_game_stats to Stats_for_whole_year
            Weekly_stats_for_whole_year = pd.concat([Weekly_stats_for_whole_year, week_game_stats], ignore_index=True)
            
        except:
            print('this week didnt work')
            continue
        
    return Weekly_stats_for_whole_year

'''
gets the stats of all games from the start_year to the end_year

generic_url: url of the score board on espn. (e.g. https://www.espn.com/nfl/scoreboard/_/)
'''
def GetWeekly_stats_ALLYEARS(generic_url, start_year, end_year):
    Complete_stats = None
    for year in range(start_year, end_year+1):
        print(f"********************** YEAR {year} ************************")
        year_stats = GetYearGameStats(generic_url, year)
        if(not isinstance(Complete_stats, pd.DataFrame)):
            Complete_stats = pd.DataFrame(columns=list(year_stats.columns))
        Complete_stats = pd.concat([Complete_stats, year_stats], ignore_index=True)

    return Complete_stats

'''
gets the stats from a certain specific game. Returns a dictionary with the stat names, home team, and away team stats.

The game webpage looks something like this --> https://www.espn.com/nfl/matchup/_/gameId/220905019

game_ID: 9 digit number which is the ID of a specific game, getting us to the above webpage.
headers: The User-Agent to make requests to a website easier for the scraper. It is sort of a polite way of saying that we are 
scraping this website in a polite and ethical way. 
'''
def get_game_stats(game_page, game_soup, game_ID, headers):
    game_teamstats_link = "https://www.espn.com/nfl/matchup/_/gameId/" + game_ID
    
    # print(game_teamstats_link) #this is where we get the stats for a given game
    game_teamstats_page = requests.get(game_teamstats_link, headers=headers, timeout=10)
    teamstats_soup = BeautifulSoup(game_teamstats_page.content.decode(), "html.parser")
    left_team, right_team, has_teams = find_teams(teamstats_soup)
    if(has_teams):
        stat_names = []
        left_team_stats = []
        right_team_stats = []
        
        #stats includes many generic stats of this game like first downs, total plays, TOP, etc.
        stats = teamstats_soup.find_all("tr", {"class": "Table__TR Table__TR--sm Table__even"})
        #scores contains the div class instances that have the scores of the football games
        scores = teamstats_soup.find_all("div", {"class": "Gamestrip__Score relative tc w-100 fw-heavy h2 clr-gray-01"})
        for i in range(2, 27):
            try:
                stat_names.append(str(stats[i].next.next))
                left_team_stats.append(str(stats[i].next.next.next.next))
                right_team_stats.append(str(stats[i].next.next.next.next.next.next))
            except:
                stat_names.append("N/A")
                left_team_stats.append("N/A")
                right_team_stats.append("N/A")
        game_stats = {}
        game_stats["stat names"] = stat_names
        
        #getting these stats (the score of the game and win/loss) to the game stats
        score_stat_names = ['Points scored', 'Points allowed', 'Win(1) / Loss(0)']
        score_stats_left = [scores[0].next, scores[1].next, int(int(scores[0].next) > int(scores[1].next))]
        score_stats_right = [scores[1].next, scores[0].next, int(int(scores[0].next) < int(scores[1].next))]
        
        game_stats[left_team] = left_team_stats
        game_stats[right_team] = right_team_stats
        
        #now adding on the score stats
        game_stats[left_team].extend(score_stats_left)
        game_stats[right_team].extend(score_stats_right)
        game_stats["stat names"].extend(score_stat_names)
        return game_stats
    else:
        return None

'''
gets the team names from the teamstats_soup
'''
def find_teams(teamstats_soup):
    has_teams = teamstats_soup.find_all("div", {"class": "Gamestrip__TeamContent flex tc w-100"})
    if(len(has_teams) == 0): 
        return None, None, False
    left_team = str(has_teams[0].next.next.next.next.next.next).split(">")[-2].split("<")[-2]
    right_team = str(has_teams[1].next.next.next.next.next.next).split(">")[-2].split("<")[-2]
    # print("TEAMS: " + left_team + " and " + right_team)
    return left_team, right_team, True




#GetAllQBStats(2022, "https://opensource.com/article/21/9/web-scraping-python-beautiful-soup")

# GetCurrentWeekStats("https://www.espn.com/nfl/scoreboard/_/week/1/year/2002")
#GetTeamStats(2022, "https://www.espn.com/nfl/team/_/name/buf/buffalo-bills")
# generic_url = "https://www.espn.com/nfl/scoreboard/_/"
# GetYearGameStats(generic_url, 2000)

''' example of how to get and save the stats from 2002 to 2020 '''
# yearly_stats = GetWeekly_stats_ALLYEARS(generic_url, 2002, 2020)
# save_stats(yearly_stats, "/Users/jakehirst/Desktop/sportsbetting/nfl/Stats_from_2002_thru_2020.json")

# stats_filepath = "/Users/jakehirst/Desktop/sportsbetting/nfl/Stats_from_2002_thru_2020.json"
# turn_json_into_dataframe("/Users/jakehirst/Desktop/sportsbetting/nfl/Stats_from_2002_thru_2020.json")
# current_stats = load_stats(stats_filepath)