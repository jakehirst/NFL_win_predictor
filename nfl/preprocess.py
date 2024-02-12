import pandas as pd
import numpy as np
import torch

'''This will help to turn the data into data that is suitable for the lstm'''


'''
Turns dashed columns into two separate columns with floats (or ints)
columns that have things like penalties-pentalty yards, Comp-Att, or 
possetion time need to be turned into multiple columns or floats to be usable.
'''
def clean_data(raw_data_filepath):
    df = pd.read_excel(raw_data_filepath)
    # df = df.dropna() #have to get rid of nans rows for this to work.
    column_means = df.mean(axis=0)
    df.fillna(column_means, inplace=True) #replacing all nans with the mean of the column
        
    #dictionary that contains all of the current column names as keys. Label is the two columns to replace it with.
    columns_with_dashes = {'3rd down efficiency': ['3rd down conv', '3rd down att'],
                           '4th down efficiency': ['4th down conv', '4th down att'],
                           'Comp-Att': ['Pass comp', 'Pass att'],
                           'Sacks-Yards Lost': ['Opponent sacks', 'Yds lost on sacks'],
                           'Red Zone (Made-Att)': ['Redzn Conv', 'Redzn Att'],
                           'Penalties': ['Tot penalties', 'Penalty Yds'],
                           }
    
    new_df = df.copy()

    for col in columns_with_dashes.keys():
        #replace the dashed column with two separate columns
        new_df[[columns_with_dashes[col][0], columns_with_dashes[col][1]]] = new_df[col].str.split('-', expand=True)
        #turn those columns into numbers (int or float) instead of strings
        new_df[columns_with_dashes[col][0]] = pd.to_numeric(new_df[columns_with_dashes[col][0]])
        new_df[columns_with_dashes[col][1]] = pd.to_numeric(new_df[columns_with_dashes[col][1]])
        #delete the original column
        new_df = new_df.drop(columns=[col])

    '''Time of possession needs to be handled differently.'''
    minutes, seconds = new_df['Possession'].str.split(':', expand=True).astype(int).values.T
    #Convert minutes to seconds
    total_seconds = minutes * 60 + seconds
    #Replace the 'Possession' column with the total seconds
    new_df['TOP(sec)'] = total_seconds
    #Delete the original time of possesion column
    new_df = new_df.drop(columns=['Possession'])
    
    #move the win column to the far right for convenience
    columns = [col for col in new_df.columns if col != 'Win(1) / Loss(0)']
    columns.append('Win(1) / Loss(0)')
    new_df = new_df[columns]
    
    '''this section is genious'''
    #Changing all the names of the teams to be just the mascot name. 
    #This helps the case when a team moves locations (St. Louis Rams vs LA Rams)
    # Simplify the 'teams' column to just the mascot (the last word)
    new_df['Team name'] = new_df['Team name'].str.split().str[-1]
    new_df['Opponent name'] = new_df['Opponent name'].str.split().str[-1]
    
    #we have to deal with the special case where at one point 'Washington' did not have a mascot.
    new_df['Team name'] = new_df['Team name'].replace(['Redskins', 'Commanders'], 'Washington')
    new_df['Opponent name'] = new_df['Opponent name'].replace(['Redskins', 'Commanders'], 'Washington')
    '''this section is genious'''

    
    old_filename = raw_data_filepath.split('/')[-1]
    new_filename = old_filename.removesuffix('.xlsx') + '_cleaned.csv'
    new_filepath = raw_data_filepath.removesuffix(old_filename) + new_filename
    #save the cleaned dataframe
    # new_df.to_excel(new_filepath, index=False, engine='openpyxl')
    new_df.to_csv(new_filepath, index=False)
    
    return new_df 

'''
given a week and a team, get n games prior to that week
returns the last n games from that team in a pandas dataframe.

week: the week of the label (game at hand)
year: year of the label
n: number of games to get before this week
data: cleaned dataframe 
'''
def get_past_n_games(data, week, year, team, n):
    #in case the data is not sorted yet, sort by year, then week
    data = data.sort_values(by=['Year', 'Week'])
    #then just look at this team's games
    team_games = data[data['Team name'] == team]
    # Filter games that occurred before the inputted week and year
    past_games = team_games[(team_games['Year'] < year) | 
                            ((team_games['Year'] == year) & (team_games['Week'] < week))]    
    # Get the last n games
    past_n_games = past_games.tail(n)
    return past_n_games


'''
organizes data into batches for an RNN or LSTM to learn from. does not split into training and testing.

data: should be cleaned
n: number of previous games per batch.
'''
def batch_all_data_for_RNN(cleaned_data_filepath, n):
    data = pd.read_csv(cleaned_data_filepath)
    #assume the data is not sorted and sort it real quick.
    sorted_data = data.sort_values(by=['Year', 'Week'])
    #have to start at least n+1 weeks into the data...
    # condition = (sorted_data['Year'] == sorted_data.iloc[0]['Year']) & (sorted_data['Week'] == n+1)
    # start_week_row_num = sorted_data[condition].index.min() if condition.any() else None #the iloc of the first week we will start on.
    # Filter the DataFrame where 'week' column is n+1 and 'year' column is 2002
    filtered_df = sorted_data[(sorted_data['Week'] == n + 2) & (sorted_data['Year'] == sorted_data.iloc[0]['Year'])]
    # Get the index of the top row in the filtered DataFrame
    if not filtered_df.empty:
        start_week_row_num = filtered_df.index[0]
        print(f"The row number (index) of the top row is: {start_week_row_num}")
    else:
        print("No rows found matching the criteria.")
    
    features = []
    labels = []
    game_tracker = []
    
    i = 0
    for game_num in range(start_week_row_num, len(sorted_data)-1): #TODO might have to go by 2s since the games repeat.
        team_0, team_1 = sorted_data.iloc[game_num]['Team name'], sorted_data.iloc[game_num]['Opponent name']
        week = sorted_data.iloc[game_num]['Week']
        year = sorted_data.iloc[game_num]['Year']
        
        team_0_games = get_past_n_games(sorted_data, week, year, team_0, n)
        team_1_games = get_past_n_games(sorted_data, week, year, team_1, n)
        team_1_games.columns = ['(Opp)' + col for col in team_1_games.columns]
        
        #TODO have to deal with when one of the teams switches locations... LA rams do not have any previous games before 2016...
        if(len(team_0_games) != n or len(team_1_games) != n):
            print(f'Dont have {n} previous games for either home or away team')

        #append the stats from the games of team 1 to the games of team 0...
        both_team_games = pd.concat([team_0_games.reset_index(drop=True), team_1_games.reset_index(drop=True)], axis=1)
        #get rid of things we dont want to use as features
        cols_to_get_rid_of = ['Team name', 'Win(1) / Loss(0)', '(Opp)Year', '(Opp)Week', 'Opponent name']
        for the_col in cols_to_get_rid_of:
            cols_to_remove = [col for col in both_team_games.columns if the_col in col]
            # Remove the columns from the DataFrame
            both_team_games = both_team_games.drop(cols_to_remove, axis=1)
        
        #get the current game that is associated with the label
        all_team_games = sorted_data[sorted_data['Team name'] == team_0]
        # Filter games that occurred before the inputted week and year
        current_game = all_team_games[((all_team_games['Year'] == year) & (all_team_games['Week'] == week))]   
        
        
        #which team is the winner? This will be the label. 0 is for team 0, 1 is for team 1.
        winner = ((sorted_data[sorted_data['Team name'] == team_1][(sorted_data['Week'] == week) & (sorted_data['Year'] == year)])['Win(1) / Loss(0)'])._values[0]
        
        features.append(both_team_games.to_numpy())
        labels.append(winner)
        game_tracker.append(f'Week {week} {year}, team0 ({team_0}) vs team1 ({team_1}): winner = team {winner} , score = {current_game["Points scored"].iloc[0]} - {current_game["Points allowed"].iloc[0]}')
        print(f'Week {week} {year}, team0 ({team_0}) vs team1 ({team_1}): winner = team {winner} , score = {current_game["Points scored"].iloc[0]} - {current_game["Points allowed"].iloc[0]}')
        print(i)
        i+=1
        
        
        

    
    X = np.array(features)
    Y = np.array(labels)
    
    saving_folder_path = cleaned_data_filepath.removesuffix(cleaned_data_filepath.split('/')[-1])
    saving_name = cleaned_data_filepath.split('/')[-1].removesuffix('_cleaned.csv')
    
    np.save(f'{saving_folder_path}features_{saving_name}', X)
    np.save(f'{saving_folder_path}labels_{saving_name}', Y)
    np.save(f'{saving_folder_path}game_log{saving_name}', np.array(game_tracker))
    
    # np.save('/Users/jakehirst/Desktop/sportsbetting/nfl/data/features.npy', X)
    # np.save('/Users/jakehirst/Desktop/sportsbetting/nfl/data/labels.npy', Y)
    # np.save('/Users/jakehirst/Desktop/sportsbetting/nfl/data/game_log.npy', np.array(game_tracker))
    return

# raw_data_filepath = '/Users/jakehirst/Desktop/sportsbetting/nfl/data/Stats_from_2002_thru_2022.xlsx'
raw_data_filepath = '/Users/jakehirst/Desktop/sportsbetting/nfl/data/Stats_from_2023_thru_2024.xlsx'
clean_data(raw_data_filepath)

# cleaned_data_filepath = '/Users/jakehirst/Desktop/sportsbetting/nfl/data/Stats_from_2002_thru_2022_cleaned.csv'
cleaned_data_filepath = '/Users/jakehirst/Desktop/sportsbetting/nfl/data/Stats_from_2023_thru_2024_cleaned.csv'

batch_all_data_for_RNN(cleaned_data_filepath, 10)
