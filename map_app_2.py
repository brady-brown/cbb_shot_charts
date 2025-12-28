"""
College Basketball Shot Chart Web Application

This Flask application provides a web interface to view shot charts for college basketball games.
It loads play-by-play and box score data from the sportsdataverse package, allowing users to:
1. Select a team
2. Choose a game from that team's schedule
3. Pick a player from that game
4. View a matplotlib-generated shot chart showing all of that player's shot attempts

Run this with: python app.py
Then open http://localhost:5000 in your browser
"""

# ============================================================================
# IMPORTS
# ============================================================================

from flask import Flask, jsonify
import sportsdataverse.mbb as mbb  # College basketball data from sportsdataverse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (required for Flask - no GUI windows)
import matplotlib.pyplot as plt
from mplbasketball import Court  # Library to draw basketball court diagrams
import io  # For handling image data in memory
import base64  # For encoding images to send to browser

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)

# Global variables to store loaded data (loaded once at startup for performance)
# These are accessible by all routes without reloading data for each request
pbp_df = None  # Play-by-play DataFrame - contains all shot locations and game events
box_df = None  # Box score DataFrame - contains player stats and names
all_teams = None  # DataFrame of all teams in the dataset

# Manual conference mappings for major D1 conferences
# This is a fallback in case the API doesn't provide conference data
CONFERENCE_MAP = {
    # ACC
    'Boston College': 'ACC', 'California': 'ACC', 'Clemson': 'ACC', 'Duke': 'ACC',
    'Florida State': 'ACC', 'Georgia Tech': 'ACC', 'Louisville': 'ACC', 'Miami': 'ACC',
    'North Carolina': 'ACC', 'NC State': 'ACC', 'Notre Dame': 'ACC', 'Pittsburgh': 'ACC',
    'SMU': 'ACC', 'Stanford': 'ACC', 'Syracuse': 'ACC', 'Virginia': 'ACC',
    'Virginia Tech': 'ACC', 'Wake Forest': 'ACC',
    
    # Big Ten
    'Illinois': 'Big Ten', 'Indiana': 'Big Ten', 'Iowa': 'Big Ten', 'Maryland': 'Big Ten',
    'Michigan': 'Big Ten', 'Michigan State': 'Big Ten', 'Minnesota': 'Big Ten', 'Nebraska': 'Big Ten',
    'Northwestern': 'Big Ten', 'Ohio State': 'Big Ten', 'Oregon': 'Big Ten', 'Penn State': 'Big Ten',
    'Purdue': 'Big Ten', 'Rutgers': 'Big Ten', 'UCLA': 'Big Ten', 'USC': 'Big Ten',
    'Washington': 'Big Ten', 'Wisconsin': 'Big Ten',
    
    # Big 12
    'Arizona': 'Big 12', 'Arizona State': 'Big 12', 'Baylor': 'Big 12', 'BYU': 'Big 12',
    'UCF': 'Big 12', 'Cincinnati': 'Big 12', 'Colorado': 'Big 12', 'Houston': 'Big 12',
    'Iowa State': 'Big 12', 'Kansas': 'Big 12', 'Kansas State': 'Big 12', 'Oklahoma State': 'Big 12',
    'TCU': 'Big 12', 'Texas Tech': 'Big 12', 'Utah': 'Big 12', 'West Virginia': 'Big 12',
    
    # SEC
    'Alabama': 'SEC', 'Arkansas': 'SEC', 'Auburn': 'SEC', 'Florida': 'SEC',
    'Georgia': 'SEC', 'Kentucky': 'SEC', 'LSU': 'SEC', 'Ole Miss': 'SEC',
    'Mississippi State': 'SEC', 'Missouri': 'SEC', 'Oklahoma': 'SEC', 'South Carolina': 'SEC',
    'Tennessee': 'SEC', 'Texas': 'SEC', 'Texas A&M': 'SEC', 'Vanderbilt': 'SEC',
    
    # Big East
    'Butler': 'Big East', 'Creighton': 'Big East', 'Connecticut': 'Big East', 'DePaul': 'Big East',
    'Georgetown': 'Big East', 'Marquette': 'Big East', 'Providence': 'Big East', 'Seton Hall': 'Big East',
    "St. John's": 'Big East', 'Villanova': 'Big East', 'Xavier': 'Big East',
    
    # American Athletic
    'Charlotte': 'American', 'East Carolina': 'American', 'Florida Atlantic': 'American', 
    'Memphis': 'American', 'Navy': 'American', 'North Texas': 'American', 'Rice': 'American',
    'South Florida': 'American', 'Temple': 'American', 'Tulane': 'American', 'Tulsa': 'American',
    'UAB': 'American', 'UTSA': 'American',
    
    # Mountain West
    'Air Force': 'Mountain West', 'Boise State': 'Mountain West', 'Colorado State': 'Mountain West',
    'Fresno State': 'Mountain West', 'Nevada': 'Mountain West', 'New Mexico': 'Mountain West',
    'San Diego State': 'Mountain West', 'San Jose State': 'Mountain West', 'UNLV': 'Mountain West',
    'Utah State': 'Mountain West', 'Wyoming': 'Mountain West',
    
    # West Coast Conference
    'Gonzaga': 'WCC', 'Saint Mary\'s': 'WCC', 'San Francisco': 'WCC', 'Santa Clara': 'WCC',
    'Loyola Marymount': 'WCC', 'Pacific': 'WCC', 'Pepperdine': 'WCC', 'Portland': 'WCC',
    'San Diego': 'WCC',
    
    # Atlantic 10
    'Davidson': 'Atlantic 10', 'Dayton': 'Atlantic 10', 'Duquesne': 'Atlantic 10', 'Fordham': 'Atlantic 10',
    'George Mason': 'Atlantic 10', 'George Washington': 'Atlantic 10', 'La Salle': 'Atlantic 10',
    'Loyola Chicago': 'Atlantic 10', 'Massachusetts': 'Atlantic 10', 'Rhode Island': 'Atlantic 10',
    'Richmond': 'Atlantic 10', 'Saint Joseph\'s': 'Atlantic 10', 'Saint Louis': 'Atlantic 10',
    'St. Bonaventure': 'Atlantic 10', 'VCU': 'Atlantic 10',
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_data():
    """
    Load the season data once at startup.
    
    This function:
    1. Loads play-by-play data for the 2025-26 season (stored as 2026 in the API)
    2. Loads box score data for player names and stats
    3. Extracts all unique teams from both home and away games
    4. Tries to extract conference information from available columns
    5. Stores everything in global variables for fast access
    
    Called once when the Flask app starts, not on every request.
    """
    global pbp_df, box_df, all_teams
    
    print("Loading 2025-26 season data...")
    
    # Load play-by-play data - contains every event in every game including shot coordinates
    # seasons=[2026] refers to the 2025-26 season (API uses end year)
    pbp_df = mbb.load_mbb_pbp(seasons=[2026], return_as_pandas=True)
    
    # Load box score data - contains player names, IDs, and game statistics
    box_df = mbb.load_mbb_player_boxscore(seasons=[2026], return_as_pandas=True)
    
    print("Processing team and conference data...")
    
    # Extract all unique teams from the play-by-play data
    # Get home teams
    home_teams = pbp_df[['home_team_id', 'home_team_name']].rename(
        columns={'home_team_id': 'team_id', 'home_team_name': 'team_name'}
    )
    
    # Get away teams
    away_teams = pbp_df[['away_team_id', 'away_team_name']].rename(
        columns={'away_team_id': 'team_id', 'away_team_name': 'team_name'}
    )
    
    # Combine home and away teams, remove duplicates
    all_teams = pd.concat([home_teams, away_teams]).drop_duplicates().dropna()
    
    # Map conferences using our manual dictionary
    all_teams['conference'] = all_teams['team_name'].map(CONFERENCE_MAP)
    
    # For teams not in our dictionary, mark as "Other"
    all_teams['conference'] = all_teams['conference'].fillna('Other')
    
    # Sort by team name
    all_teams = all_teams.sort_values('team_name').reset_index(drop=True)
    
    print("Data loaded successfully!")
    print(f"Loaded {len(all_teams)} teams")
    
    # Print conference distribution
    conf_counts = all_teams['conference'].value_counts()
    print(f"Conference distribution:")
    for conf, count in conf_counts.head(10).items():
        print(f"  {conf}: {count} teams")

def get_team_games(team_id):
    """
    Get all games for a specific team.
    
    Args:
        team_id (int): The unique identifier for the team
        
    Returns:
        DataFrame: Contains game information including dates, opponents, scores
        
    Process:
    1. Filter pbp_df to only rows where this team played (either home or away)
    2. Group by game_id to get one row per game
    3. Use .last() to get final scores (first row would be 0-0)
    4. Sort by date and reset index so games are numbered 1, 2, 3...
    """
    # Filter to only games where this team played (either as home or away team)
    team_games = pbp_df[
        (pbp_df['home_team_id'] == team_id) | 
        (pbp_df['away_team_id'] == team_id)
    ].copy()
    
    # Group by game_id to collapse multiple rows per game into one row
    # Use .last() instead of .first() to get FINAL scores (not starting 0-0)
    # Select relevant columns: game details, team names, scores
    game_info = team_games.groupby('game_id').last()[
        ['game_date', 'home_team_name', 'away_team_name', 'home_score', 'away_score', 
         'home_team_id', 'away_team_id']
    ].reset_index()
    
    # Sort by date (chronological order) and reset index
    # reset_index(drop=True) ensures games are numbered 1, 2, 3... not using original DataFrame indices
    game_info = game_info.sort_values('game_date').reset_index(drop=True)
    
    return game_info

def get_player_shots(game_id, team_id):
    """
    Get shot data for a specific game and team.
    
    Args:
        game_id (int): Unique identifier for the game
        team_id (int): Unique identifier for the team
        
    Returns:
        tuple: (team_shots DataFrame, team_players DataFrame)
            - team_shots: All field goal attempts with coordinates and results
            - team_players: All players who played in the game (even if no shots)
    
    Process:
    1. Get player names from box score for this specific game
    2. Merge names into play-by-play data
    3. Filter to only this team's field goal attempts (excluding free throws)
    4. Also return list of all players for selection menu
    """
    # Get box score data for this specific game only
    game_box = box_df[box_df['game_id'] == game_id].copy()
    
    # Extract player names and IDs, remove duplicates
    names = game_box[['athlete_id', 'athlete_display_name']].drop_duplicates()
    names['athlete_id'] = names['athlete_id'].astype(float)
    
    # Get play-by-play data for this game
    game_shots = pbp_df[pbp_df['game_id'] == game_id].copy()
    
    # Merge player names into the play-by-play data
    # athlete_id_1 in pbp_df corresponds to the player taking the action
    game_shots = game_shots.merge(names, left_on='athlete_id_1', right_on='athlete_id', how='left')
    
    # Filter to get only field goal attempts for this team
    # Conditions:
    # 1. team_id matches our team
    # 2. shooting_play == True (it's a shot attempt)
    # 3. NOT a free throw (we only want field goals for the shot chart)
    team_shots = game_shots[
        (game_shots['team_id'] == team_id) & 
        (game_shots['shooting_play'] == True) &
        (~game_shots['type_text'].str.contains('FreeThrow', case=False, na=False))
    ].copy()
    
    # Get all players from this team in this game (for the selection menu)
    # This includes players who didn't take any shots
    team_players = game_box[game_box['team_id'] == team_id][['athlete_id', 'athlete_display_name']].drop_duplicates()
    
    return team_shots, team_players

def create_shot_chart(player_shots, player_name, game_info, team_id):
    """
    Create a shot chart using matplotlib and return as base64-encoded image.
    
    Args:
        player_shots (DataFrame): Shot attempts with coordinates and results
        player_name (str): Name of the player (or 'Team View' for all players)
        game_info (Series): Game details (opponent, date, score)
        team_id (int): Team identifier to determine home/away
        
    Returns:
        str: Base64-encoded PNG image that can be embedded in HTML
        
    Process:
    1. Draw basketball court using mplbasketball
    2. Plot made shots as green circles, missed shots as red X's
    3. Create descriptive title with opponent and score
    4. Convert matplotlib figure to PNG and encode as base64
    """
    # Create NCAA basketball court using mplbasketball library
    # origin="center" means (0,0) is at center court
    # units="ft" means coordinates are in feet
    court = Court(court_type="ncaa", origin="center", units="ft")
    
    # Draw the court and get figure and axis objects
    # orientation="h" means horizontal (full court side-by-side)
    fig, ax = court.draw(orientation="h")
    fig.set_size_inches(14, 10)  # Set figure size
    
    # Split shots into makes and misses based on scoring_play column
    makes = player_shots[player_shots['scoring_play'] == True]
    misses = player_shots[player_shots['scoring_play'] == False]
    
    # Plot made shots as green circles
    # s=120: marker size
    # edgecolors='white': white border around each dot
    # zorder=3: draw on top of court lines
    # alpha=0.8: slightly transparent
    ax.scatter(makes['coordinate_x'], makes['coordinate_y'], 
              color='green', s=120, edgecolors='white', linewidths=2, 
              zorder=3, label=f'Made ({len(makes)})', alpha=0.8)
    
    # Plot missed shots as red X's
    ax.scatter(misses['coordinate_x'], misses['coordinate_y'], 
              color='red', marker='x', s=120, linewidths=2.5, 
              zorder=3, label=f'Miss ({len(misses)})', alpha=0.8)
    
    # Create title with game context
    # Format date as MM/DD/YYYY
    date = pd.to_datetime(game_info['game_date']).strftime('%m/%d/%Y')
    
    # Determine if team was home or away, and extract relevant scores
    if game_info['home_team_id'] == team_id:
        opponent = game_info['away_team_name']
        location = "vs"  # Home game
        team_score = game_info['home_score']
        opp_score = game_info['away_score']
    else:
        opponent = game_info['home_team_name']
        location = "@"  # Away game
        team_score = game_info['away_score']
        opp_score = game_info['home_score']
    
    # Create score string with W/L indicator
    if pd.notna(team_score):
        result = "W" if team_score > opp_score else "L"
        score_str = f" ({result} {int(team_score)}-{int(opp_score)})"
    else:
        score_str = ""  # Game hasn't been played yet
    
    # Construct full title
    title = f"{player_name} Shot Chart - {location} {opponent} ({date}){score_str}"
    
    # Add legend and title to the plot
    plt.legend(loc='upper left', fontsize=10)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()  # Adjust spacing to prevent label cutoff
    
    # Convert the matplotlib figure to a PNG image in memory
    buf = io.BytesIO()  # Create a bytes buffer in memory
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')  # Save figure to buffer
    buf.seek(0)  # Reset buffer position to beginning
    
    # Encode the image as base64 so it can be embedded in HTML
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Close the matplotlib figure to free memory
    plt.close()
    
    return img_base64

# ============================================================================
# FLASK API ROUTES
# ============================================================================

@app.route('/')
def index():
    """
    Main page route - serves the HTML interface.
    
    Returns the HTML template directly (embedded in this file as HTML_TEMPLATE).
    When user visits http://localhost:5000, they see this page.
    """
    return HTML_TEMPLATE

@app.route('/api/teams')
def get_teams():
    """
    API endpoint to get all teams with conference information.
    
    Returns:
        JSON array of team objects: [{"team_id": 8, "team_name": "Arkansas", "conference": "SEC"}, ...]
        
    Called by JavaScript when the page loads to populate the team selection grid.
    """
    # Convert DataFrame to list of dictionaries for JSON serialization
    teams_list = all_teams.to_dict('records')
    return jsonify(teams_list)

@app.route('/api/conferences')
def get_conferences():
    """
    API endpoint to get all unique conferences.
    
    Returns:
        JSON array of conference names, sorted alphabetically
        
    Called by JavaScript to populate the conference filter dropdown.
    """
    # Get unique conferences, sort them, and convert to list
    conferences = sorted(all_teams['conference'].dropna().unique().tolist())
    return jsonify(conferences)

@app.route('/api/games/<int:team_id>')
def get_games(team_id):
    """
    API endpoint to get all games for a specific team.
    
    Args:
        team_id (int): Team ID from the URL path (e.g., /api/games/8 for Arkansas)
        
    Returns:
        JSON array of game objects with formatted data:
        [{
            "game_id": 401820294,
            "date": "11/27/2025",
            "opponent": "Houston",
            "location": "@",
            "score": "W 80-71"
        }, ...]
        
    Called by JavaScript when user selects a team.
    """
    # Get raw game data from database
    game_info = get_team_games(team_id)
    
    games_list = []
    
    # Process each game to create user-friendly display data
    for idx, row in game_info.iterrows():
        # Determine if this team was home or away
        if row['home_team_id'] == team_id:
            opponent = row['away_team_name']
            location = "vs"  # Home game
            team_score = row['home_score']
            opp_score = row['away_score']
        else:
            opponent = row['home_team_name']
            location = "@"  # Away game
            team_score = row['away_score']
            opp_score = row['home_score']
        
        # Format score string with W/L indicator
        if pd.notna(team_score) and pd.notna(opp_score):
            result = "W" if team_score > opp_score else "L"
            score_str = f"{result} {int(team_score)}-{int(opp_score)}"
        else:
            score_str = "TBD"  # Game not played yet
        
        # Add formatted game data to list
        games_list.append({
            'game_id': int(row['game_id']),
            'date': pd.to_datetime(row['game_date']).strftime('%m/%d/%Y'),
            'opponent': opponent,
            'location': location,
            'score': score_str
        })
    
    return jsonify(games_list)

@app.route('/api/players/<int:game_id>/<int:team_id>')
def get_players(game_id, team_id):
    """
    API endpoint to get all players who played in a specific game for a team.
    
    Args:
        game_id (int): Game ID from URL
        team_id (int): Team ID from URL
        
    Returns:
        JSON array of player objects:
        [{
            "player_id": 0,
            "name": "Team View",
            "shots": 57,
            "makes": 24,
            "percentage": 42.1
        }, ...]
        
    Note: player_id 0 is reserved for "Team View" option (shows all players' shots).
    Called by JavaScript when user selects a game.
    """
    # Get shot data and player roster
    team_shots, team_players = get_player_shots(game_id, team_id)
    
    players_list = []
    
    # First, add "Team View" option (shows all players' shots together)
    total_shots = len(team_shots)
    total_makes = len(team_shots[team_shots['scoring_play'] == True])
    players_list.append({
        'player_id': 0,  # Reserved ID for team view
        'name': 'Team View',
        'shots': total_shots,
        'makes': total_makes,
        'percentage': (total_makes / total_shots * 100) if total_shots > 0 else 0
    })
    
    # Add each individual player with their stats
    for idx, player in enumerate(sorted(team_players['athlete_display_name'].dropna().unique()), 1):
        # Filter to only this player's shots
        player_shot_data = team_shots[team_shots['athlete_display_name'] == player]
        shots_count = len(player_shot_data)
        makes_count = len(player_shot_data[player_shot_data['scoring_play'] == True])
        
        players_list.append({
            'player_id': idx,
            'name': player,
            'shots': shots_count,
            'makes': makes_count,
            'percentage': (makes_count / shots_count * 100) if shots_count > 0 else 0
        })
    
    return jsonify(players_list)

@app.route('/api/search-player/<player_name>')
def search_player(player_name):
    """
    API endpoint to search for a player across all teams and games.
    
    Args:
        player_name (str): Player name to search for (partial match)
        
    Returns:
        JSON array of players matching the search:
        [{
            "player_name": "Cameron Boozer",
            "team_name": "Duke",
            "team_id": 150,
            "total_games": 8,
            "total_shots": 96,
            "total_makes": 24
        }, ...]
        
    Called by JavaScript when user searches for a player globally.
    """
    # Get all unique players with their team info
    players = box_df[['athlete_display_name', 'team_display_name', 'team_id']].drop_duplicates()
    
    # Fuzzy search - split search term into parts
    search_parts = player_name.lower().split()
    
    # Filter players where all search parts appear in the name
    matching_players = []
    for _, player in players.iterrows():
        player_name_lower = str(player['athlete_display_name']).lower()
        if all(part in player_name_lower for part in search_parts):
            matching_players.append(player)
    
    if len(matching_players) == 0:
        return jsonify([])
    
    results = []
    
    for player in matching_players:
        player_name = player['athlete_display_name']
        team_id = player['team_id']
        
        # Get all games for this player
        player_games = box_df[
            (box_df['athlete_display_name'] == player_name) &
            (box_df['team_id'] == team_id)
        ]
        
        # Count games with shots
        games_with_shots = 0
        total_shots = 0
        total_makes = 0
        
        for game_id in player_games['game_id'].unique():
            team_shots, _ = get_player_shots(game_id, team_id)
            player_shots = team_shots[team_shots['athlete_display_name'] == player_name]
            
            if len(player_shots) > 0:
                games_with_shots += 1
                total_shots += len(player_shots)
                total_makes += len(player_shots[player_shots['scoring_play'] == True])
        
        if games_with_shots > 0:
            results.append({
                'player_name': player_name,
                'team_name': player['team_display_name'] if pd.notna(player['team_display_name']) else 'Unknown',
                'team_id': int(team_id),
                'total_games': games_with_shots,
                'total_shots': total_shots,
                'total_makes': total_makes
            })
    
    return jsonify(results)

@app.route('/api/player-games/<int:team_id>/<player_name>')
def get_player_games(team_id, player_name):
    """
    API endpoint to get all games for a specific player.
    
    Args:
        team_id (int): Team ID
        player_name (str): Player name
        
    Returns:
        JSON array of games where this player played with shot data
    """
    # Get all games for this player
    player_games = box_df[
        (box_df['athlete_display_name'] == player_name) &
        (box_df['team_id'] == team_id)
    ]
    
    results = []
    
    for game_id in player_games['game_id'].unique():
        # Get shot data
        team_shots, _ = get_player_shots(game_id, team_id)
        player_shots = team_shots[team_shots['athlete_display_name'] == player_name]
        
        if len(player_shots) == 0:
            continue
        
        # Get game info
        game_data = pbp_df[pbp_df['game_id'] == game_id]
        if len(game_data) == 0:
            continue
            
        game_row = game_data.iloc[-1]
        
        # Determine opponent and scores
        if game_row['home_team_id'] == team_id:
            opponent = game_row['away_team_name']
            location = "vs"
            team_score = game_row['home_score']
            opp_score = game_row['away_score']
        else:
            opponent = game_row['home_team_name']
            location = "@"
            team_score = game_row['away_score']
            opp_score = game_row['home_score']
        
        if pd.notna(team_score):
            result = "W" if team_score > opp_score else "L"
            score_str = f"{result} {int(team_score)}-{int(opp_score)}"
        else:
            score_str = "TBD"
        
        results.append({
            'game_id': int(game_id),
            'date': pd.to_datetime(game_row['game_date']).strftime('%m/%d/%Y'),
            'opponent': opponent,
            'location': location,
            'score': score_str,
            'shots': len(player_shots),
            'makes': len(player_shots[player_shots['scoring_play'] == True])
        })
    
    # Sort by date
    results = sorted(results, key=lambda x: x['date'])
    
    return jsonify(results)

@app.route('/api/player-season-chart/<int:team_id>/<player_name>')
def get_player_season_chart(team_id, player_name):
    """
    API endpoint to generate a shot chart for a player's entire season.
    
    Args:
        team_id (int): Team ID
        player_name (str): Player name
        
    Returns:
        JSON with image and stats for all games combined
    """
    # Get all games for this player
    player_games = box_df[
        (box_df['athlete_display_name'] == player_name) &
        (box_df['team_id'] == team_id)
    ]
    
    # Collect all shots from all games
    all_shots = []
    
    for game_id in player_games['game_id'].unique():
        team_shots, _ = get_player_shots(game_id, team_id)
        player_shots = team_shots[team_shots['athlete_display_name'] == player_name]
        
        if len(player_shots) > 0:
            all_shots.append(player_shots)
    
    if len(all_shots) == 0:
        return jsonify({'error': 'No shots found'}), 404
    
    # Combine all shots
    combined_shots = pd.concat(all_shots, ignore_index=True)
    
    # Create simplified shot chart for season view
    combined_shots['short_name'] = combined_shots['athlete_display_name'].apply(
        lambda x: f"{x.split(' ')[0][0]}. {x.split(' ')[-1]}" if pd.notnull(x) else ""
    )
    
    # Draw Court
    court = Court(court_type="ncaa", origin="center", units="ft")
    fig, ax = court.draw(orientation="h")
    fig.set_size_inches(14, 10)
    
    # Plot the shots
    makes = combined_shots[combined_shots['scoring_play'] == True]
    misses = combined_shots[combined_shots['scoring_play'] == False]
    
    ax.scatter(makes['coordinate_x'], makes['coordinate_y'], 
              color='green', s=120, edgecolors='white', linewidths=2, 
              zorder=3, label=f'Made ({len(makes)})', alpha=0.8)
    ax.scatter(misses['coordinate_x'], misses['coordinate_y'], 
              color='red', marker='x', s=120, linewidths=2.5, 
              zorder=3, label=f'Miss ({len(misses)})', alpha=0.8)
    
    # Create title
    title = f"{player_name} - 2025-26 Season Shot Chart ({len(player_games['game_id'].unique())} games)"
    
    plt.legend(loc='upper left', fontsize=10)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    makes_count = len(makes)
    total = len(combined_shots)
    
    return jsonify({
        'image': img_base64,
        'stats': {
            'makes': makes_count,
            'misses': total - makes_count,
            'total': total,
            'percentage': (makes_count / total * 100) if total > 0 else 0,
            'games': len(player_games['game_id'].unique())
        }
    })
@app.route('/api/shot-chart/<int:game_id>/<int:team_id>/<player_name>')
def get_shot_chart(game_id, team_id, player_name):
    """
    API endpoint to generate a shot chart image.
    
    Args:
        game_id (int): Game ID from URL
        team_id (int): Team ID from URL
        player_name (str): Player name from URL (URL-encoded, e.g., "Cameron%20Boozer")
        
    Returns:
        JSON object containing:
        {
            "image": "base64-encoded PNG image data",
            "stats": {
                "makes": 3,
                "misses": 9,
                "total": 12,
                "percentage": 25.0
            }
        }
        
    The image is base64-encoded so it can be embedded directly in HTML:
    <img src="data:image/png;base64,{image_data}">
    
    Called by JavaScript when user selects a player.
    """
    # Get all shot data for this game and team
    team_shots, team_players = get_player_shots(game_id, team_id)
    
    # Get game information for title and context
    game_info = get_team_games(team_id)
    game_info = game_info[game_info['game_id'] == game_id].iloc[0]
    
    # Filter shots based on whether user selected team view or individual player
    if player_name == 'Team View':
        player_shots = team_shots  # All shots from the team
    else:
        # Only shots from this specific player
        player_shots = team_shots[team_shots['athlete_display_name'] == player_name].copy()
    
    # Return error if no shots found (shouldn't happen if UI prevents it)
    if len(player_shots) == 0:
        return jsonify({'error': 'No shots found'}), 404
    
    # Generate the shot chart as a base64-encoded image
    img_base64 = create_shot_chart(player_shots, player_name, game_info, team_id)
    
    # Calculate shooting statistics
    makes = len(player_shots[player_shots['scoring_play'] == True])
    total = len(player_shots)
    
    # Return image and stats as JSON
    return jsonify({
        'image': img_base64,
        'stats': {
            'makes': makes,
            'misses': total - makes,
            'total': total,
            'percentage': (makes / total * 100) if total > 0 else 0
        }
    })

# ============================================================================
# HTML TEMPLATE - Frontend Interface
# ============================================================================

# This is the entire frontend in one HTML string.
# It's embedded here for simplicity, but could be a separate file.

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>College Basketball Shot Charts</title>
    <style>
        /* ================================================================
           GLOBAL STYLES - Applied to all elements
           ================================================================ */
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; /* Include padding/border in element width */
        }
        
        /* ================================================================
           BODY - Main page container with gradient background
           ================================================================ */
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh; /* Full viewport height */
            padding: 20px;
        }
        
        /* ================================================================
           MAIN CONTAINER - White card that holds all content
           ================================================================ */
        .container {
            max-width: 1200px;
            margin: 0 auto; /* Center horizontally */
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2); /* Drop shadow */
        }
        
        /* ================================================================
           TYPOGRAPHY - Headers and text styles
           ================================================================ */
        h1 {
            color: #1e3c72;
            margin-bottom: 10px;
            font-size: 32px;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 16px;
        }
        
        /* ================================================================
           BACK BUTTON - Navigation button in top left
           ================================================================ */
        .back-btn {
            background: #6c757d;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .back-btn:hover { 
            background: #5a6268; /* Darker on hover */
        }
        
        /* ================================================================
           SEARCH BOX - Team search input field
           ================================================================ */
        .search-box {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .search-box:focus {
            outline: none; /* Remove default browser outline */
            border-color: #1e3c72; /* Blue border when focused */
        }
        
        /* ================================================================
           DROPDOWN - For player search suggestions
           ================================================================ */
        .dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 2px solid #1e3c72;
            border-top: none;
            border-radius: 0 0 6px 6px;
            max-height: 300px;
            overflow-y: auto;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .dropdown-item {
            padding: 12px 15px;
            cursor: pointer;
            border-bottom: 1px solid #f0f0f0;
            transition: background 0.2s;
        }
        
        .dropdown-item:hover {
            background: #f8f9fa;
        }
        
        .dropdown-item:last-child {
            border-bottom: none;
        }
        
        .dropdown-item-name {
            font-weight: 600;
            color: #333;
            font-size: 16px;
        }
        
        .dropdown-item-details {
            font-size: 13px;
            color: #666;
            margin-top: 3px;
        }
        
        /* ================================================================
           GRID LAYOUT - For displaying teams/games/players as cards
           ================================================================ */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); /* Responsive columns */
            gap: 15px; /* Space between cards */
            margin-bottom: 20px;
        }
        
        /* ================================================================
           CARD - Individual clickable items (teams, games, players)
           ================================================================ */
        .card {
            padding: 20px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s; /* Smooth hover animation */
            background: white;
        }
        .card:hover {
            border-color: #1e3c72; /* Blue border on hover */
            background: #f8f9fa; /* Light gray background */
            transform: translateY(-2px); /* Lift up slightly */
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); /* Add shadow */
        }
        
        .card-title {
            font-size: 18px;
            font-weight: 600;
            color: #333;
        }
        
        .card-subtitle {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        
        /* ================================================================
           BADGE - Win/Loss indicators on game cards
           ================================================================ */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 5px;
        }
        .badge-win {
            background: #d4edda; /* Light green */
            color: #155724; /* Dark green */
        }
        .badge-loss {
            background: #f8d7da; /* Light red */
            color: #721c24; /* Dark red */
        }
        
        /* ================================================================
           LOADING SPINNER - Shown while data is being fetched
           ================================================================ */
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .spinner {
            border: 4px solid #f3f3f3; /* Light gray */
            border-top: 4px solid #1e3c72; /* Blue top */
            border-radius: 50%; /* Make it circular */
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite; /* Rotate continuously */
            margin: 0 auto;
        }
        
        /* Keyframe animation for spinner rotation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* ================================================================
           SHOT CHART - Displays the matplotlib-generated image
           ================================================================ */
        .shot-chart-container {
            text-align: center;
            margin-top: 20px;
        }
        
        .shot-chart-img {
            max-width: 100%; /* Responsive - scales down on small screens */
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* ================================================================
           STATS DISPLAY - Shows makes, misses, percentage below shot chart
           ================================================================ */
        .stats {
            display: flex;
            justify-content: center;
            gap: 40px; /* Space between stat items */
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #1e3c72;
        }
        
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        
        /* ================================================================
           BUTTON GROUP - Action buttons below shot chart
           ================================================================ */
        .button-group {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #1e3c72;
            color: white;
        }
        .btn-primary:hover {
            background: #16325c; /* Darker on hover */
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background: #5a6268; /* Darker on hover */
        }
    </style>
</head>
<body>
    <!-- Main content container - JavaScript will populate this -->
    <div class="container">
        <div id="app">
            <h1>College Basketball Shot Charts</h1>
            <p class="subtitle">2025-26 Season</p>
            <div class="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px;">Loading data...</p>
            </div>
        </div>
    </div>

    <script>
        /* ================================================================
           GLOBAL STATE - Stores current selections
           ================================================================ */
        
        // Currently selected team (object with id and name)
        let currentTeam = null;
        
        // Currently selected game (object with game info)
        let currentGame = null;
        
        // All teams loaded from API (array of team objects)
        let allTeams = [];
        
        // All conferences loaded from API (array of conference names)
        let allConferences = [];
        
        // Current search and filter state
        let currentSearch = '';
        let currentConference = 'all';
        
        // Player selection state
        let currentPlayers = [];
        let currentPlayerSearch = '';

        /* ================================================================
           INITIALIZATION - Load teams when page loads
           ================================================================ */
        
        /**
         * Load all teams and conferences from the API and display team selection.
         * This is called once when the page loads.
         */
        async function loadTeams() {
            // Fetch teams from Flask API endpoint
            const teamsResponse = await fetch('/api/teams');
            allTeams = await teamsResponse.json(); // Parse JSON response
            
            // Fetch conferences from Flask API endpoint
            const conferencesResponse = await fetch('/api/conferences');
            allConferences = await conferencesResponse.json();
            
            showTeamSelect(); // Display team selection UI
        }

        /* ================================================================
           TEAM SELECTION
           ================================================================ */
        
        /**
         * Display the team selection interface with search and conference filter functionality.
         * 
         * @param {string} searchTerm - Current search text (empty string for all teams)
         * @param {string} conferenceFilter - Current conference filter ('all' or specific conference)
         */
        function showTeamSelect(searchTerm = '', conferenceFilter = 'all') {
            // Save current state
            currentSearch = searchTerm;
            currentConference = conferenceFilter;
            
            // Filter teams based on search term (case-insensitive)
            let filtered = allTeams;
            
            // Apply conference filter first
            if (conferenceFilter !== 'all') {
                filtered = filtered.filter(t => t.conference === conferenceFilter);
            }
            
            // Then apply search filter
            if (searchTerm) {
                filtered = filtered.filter(t => 
                    t.team_name.toLowerCase().includes(searchTerm.toLowerCase())
                );
            }

            // Build HTML string for the team selection UI
            const html = `
                <h2>Select a Team</h2>
                
                <!-- Filter Controls -->
                <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                    <select class="search-box" id="conference-filter" style="flex: 0 0 250px;">
                        <option value="all">All Conferences</option>
                        ${allConferences.map(conf => `
                            <option value="${conf}" ${conferenceFilter === conf ? 'selected' : ''}>
                                ${conf}
                            </option>
                        `).join('')}
                    </select>
                    
                    <input type="text" class="search-box" id="team-search" 
                           placeholder="Search teams..." 
                           value="${searchTerm}"
                           style="flex: 1;">
                </div>
                
                <!-- Results count -->
                <p style="color: #666; margin-bottom: 15px; font-size: 14px;">
                    Showing ${filtered.length} team${filtered.length !== 1 ? 's' : ''}
                    ${conferenceFilter !== 'all' ? ` in ${conferenceFilter}` : ''}
                </p>
                
                <div class="grid">
                    ${filtered.length > 0 
                        ? filtered.map(team => `
                            <div class="card" onclick="selectTeam(${team.team_id}, '${team.team_name.replace(/'/g, "\\'")}')">
                                <div class="card-title">${team.team_name}</div>
                                <div class="card-subtitle">${team.conference || 'Independent'}</div>
                            </div>
                        `).join('')
                        : '<p style="grid-column: 1/-1; text-align: center; padding: 40px; color: #666;">No teams found. Try adjusting your filters.</p>'
                    }
                </div>
            `;
            
            // Update the page with new HTML
            document.getElementById('app').innerHTML = html;
            
            // Attach event listener to conference dropdown
            document.getElementById('conference-filter').addEventListener('change', (e) => {
                showTeamSelect(currentSearch, e.target.value);
            });
            
            // Attach event listener to search box for real-time filtering
            const searchBox = document.getElementById('team-search');
            searchBox.addEventListener('input', (e) => {
                showTeamSelect(e.target.value, currentConference); // Re-render with search term
            });
            
            // Set focus back to search box and move cursor to end
            searchBox.focus();
            searchBox.setSelectionRange(searchTerm.length, searchTerm.length);
        }
        
        /**
         * Call the player search API
         */
        async function searchPlayerAPI(playerName) {
            try {
                const response = await fetch(`/api/search-player/${encodeURIComponent(playerName)}`);
                return await response.json();
            } catch (error) {
                console.error('Search error:', error);
                return [];
            }
        }
        
        /**
         * Show dropdown with player suggestions
         */
        function showPlayerDropdown(results) {
            const dropdown = document.getElementById('player-dropdown');
            
            if (results.length === 0) {
                dropdown.style.display = 'none';
                return;
            }
            
            dropdown.className = 'dropdown';
            dropdown.style.display = 'block';
            dropdown.innerHTML = results.map(player => `
                <div class="dropdown-item" onclick='selectPlayerFromDropdown(${JSON.stringify(player).replace(/'/g, "\\'")})'>
                    <div class="dropdown-item-name">${player.player_name}</div>
                    <div class="dropdown-item-details">
                        ${player.team_name} • ${player.total_games} games • ${player.total_makes}/${player.total_shots} FG (${(player.total_makes/player.total_shots*100).toFixed(1)}%)
                    </div>
                </div>
            `).join('');
        }
        
        /**
         * Select a player from the dropdown
         */
        function selectPlayerFromDropdown(player) {
            document.getElementById('player-dropdown').style.display = 'none';
            showPlayerList([player]);
        }
        
        /**
         * Show list of matching players to choose from
         */
        function showPlayerList(players) {
            const html = `
                <button class="back-btn" onclick="showTeamSelect('', 'all')">← Back to Team Selection</button>
                <h2>Select Player</h2>
                <p class="subtitle">Found ${players.length} player${players.length !== 1 ? 's' : ''} with shot data</p>
                
                <div class="grid">
                    ${players.map(player => `
                        <div class="card" onclick='loadPlayerGames(${player.team_id}, "${player.player_name.replace(/"/g, '&quot;')}", "${player.team_name.replace(/"/g, '&quot;')}")'>
                            <div class="card-title">${player.player_name}</div>
                            <div class="card-subtitle">
                                ${player.team_name}<br>
                                ${player.total_games} games • ${player.total_makes}/${player.total_shots} FG (${(player.total_makes/player.total_shots*100).toFixed(1)}%)
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            document.getElementById('app').innerHTML = html;
        }
        
        /**
         * Load all games for a specific player
         */
        async function loadPlayerGames(teamId, playerName, teamName) {
            document.getElementById('app').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            
            currentTeam = { id: teamId, name: teamName };
            
            const response = await fetch(`/api/player-games/${teamId}/${encodeURIComponent(playerName)}`);
            const games = await response.json();
            
            showPlayerGames(playerName, games);
        }
        
        /**
         * Show all games for a player with option to view season chart
         */
        function showPlayerGames(playerName, games) {
            const totalShots = games.reduce((sum, g) => sum + g.shots, 0);
            const totalMakes = games.reduce((sum, g) => sum + g.makes, 0);
            
            const html = `
                <button class="back-btn" onclick="showTeamSelect('', 'all')">← Back to Search</button>
                <h2>${playerName}</h2>
                <p class="subtitle">${currentTeam.name} • ${games.length} games</p>
                
                <!-- Season-wide option -->
                <div style="margin-bottom: 30px;">
                    <div class="card" onclick='viewSeasonChart("${playerName.replace(/"/g, '&quot;')}")' 
                         style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; border: none;">
                        <div class="card-title" style="color: white; font-size: 20px;">📊 View Entire Season Shot Chart</div>
                        <div class="card-subtitle" style="color: rgba(255,255,255,0.9);">
                            All ${games.length} games combined • ${totalMakes}/${totalShots} FG (${(totalMakes/totalShots*100).toFixed(1)}%)
                        </div>
                    </div>
                </div>
                
                <h3 style="margin-bottom: 15px; color: #333;">Individual Games</h3>
                <div class="grid">
                    ${games.map(game => `
                        <div class="card" onclick='selectPlayerGame(${JSON.stringify(game).replace(/'/g, "\\'")},"${playerName.replace(/"/g, '&quot;')}")'>
                            <div class="card-title">${game.date}</div>
                            <div class="card-subtitle">
                                ${game.location} ${game.opponent}<br>
                                ${game.score}
                            </div>
                            <div style="margin-top: 8px; font-size: 14px; color: #1e3c72; font-weight: 600;">
                                ${game.makes}/${game.shots} FG (${(game.makes/game.shots*100).toFixed(1)}%)
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            document.getElementById('app').innerHTML = html;
        }
        
        /**
         * View season-wide shot chart for a player
         */
        async function viewSeasonChart(playerName) {
            document.getElementById('app').innerHTML = '<div class="loading"><div class="spinner"></div><p style="margin-top: 15px;">Generating season shot chart...</p></div>';
            
            const response = await fetch(`/api/player-season-chart/${currentTeam.id}/${encodeURIComponent(playerName)}`);
            const data = await response.json();
            
            showSeasonShotChart(data, playerName);
        }
        
        /**
         * Display season-wide shot chart
         */
        function showSeasonShotChart(data, playerName) {
            const html = `
                <button class="back-btn" onclick="loadPlayerGames(${currentTeam.id}, '${playerName.replace(/'/g, "\\'")}', '${currentTeam.name.replace(/'/g, "\\'")}')">
                    ← Back to Game List
                </button>
                <h2>${playerName} - 2025-26 Season</h2>
                <p class="subtitle">${currentTeam.name} • All ${data.stats.games} games combined</p>
                
                <div class="shot-chart-container">
                    <img src="data:image/png;base64,${data.image}" class="shot-chart-img">
                </div>
                
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value">${data.stats.makes}</div>
                        <div class="stat-label">Makes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.stats.misses}</div>
                        <div class="stat-label">Misses</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.stats.percentage.toFixed(1)}%</div>
                        <div class="stat-label">FG%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.stats.games}</div>
                        <div class="stat-label">Games</div>
                    </div>
                </div>
                
                <div class="button-group">
                    <button class="btn btn-primary" onclick="loadPlayerGames(${currentTeam.id}, '${playerName.replace(/'/g, "\\'")}', '${currentTeam.name.replace(/'/g, "\\'")}')">
                        View Individual Games
                    </button>
                    <button class="btn btn-secondary" onclick="showTeamSelect('', 'all')">
                        Search Another Player
                    </button>
                </div>
            `;
            document.getElementById('app').innerHTML = html;
        }
        
        /**
         * Jump directly to a player's shot chart from search results
         */
        function selectPlayerGame(game, playerName) {
            // Set current game
            currentGame = {
                game_id: game.game_id,
                date: game.date,
                opponent: game.opponent,
                location: game.location,
                score: game.score
            };
            
            // Go directly to shot chart
            selectPlayer(playerName);
        }

        /* ================================================================
           GAME SELECTION
           ================================================================ */
        
        /**
         * Handle team selection - fetch games and display game selection UI.
         * 
         * @param {number} teamId - The ID of the selected team
         * @param {string} teamName - The name of the selected team
         */
        async function selectTeam(teamId, teamName) {
            // Store selected team in global state
            currentTeam = { id: teamId, name: teamName };
            
            // Show loading spinner while fetching games
            document.getElementById('app').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            
            // Fetch games for this team from Flask API
            const response = await fetch(`/api/games/${teamId}`);
            const games = await response.json();
            
            // Display game selection UI
            showGameSelect(games);
        }

        /**
         * Display the game selection interface.
         * 
         * @param {Array} games - Array of game objects from the API
         */
        function showGameSelect(games) {
            const html = `
                <button class="back-btn" onclick="showTeamSelect('')">← Change Team</button>
                <h2>${currentTeam.name} - 2025-26 Games</h2>
                <div class="grid">
                    ${games.map((game, idx) => `
                        <div class="card" onclick='selectGame(${JSON.stringify(game)})'>
                            <div class="card-title">${idx + 1}. ${game.date}</div>
                            <div class="card-subtitle">${game.location} ${game.opponent}</div>
                            <span class="badge ${game.score.startsWith('W') ? 'badge-win' : 'badge-loss'}">
                                ${game.score}
                            </span>
                        </div>
                    `).join('')}
                </div>
            `;
            document.getElementById('app').innerHTML = html;
        }

        /* ================================================================
           PLAYER SELECTION
           ================================================================ */
        
        /**
         * Handle game selection - fetch players and display player selection UI.
         * 
         * @param {Object} game - The selected game object
         */
        async function selectGame(game) {
            // Store selected game in global state
            currentGame = game;
            
            // Show loading spinner
            document.getElementById('app').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            
            // Fetch players for this game from Flask API
            const response = await fetch(`/api/players/${game.game_id}/${currentTeam.id}`);
            const players = await response.json();
            
            // Display player selection UI
            showPlayerSelect(players);
        }

        /**
         * Display the player selection interface.
         * 
         * @param {Array} players - Array of player objects from the API
         * @param {string} searchTerm - Current search text for filtering players
         */
        function showPlayerSelect(players, searchTerm = '') {
            // Save players and search state
            currentPlayers = players;
            currentPlayerSearch = searchTerm;
            
            // Filter players based on search term
            const filteredPlayers = searchTerm
                ? players.filter(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()))
                : players;
            
            // Escape apostrophes in team name for onclick handler (e.g., "St. John's")
            const html = `
                <button class="back-btn" onclick="selectTeam(${currentTeam.id}, '${currentTeam.name.replace(/'/g, "\\'")}')">
                    ← Back to Games
                </button>
                <h2>${currentGame.location} ${currentGame.opponent}</h2>
                <p class="subtitle">${currentGame.date} • ${currentGame.score}</p>
                
                <!-- Player search box -->
                <input type="text" class="search-box" id="player-search" 
                       placeholder="Search players..." 
                       value="${searchTerm}"
                       style="margin-bottom: 20px;">
                
                <!-- Results count -->
                <p style="color: #666; margin-bottom: 15px; font-size: 14px;">
                    Showing ${filteredPlayers.length} player${filteredPlayers.length !== 1 ? 's' : ''}
                </p>
                
                <div class="grid">
                    ${filteredPlayers.length > 0 
                        ? filteredPlayers.map(player => {
                            const hasShots = player.shots > 0;
                            const clickHandler = hasShots 
                                ? `onclick='selectPlayer("${player.name.replace(/"/g, '&quot;')}")'`
                                : `style="opacity: 0.5; cursor: not-allowed;"`;
                            
                            return `
                                <div class="card" ${clickHandler}>
                                    <div class="card-title">${player.name}</div>
                                    <div class="card-subtitle">
                                        ${hasShots
                                            ? `${player.makes}/${player.shots} FG (${player.percentage.toFixed(1)}%)`
                                            : 'No shots - cannot view chart'}
                                    </div>
                                </div>
                            `;
                        }).join('')
                        : '<p style="grid-column: 1/-1; text-align: center; padding: 40px; color: #666;">No players found matching your search.</p>'
                    }
                </div>
            `;
            document.getElementById('app').innerHTML = html;
            
            // Attach event listener to search box for real-time filtering
            const searchBox = document.getElementById('player-search');
            searchBox.addEventListener('input', (e) => {
                showPlayerSelect(currentPlayers, e.target.value);
            });
            
            // Set focus back to search box and move cursor to end
            searchBox.focus();
            searchBox.setSelectionRange(searchTerm.length, searchTerm.length);
        }

        /* ================================================================
           SHOT CHART DISPLAY
           ================================================================ */
        
        /**
         * Handle player selection - fetch shot chart and display it.
         * 
         * @param {string} playerName - The name of the selected player
         */
        async function selectPlayer(playerName) {
            // Show loading spinner
            document.getElementById('app').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            
            // Fetch shot chart image and stats from Flask API
            // encodeURIComponent handles spaces and special characters in names
            const response = await fetch(`/api/shot-chart/${currentGame.game_id}/${currentTeam.id}/${encodeURIComponent(playerName)}`);
            const data = await response.json();
            
            // Display shot chart UI
            showShotChart(data, playerName);
        }

        /**
         * Display the shot chart with statistics and action buttons.
         * 
         * @param {Object} data - Shot chart data from API (image and stats)
         * @param {string} playerName - Name of the player
         */
        function showShotChart(data, playerName) {
            // Properly escape JSON and strings for onclick handlers
            const gameJson = JSON.stringify(currentGame).replace(/'/g, "\\'");
            const teamNameEscaped = currentTeam.name.replace(/'/g, "\\'");
            
            const html = `
                <button class="back-btn" onclick='selectGame(${gameJson})'>
                    ← Back to Players
                </button>
                <h2>${playerName}</h2>
                <p class="subtitle">${currentGame.location} ${currentGame.opponent} • ${currentGame.date}</p>
                
                <!-- Shot chart image (base64-encoded PNG from matplotlib) -->
                <div class="shot-chart-container">
                    <img src="data:image/png;base64,${data.image}" class="shot-chart-img">
                </div>
                
                <!-- Shooting statistics -->
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value">${data.stats.makes}</div>
                        <div class="stat-label">Makes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.stats.misses}</div>
                        <div class="stat-label">Misses</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.stats.percentage.toFixed(1)}%</div>
                        <div class="stat-label">FG%</div>
                    </div>
                </div>
                
                <!-- Action buttons for navigation -->
                <div class="button-group">
                    <button class="btn btn-primary" onclick='selectGame(${gameJson})'>
                        View Another Player
                    </button>
                    <button class="btn btn-secondary" onclick="selectTeam(${currentTeam.id}, '${teamNameEscaped}')">
                        View Another Game
                    </button>
                    <button class="btn btn-secondary" onclick="showTeamSelect('')">
                        Change Team
                    </button>
                </div>
            `;
            document.getElementById('app').innerHTML = html;
        }

        /* ================================================================
           START APPLICATION
           ================================================================ */
        
        // Initialize the app by loading teams when page loads
        loadTeams();
    </script>
</body>
</html>
'''

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

@app.route('/templates/index.html')
def template():
    """Fallback route for template (not actually used, kept for compatibility)"""
    return HTML_TEMPLATE

# Replace the bottom section of your app.py (after if __name__ == '__main__':) with this:

if __name__ == '__main__':
    """
    Main entry point - run this script to start the Flask web server.
    """
    import os
    
    # Load data once at startup
    load_data()
    
    # Print startup message
    print("\n" + "="*80)
    print("College Basketball Shot Chart Viewer is ready!")
    print("="*80 + "\n")
    
    # Get port from environment variable (for cloud deployment) or use 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Start Flask server
    # Use 0.0.0.0 to accept external connections
    app.run(debug=False, port=port, host='0.0.0.0')