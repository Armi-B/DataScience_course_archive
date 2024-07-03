import random

def roll_dice():
    return random.randint(1, 6)

def move_piece(player, steps):
    # Move the player's piece
    # This can be customized based on the game rules
    player['position'] += steps

def is_winner(player):
    # Check if the player has reached the winning position
    # This can be customized based on the game rules
    return player['position'] >= 100

def simulate_game():
    players = [{'position': 0} for _ in range(4)]
    while True:
        for player in players:
            steps = roll_dice()
            move_piece(player, steps)
            if is_winner(player):
                return players.index(player)

def monte_carlo_simulation(iterations):
    results = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(iterations):
        winner = simulate_game()
        results[winner] += 1
    probabilities = {player: (results[player] / iterations) * 100 for player in results}
    return probabilities

if __name__ == "__main__":
    iterations = 100000
    probabilities = monte_carlo_simulation(iterations)
    for player, probability in probabilities.items():
        print(f"Player {player+1} probability of winning: {probability:.2f}%")
