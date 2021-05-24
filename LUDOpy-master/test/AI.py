import unittest
import sys
import numpy as np
import copy
sys.path.append("../")

num_wins = 0

#class QLearningPlayer():
    def __init__(self, filepath):
        self.Q = self.loadQTable(filepath)
        self.numOfPieces = 4
        self.numOfStates = 6 * self.numOfPieces
        self.numOfActions = 4

    def loadQTable(self,filepath):
        print("Loading Q Table.")
        return np.load(filepath)

    def getQValue(self, state, diceIdx, action):
        idx = state+[diceIdx]+[action]
        return self.Q[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5],idx[6],idx[7],idx[8],idx[9],idx[10],idx[11],idx[12],idx[13],idx[14],idx[15],idx[16],idx[17],idx[18],idx[19],idx[20],idx[21],idx[22],idx[23],idx[24],idx[25]]

    def getState(self, playerPieces, enemyPieces):
        def distanceBetweenTwoPieces(piece, enemy, i):
            if enemy == 0 or enemy >= 53 or piece == 0 or piece >= 53:
                return 1000
            enemy_relative_to_piece = (enemy + 13 * i) % 52
            if enemy_relative_to_piece == 0: enemy_relative_to_piece = 52
            distances = [enemy_relative_to_piece - piece, (enemy_relative_to_piece - 52) - piece]
            return distances[np.argmin(list(map(abs,distances)))]

        HOME = 0
        SAFE = 1
        VULNERABLE = 2
        ATTACKING = 3
        FINISHLINE = 4
        FINISHED = 5

        home = [0]
        globes = [1, 9, 14, 22, 27, 35, 40, 48]
        unsafe_globes = [14, 27, 40]

        state = []
        for playerPiece in playerPieces:
            pieceState = [0] * (int)(self.numOfStates / self.numOfPieces)

            #Calculating the relative distance of all the enemy pieces to the players piece
            distanceToEnemy = []
            for i, enemy in enumerate(enemyPieces):
                for enemyPiece in enemy:
                    distanceToEnemy.append(distanceBetweenTwoPieces(playerPiece, enemyPiece, i + 1))

            if playerPiece in home:
                pieceState[HOME] = 1

            if playerPiece in globes:
                pieceState[SAFE] = 1

            vulnerable = any([-6 <= relativePosition < 0 for relativePosition in distanceToEnemy])
            if (vulnerable and playerPiece not in globes) or playerPiece in unsafe_globes: pieceState[VULNERABLE] = 1

            attacking = any([0 < relativePosition <= 6 for relativePosition in distanceToEnemy])
            if attacking: pieceState[ATTACKING] = 1

            if playerPiece >= 53:
                pieceState[FINISHLINE] = 1

            if playerPiece == 59:
                pieceState[FINISHED] = 1

            state += pieceState
        return state

    def getNextAction(self, state, dice, movePieces):
        diceIdx = dice - 1
        bestAction = movePieces[0]
        bestQValue = self.getQValue(state, diceIdx, bestAction)
        for action in movePieces:
            if self.getQValue(state,diceIdx, action) > bestQValue:
                bestAction = action
                bestQValue = self.getQValue(state,diceIdx,action)

        return bestAction


class genetic:

    #Initialize the GA
    def __init__(self):
        self.f = open("average_winrate.txt", "a")
        self.f2 = open("highest_winrate.txt", "a")
        self.f3 = open("best_weight.txt", "a")
        self.f4 = open("Lowest_winrate.txt", "a")
        self.weights = []
        self.winrate = []
        self.temp_weights = [0,0,0,0,0,0,0,0,0,0]
        self.new_weights = []
        self.stars = [5, 12, 18, 25, 31, 38, 44, 51] 
        self.safe_field = [1, 9, 22, 35, 48, 53, 54, 55, 56, 57, 58, 59]
        self.globus = [1, 9, 22, 35, 48, 53]
        self.num_games = 100
        self.pop_size = 100
        self.survive_num = 10
        self.mutate_rate = 20
        self.generations = 30
        self.new_pop = 10
        self.crossover_num = 40
        self.mutate_num = self.pop_size-self.survive_num-self.new_pop- self.crossover_num
        self.pop = 0
        #1: Move out of starting area
        #2: Enter goal
        #3: Hit opponent home
        #4: hit a star
        #5: Hit a dangerous field
        #6: Enter safe field Tag højde for om der er andre brikker der i forvejen
        #7: Hit itself home


    #Shutdown all files where the winrate and rates are being saved
    def shutdown(self):
        print("Shutting down")
        self.f.close()
        self.f2.close()
        self.f3.close()
        self.f4.close()

    #Initialize random weights for the genes
    def init_weights(self):
        for p in range(self.pop_size):
            for i in range(len(self.temp_weights)):
                self.temp_weights[i] = np.random.randint(-100,100)
            self.weights.append(self.temp_weights[:])
        
    #Check if an enemy is safe or can be hit home
    def enemy_safe(self, piece, enemy_piece, index):
        count = 0
        enemy_safe = copy.deepcopy(self.safe_field)
        for i in range(len(enemy_safe)):
            if enemy_safe[i] < 54 and enemy_safe[i] != 0:
                enemy_safe[i] += (13*(index+1))
                enemy_safe[i] = enemy_safe[i] % 53 + 1
        
        for safe in enemy_safe:
            if piece == safe:
                return True
        for pieces in enemy_piece:
            if piece == pieces and piece != 0:
                count += 1
            if count == 2:
                return True
        return False
    

    #Evaluate a single set of weights over 10000 games
    def eval_single(self):
        #Baseline 25.1% winrate
        games = 10000
        #num_wins = 0
        num_wins1 = 0
        num_wins2 = 0
        num_wins3 = 0
        num_wins4 = 0
        #weight = [73, 69, 37, 86, 12, 17, 27, 5, 13, -82]
        weight = [73, 69, 37, 86, 12, 17, 27, 5, 13, -82]
        weight = [55, 30, 81, 79, 35, 17, -1, -39, 75, -69]
        #1 giver stort boost ved positiv, 2 ingenting, 3lille boost ved positiv 29.2%, 4 umiddelbart ikke det store, svinger en del 25%, 5 dårligt hvis positiv, 
        #6 ikke det store, 7 ikke det store 26% winrate ved negativ, 8 giver boost ved positiv, 9 lille boost ved positiv, 10 lille boost ved negativ
        for i in range(games):
            result = game(self, weight)
            if (result == 1):
                num_wins1 += 1
            elif (result == 2):
                num_wins2 += 1
            elif (result == 3):
                num_wins3 += 1
            elif (result == 4):
                num_wins4 += 1
            #num_wins += game(self, weight)
            if (i%100 == 0 and i != 0):
                print("Games %d" %i)
                print("winrate player 1: %g" % float(num_wins1/i))
                print("winrate player 2: %g" % float(num_wins2/i))
                print("winrate player 3: %g" % float(num_wins3/i))
                print("winrate player 4: %g" % float(num_wins4/i))
        print("winrate final 1: %g" % float(num_wins1/games))
        print("winrate final 2: %g" % float(num_wins2/games))
        print("winrate final 3: %g" % float(num_wins3/games))
        print("winrate final 4: %g" % float(num_wins4/games))

    #Finds the best move based on the utility function
    def eval(self, player_pieces, enemy_pieces, dice, move_pieces, weight):
        value = [0,0,0,0]
        for i in range(len(move_pieces)):
            value = self.best_move(player_pieces, enemy_pieces, dice, move_pieces, i, weight, value)
        index = move_pieces[0]
        max_val = 0
        
        for i in range(len(value)): 
            if value[i] > max_val:
                max_val = value[i]
                index = i
            if max_val == 0:
                for n in range(len(move_pieces)):
                    if (move_pieces[n] == 0):
                        index = move_pieces[n]
                        break
                    index = move_pieces[0]
        return index

    #Utility function to find the best move
    def best_move(self, player_pieces, enemy_pieces, dice, move_pieces, i, weight, value):
        
        #Makes the enemy pieces have same numbering as player pieces
        for m, enemy_piece in enumerate(enemy_pieces):
            for n in range(len(enemy_piece)):
                if enemy_piece[n] < 54 and enemy_piece[n] != 0:
                    enemy_piece[n] += (13*(m+1)) 
                    if (enemy_piece[n] >= 53):
                        enemy_piece[n] = enemy_piece[n] % 53 + 1
        player_after_move = player_pieces[move_pieces[i]] + dice
        
        #State 1: Move out of starting area
        if (player_pieces[move_pieces[i]] == 0 and dice == 6):
            value[move_pieces[i]] += weight[0]

        #State 2: Enter goal
        if (player_pieces[move_pieces[i]] + dice == 59):
            value[move_pieces[i]] += weight[1]

        #State 3: Hit opponent home
        for n,enemy_piece in enumerate(enemy_pieces):
            for piece in enemy_piece:
                if (player_after_move == piece):
                    if (piece < 54 and piece != 0):
                        if not(self.enemy_safe(piece, enemy_piece, n)):
                            value[move_pieces[i]] += weight[2]
                            #print("Can kill opponent") 
                            break

            

        #State 4: hit a star
        for star in self.stars:
            if (player_after_move == star):
                value[move_pieces[i]] += weight[3]
                break

        #State 5: Hit a dangerous field
        for enemy_piece in enemy_pieces:
            for piece in enemy_piece:
                if player_after_move > piece and player_after_move < piece + 7:
                    if (piece < 54 and piece != 0):
                        value[move_pieces[i]] += weight[4]
                        break

        #State 6: Safe
        flag = True
        for field in self.safe_field:
            if (player_after_move == field):
                if player_after_move > 53: #Check if player is in goal fields
                    value[move_pieces[i]] += weight[5]
                    break
                else:
                    for n,enemy_piece in enumerate(enemy_pieces): #Check if it will hit another player on safe field
                        for piece in enemy_piece:
                            if (player_after_move == piece):
                                flag = False
                    if flag:
                        value[move_pieces[i]] += weight[5]
            else:
                for player in player_pieces: #Check if hitting own pieces
                    if (player == player_after_move):
                        value[move_pieces[i]] += weight[5]


        #State 7: Hit itself home
        for n,enemy_piece in enumerate(enemy_pieces):
            for piece in enemy_piece:
                if (player_after_move == piece):
                    if (piece < 54 and piece != 0):
                        if self.enemy_safe(piece, enemy_piece, n):
                            value[move_pieces[i]] += weight[6] 
                            #print("Can kill itself")


        #State 8: field with possible to hit other home
        for enemy_piece in enemy_pieces:
            for enemy in enemy_piece:
                if enemy > player_pieces[move_pieces[i]] + 6 and enemy <= player_after_move + 6:
                    if (enemy < 54 and enemy != 0):
                        value[move_pieces[i]] += weight[7]

        #State 9: move out of danger
        for enemy_piece in enemy_pieces:
            for enemy in enemy_piece:
                if (player_pieces[move_pieces[i]] > enemy and player_pieces[move_pieces[i]] <= enemy + 6):
                    if (player_after_move > enemy + 6):
                        if (enemy < 54 and enemy != 0 and player_pieces[move_pieces[i]] < 54):
                            value[move_pieces[i]] += weight[8]

        #State 10: leave safe zone
        for field in self.safe_field:
            if player_pieces[move_pieces[i]] == field and player_pieces[move_pieces[i]] < 54:
                value[move_pieces[i]] += weight[9]

        return value


    #Pick the individuals to survive to next gen
    def survive(self, winrate):
        arr = np.array(winrate)
        keep_pop = arr.argsort()[-self.survive_num:][::-1]
        for i, pop in enumerate(keep_pop):
            self.new_weights[i] = self.weights[pop]

    #Create new individuals with random weights
    def make_new_pop(self):
        for i in range(self.new_pop):
            for n in range(len(self.temp_weights)):
                self.temp_weights[n] = np.random.randint(-100,100)
            self.new_weights[-(i+1)] = copy.deepcopy(self.temp_weights)

    #Mutate the surviving individuals
    def mutate(self, winrate, keep_pop):
        for i in range(len(winrate)-self.survive_num-self.new_pop):
            random_pop = np.random.randint(0, len(keep_pop))
            for n in range(len(self.weights[0])):
                self.new_weights[self.survive_num+i][n] = self.new_weights[random_pop][n] + np.random.randint(-self.mutate_rate,self.mutate_rate)
                #print(self.weights[random_pop][n])
                if (self.new_weights[self.survive_num-1+i][n] > 99):
                    self.new_weights[self.survive_num-1+i][n] = 99
                elif self.new_weights[self.survive_num-1+i][n] < -99:
                    self.new_weights[self.survive_num-1+i][n] = -99

    #Generate new individuals by crossing the surviving
    def crossover(self, winrate):
        arr = np.array(winrate)
        keep_pop = arr.argsort()[-self.survive_num:][::-1]
        for i in range(self.crossover_num):
            p1 = np.random.randint(0,len(keep_pop))
            parent1 = keep_pop[p1]
            p2 = np.random.randint(0,len(keep_pop))
            parent2 = keep_pop[p2]
            while parent1 == parent2:
                p2 = np.random.randint(0,len(keep_pop))
                parent2 = keep_pop[p2]
            bp = np.random.randint(1,len(self.weights[0])) #Breakpoint
            for n in range(len(self.weights[0])):
                if (n < bp):
                    #print("test1")
                    self.new_weights[self.survive_num + self.mutate_num+i][n] = self.weights[parent1][n]
                else:
                    #print("test2")
                    self.new_weights[self.survive_num + self.mutate_num+i][n] = self.weights[parent2][n]

    #Create new generation with new weights
    def update_weights(self):
        for m in range(self.generations):
            if m == 10: #Change number of games evaluated over based on generation
                self.num_games = 300
            elif m == 20:
                self.num_games = 500
            winrate = []
            print(self.weights)
            self.new_weights = copy.deepcopy(self.weights)
            
            for pop in range(self.pop_size): #Perform the fitness function calculation
                num_wins = 0
                self.pop = pop
                for i in range(self.num_games):
                    num_wins += game(self, self.weights[self.pop])
                winrate.append(num_wins/self.num_games)
                print("Win rate: %g Pop number: %d"  % (winrate[pop], (pop+1)))
            arr = np.array(winrate)
            
            keep_pop = arr.argsort()[-self.survive_num:][::-1]
            
            #survival
            self.survive(winrate)

            #New pop
            self.make_new_pop()

            #Mutation
            self.mutate(winrate, keep_pop)

            #crossover
            self.crossover(winrate)
            #----------------------------------------------------------------------------------------------------
            self.weights = copy.deepcopy(self.new_weights)
            
            winrate.sort()
            print("Win rate: ", winrate)
            print ("generation %d" % (m+1))
            print("Average winrate %g" % (np.sum(winrate)/self.pop_size))
            self.f.write("Generation %d: average winrate: %g \n" % ((m+1),np.sum(winrate)/self.pop_size))
            self.f2.write("Generation %d: highest winrate: %g \n" % ((m+1),np.max(winrate)))
            self.f3.write("Generation %d best weight: " % (m+1))
            for weight in self.weights[keep_pop[-1]]:
                self.f3.write(str(weight) + ", ")
            self.f3.write("\n")
            self.f4.write("Generation %d: lowest winrate: %g \n" % ((m+1),np.min(winrate)))
            #print(self.weights)
        

#Game
#player = QLearningPlayer('BestQTable.npy') #Give the path to the QTable.
def game(ga, weight):
    num_wins = 0
    import ludopy
    g = ludopy.Game()
    there_is_a_winner = False
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()
        if len(move_pieces):
            if (player_i == 0):
                piece_to_move = ga.eval(player_pieces, enemy_pieces, dice, move_pieces, weight)
            #elif (player_i == 2):
                #piece_to_move = player.getNextAction(player.getState(player_pieces,enemy_pieces), dice, move_pieces)
            else:
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
    if player_i == 0:
        return 1
    elif (player_i == 1):
        return 2
    elif (player_i == 2):
        return 3
    elif (player_i == 3):
        return 4
    else:
        return 0

    



if __name__ == '__main__':
    ga = genetic()
    ga.init_weights()
    #ga.update_weights()
    ga.eval_single()
    ga.shutdown()
