import sys
import numpy as np
import pandas as pd


def main():
    nfl_players = pd.read_csv("nfl_players.csv", index_col=0)
    freeagents = []
    for p in nfl_players.itertuples(index=False, name=None):
        if p[2] == "QB":
            freeagents.append(QB(*p))
        elif p[2] == "RB":
            freeagents.append(RB(*p))
        elif p[2] == "WR" or p[2] == "TE":
            freeagents.append(WR(*p))
        else:
            freeagents.append(NflPlayer(*p))

    num_competitors = 12
    rosters = [[] for _ in range(num_competitors)]  # empty rosters to start with

    num_rounds = 17
    turns = []
    # generate turns by snake order
    for i in range(num_rounds):
        turns += reversed(range(num_competitors)) if i % 2 else range(num_competitors)

    pick = 1

    state = DraftState(rosters, turns, freeagents)
    iterations = 1000
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        while state.GetMoves() != []:
            if (pick + 11) % 12 == 0:
                print("Round " + str(int((pick+11)/12)))
            if state.turns[0] == 2:
                move = UCT(state, iterations)
                for i in range(len(move)):
                    state.GetPlayers(move[i])
                print()
                while True:
                    player = input(str(state.turns[0] + 1) + ". ").lower()
                    if np.any([player == x.name.lower() for x in freeagents]):
                        break
                state.DoMove2(player)
                pick += 1
            else:
                move = UCT(state, iterations)[0]
                player = next(p for p in state.freeagents if p.position == move)
                print(str(state.turns[0] + 1) + ". " + player.name)
                state.DoMove(move)
                pick += 1
    else:
        while state.GetMoves() != []:
            if (pick + 11) % 12 == 0:
                print("Round " + str(int((pick+11)/12)))
            if state.turns[0] == 2:
                move = UCT(state, iterations)
                for i in range(len(move)):
                    state.GetPlayers(move[i])
                print()
            while True:
                player = input(str(state.turns[0] + 1) + ". ").lower()
                if np.any([player == x.name.lower() for x in freeagents]):
                    break
            state.DoMove2(player)
            pick += 1

    draft_results = pd.DataFrame({"Team " + str(i + 1): r for i, r in enumerate(state.rosters)})
    draft_results.to_csv('draft_results.csv', index=False)


class NflPlayer:
    def __init__(self, name, team, position, points, bye, pass_yds, pass_tds, rec, rec_yds, rec_tds, rush_yds, rush_tds):
        self.name = name
        self.team = team
        self.position = position
        self.points = points
        try:
            self.bye = int(bye)
        except:
            self.bye = np.NAN

    def __repr__(self):
        return " | ".join([self.name, self.position, self.team, str(self.bye)]).ljust(40,' ') + ((str(self.points) + " points").rjust(10,' ')).rjust(86, ' ')


class WR(NflPlayer):
    def __init__(self, name, team, position, points, bye, pass_yds, pass_tds, rec, rec_yds, rec_tds, rush_yds, rush_tds):
        super().__init__(name, team, position, points, bye, pass_yds, pass_tds, rec, rec_yds, rec_tds, rush_yds, rush_tds)
        try:
            self.rec = int(rec)
        except:
            self.rec = np.NAN
        try:
            self.rec_yds = int(rec_yds)
        except:
            self.rec_yds= np.NAN
        try:
            self.rec_tds = int(rec_tds)
        except:
            self.rec_tds = np.NAN

    def __repr__(self):
        return " | ".join([self.name, self.position, self.team, str(self.bye)]).ljust(40, ' ') + ((str(self.rec) + " rec").rjust(11, ' ') + (str(self.rec_yds) + " rec yds").rjust(16, ' ') + (str(self.rec_tds) + " rec_tds").rjust(14, ' ') + (str(self.points) + " points").rjust(14,' ')).rjust(86, ' ')


class QB(NflPlayer):
    def __init__(self, name, team, position, points, bye, pass_yds, pass_tds, rec, rec_yds, rec_tds, rush_yds, rush_tds):
        super().__init__(name, team, position, points, bye, pass_yds, pass_tds, rec, rec_yds, rec_tds, rush_yds, rush_tds)
        try:
            self.pass_yds = int(pass_yds)
        except:
            self.pass_yds = np.NAN
        try:
            self.pass_tds = int(pass_tds)
        except:
            self.pass_tds = np.NAN
        try:
            self.rush_yds = int(rush_yds)
        except:
            self.rush_yds = np.NAN
        try:
            self.rush_tds = int(rush_tds)
        except:
            self.rush_tds = np.NAN

    def __repr__(self):
        return " | ".join([self.name, self.position, self.team, str(self.bye)]).ljust(40, ' ') + ((str(self.pass_yds) + " pass_yds").rjust(17, ' ') + (str(self.pass_tds) + " pass_tds").rjust(15, ' ') + (str(self.rush_yds) + " rush yds").rjust(17, ' ') + (str(self.rush_tds) + " rush_tds").rjust(14, ' ') + (str(self.points) + " points").rjust(14,' ')).rjust(86, ' ')


class RB(WR):
    def __init__(self, name, team, position, points, bye, pass_yds, pass_tds, rec, rec_yds, rec_tds, rush_yds, rush_tds):
        super().__init__(name, team, position, points, bye, pass_yds, pass_tds, rec, rec_yds, rec_tds, rush_yds, rush_tds)
        try:
            self.rush_yds = int(rush_yds)
        except:
            self.rush_yds = np.NAN
        try:
            self.rush_tds = int(rush_tds)
        except:
            self.rush_tds = np.NAN

    def __repr__(self):
        return " | ".join([self.name, self.position, self.team, str(self.bye)]).ljust(40, ' ') + ((str(self.rush_yds) + " rush yds").rjust(17, ' ') + (str(self.rush_tds) + " rush_tds").rjust(14, ' ') + (str(self.rec) + " rec").rjust(11, ' ') + (str(self.rec_yds) + " rec yds").rjust(16, ' ') + (str(self.rec_tds) + " rec_tds").rjust(14, ' ') + (str(self.points) + " points").rjust(14,' ')).rjust(86, ' ')


class DraftState:
    def __init__(self, rosters, turns, freeagents, playerjm=None):
        self.rosters = rosters
        self.turns = turns
        self.freeagents = freeagents
        self.playerJustMoved = playerjm


def GetResult(self, playerjm):
    """ Get the game result from the viewpoint of playerjm.
    """
    if playerjm is None: return 0

    pos_wgts = {
        ("QB"): [.5, .2],
        ("WR"): [.8, .8, .8, .5, .3, .3],
        ("RB"): [.8, .8, .5, .3],
        ("TE"): [.7, .2],
        ("RB", "WR", "TE"): [.6],
        ("D"): [.3],
        ("K"): [.1]
    }

    result = 0
    # map the drafted players to the weights
    for p in self.rosters[playerjm]:
        max_wgt, _, max_pos, old_wgts = max(
            ((wgts[0], -len(lineup_pos), lineup_pos, wgts) for lineup_pos, wgts in pos_wgts.items()
             if p.position in lineup_pos),
            default=(0, 0, (), []))
        if max_wgt > 0:
            result += max_wgt * p.points
            old_wgts.pop(0)
            if not old_wgts:
                pos_wgts.pop(max_pos)

    # map the remaining weights to the top three free agents
    for pos, wgts in pos_wgts.items():
        result += np.mean([p.points for p in self.freeagents if p.position in pos][:3]) * sum(wgts)

    return result

DraftState.GetResult = GetResult


def GetMoves(self):
    """ Get all possible moves from this state.
    """
    pos_max = {"QB": 2, "WR": 7, "RB": 5, "TE": 2, "D": 1, "K": 1}

    if len(self.turns) == 0: return []

    roster_positions = np.array([p.position for p in self.rosters[self.turns[0]]], dtype=str)
    moves = [pos for pos, max_ in pos_max.items() if np.sum(roster_positions == pos) < max_]
    return moves

DraftState.GetMoves = GetMoves


def GetPlayers(self, move):
    """ Update a state by carrying out the given move.
        Must update playerJustMoved.
    """
    players = [p for p in self.freeagents if p.position == move]
    if move == "RB" or move == "WR":
        n = 10
    else:
        n = 5
    print("\n" + move)
    for i in range(0, n):
        print(str(players[i]))

DraftState.GetPlayers = GetPlayers


def DoMove(self, move):
    """ Update a state by carrying out the given move.
        Must update playerJustMoved.
    """
    player = next(p for p in self.freeagents if p.position == move)
    self.freeagents.remove(player)
    rosterId = self.turns.pop(0)
    self.rosters[rosterId].append(player)
    self.playerJustMoved = rosterId

DraftState.DoMove = DoMove


def DoMove2(self, player):
    """ Update a state by carrying out the given move.
        Must update playerJustMoved.
    """
    freeagents2 = [x.name.lower() for x in self.freeagents]
    player = self.freeagents.pop(freeagents2.index(player.lower()))
    rosterId = self.turns.pop(0)
    self.rosters[rosterId].append(player)
    self.playerJustMoved = rosterId

DraftState.DoMove2 = DoMove2


def Clone(self):
    """ Create a deep clone of this game state.
    """
    rosters = list(map(lambda r: r[:], self.rosters))
    st = DraftState(rosters, self.turns[:], self.freeagents[:],
            self.playerJustMoved)
    return st

DraftState.Clone = Clone

# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *
import random


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()  # future child nodes
        self.playerJustMoved = state.playerJustMoved  # the only part of the state that the Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        UCTK = 200
        s = sorted(self.childNodes, key=lambda c: c.wins / c.visits + UCTK * sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result


def UCT(rootstate, itermax, verbose=False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
    """

    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []:  # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []:  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []:  # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(
                node.playerJustMoved))  # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

        positions = sorted(rootnode.childNodes, key=lambda c: c.visits)

    try:
        return positions[-1].move, positions[-2].move, positions[-3].move# return the move that was most visited
    except:
        try:
            return positions[-1].move, positions[-2].move# return the move that was most visited
        except:
            return [positions[-1].move]# return the move that was most visited


if __name__ == '__main__':
    main()