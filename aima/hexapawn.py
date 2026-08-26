# hexapawn_full.py
import copy
import random
import math

# ========================== GAME ==========================
class Hexapawn:
    def __init__(self):
        self.board = [['W','W','W'],['.','.','.'],['B','B','B']]
        self.current_player = 'W'

    def get_legal_moves(self, player=None):
        if player is None:
            player = self.current_player
        moves = []
        direction = -1 if player == 'W' else 1
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == player:
                    nr, nc = r + direction, c
                    if 0 <= nr < 3 and self.board[nr][nc] == '.':
                        moves.append(((r,c),(nr,nc)))
                    for dc in [-1,1]:
                        nr, nc = r + direction, c + dc
                        if 0 <= nr < 3 and 0 <= nc < 3:
                            if self.board[nr][nc] != '.' and self.board[nr][nc] != player:
                                moves.append(((r,c),(nr,nc)))
        return moves

    def make_move(self, move):
        new_game = copy.deepcopy(self)
        (fr,fc),(tr,tc) = move
        new_game.board[tr][tc] = new_game.board[fr][fc]
        new_game.board[fr][fc] = '.'
        new_game.current_player = 'B' if self.current_player == 'W' else 'W'
        return new_game

    def is_game_over(self):
        for c in range(3):
            if self.board[0][c] == 'W':
                return True, 'W'
            if self.board[2][c] == 'B':
                return True, 'B'
        if not self.get_legal_moves():
            winner = 'B' if self.current_player == 'W' else 'W'
            return True, winner
        return False, None

    def evaluate(self):
        score = 0
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == 'W':
                    score += 10 + (2 - r)
                elif self.board[r][c] == 'B':
                    score -= 10 + r
        return score

    def __str__(self):
        rows = [f"{r}  {' '.join(self.board[r])}" for r in range(3)]
        return "\n".join(rows) + "\n   c0 c1 c2"

# ========================== PLAYERS ==========================
class HumanPlayer:
    def get_move(self, game):
        moves = game.get_legal_moves()
        print("Your legal moves:")
        for i, (f,t) in enumerate(moves):
            print(f"  {i}: from ({f[0]},{f[1]}) to ({t[0]},{t[1]})")
        while True:
            try:
                choice = int(input("Enter move number: "))
                if 0 <= choice < len(moves):
                    return moves[choice]
                print("Invalid choice.")
            except ValueError:
                print("Enter a number.")

class AlphaBetaPlayer:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def get_move(self, game):
        _, best = self._alpha_beta(game, self.max_depth, -float('inf'), float('inf'), True)
        return best

    def _alpha_beta(self, game, depth, alpha, beta, is_max):
        over, winner = game.is_game_over()
        if over:
            if winner == 'W': return 1000 + depth, None
            elif winner == 'B': return -1000 - depth, None
            return 0, None
        if depth == 0:
            return game.evaluate(), None

        moves = game.get_legal_moves()
        best_move = None
        if is_max:
            val = -float('inf')
            for m in moves:
                new_game = game.make_move(m)
                score, _ = self._alpha_beta(new_game, depth-1, alpha, beta, False)
                if score > val:
                    val = score; best_move = m
                alpha = max(alpha, val)
                if beta <= alpha: break
            return val, best_move
        else:
            val = float('inf')
            for m in moves:
                new_game = game.make_move(m)
                score, _ = self._alpha_beta(new_game, depth-1, alpha, beta, True)
                if score < val:
                    val = score; best_move = m
                beta = min(beta, val)
                if beta <= alpha: break
            return val, best_move

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = game.get_legal_moves()

    def is_terminal(self):
        over, _ = self.game.is_game_over()
        return over

    def expand(self):
        move = self.untried_moves.pop()
        new_game = self.game.make_move(move)
        child = MCTSNode(new_game, self, move)
        self.children.append(child)
        return child

    def best_child(self, c=1.414):
        best = None
        best_uct = -float('inf')
        for child in self.children:
            if child.visits == 0:
                uct = float('inf')
            else:
                uct = (child.wins / child.visits) + c * math.sqrt(2 * math.log(self.visits) / child.visits)
            if uct > best_uct:
                best_uct = uct
                best = child
        return best

    def rollout(self):
        game = self.game
        while True:
            over, winner = game.is_game_over()
            if over:
                return winner
            moves = game.get_legal_moves()
            move = random.choice(moves)
            game = game.make_move(move)

    def backpropagate(self, winner):
        node = self
        while node is not None:
            node.visits += 1
            if node.game.current_player == winner:
                node.wins += 1.0
            node = node.parent

class MCTSPlayer:
    def __init__(self, num_iterations=100):
        self.num_iterations = num_iterations

    def get_move(self, game):
        root = MCTSNode(game)
        for _ in range(self.num_iterations):
            node = root
            while node.untried_moves == [] and not node.is_terminal():
                node = node.best_child()
            if node.untried_moves:
                node = node.expand()
            winner = node.rollout()
            node.backpropagate(winner)
        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        return best_child.move if best_child else None

# ========================== MAIN ==========================
def play_game(player1_type='human', player2_type='alphabeta', ai_depth=5, mcts_iter=100):
    game = Hexapawn()
    # player1 = White, player2 = Black
    if player1_type == 'human':
        p1 = HumanPlayer()
    elif player1_type == 'alphabeta':
        p1 = AlphaBetaPlayer(max_depth=ai_depth)
    elif player1_type == 'mcts':
        p1 = MCTSPlayer(num_iterations=mcts_iter)
    else:
        p1 = HumanPlayer()

    if player2_type == 'human':
        p2 = HumanPlayer()
    elif player2_type == 'alphabeta':
        p2 = AlphaBetaPlayer(max_depth=ai_depth)
    elif player2_type == 'mcts':
        p2 = MCTSPlayer(num_iterations=mcts_iter)
    else:
        p2 = AlphaBetaPlayer(max_depth=ai_depth)

    print("=== HEXAPAWN ===")
    print(f"White: {player1_type}, Black: {player2_type}")
    print("White moves up, Black moves down.\n")

    while True:
        print(game)
        over, winner = game.is_game_over()
        if over:
            if winner == 'W':
                print("White wins!")
            elif winner == 'B':
                print("Black wins!")
            else:
                print("Draw?")
            break

        if game.current_player == 'W':
            player = p1
            print("\nWhite's turn.")
        else:
            player = p2
            print("\nBlack's turn.")

        move = player.get_move(game)
        if move is None:
            print("No moves available.")
            break
        game = game.make_move(move)
        print(f"Move: from {move[0]} to {move[1]}\n")

if __name__ == "__main__":
    # เลือกโหมดที่ต้องการ โดยเปลี่ยนพารามิเตอร์ในบรรทัดนี้
    # play_game('human', 'alphabeta')         # มนุษย์ vs Alpha-Beta (ค่าเริ่มต้น)
    # play_game('human', 'mcts', mcts_iter=200)   # มนุษย์ vs MCTS
    # play_game('alphabeta', 'mcts', ai_depth=4, mcts_iter=200)  # Alpha-Beta vs MCTS
    play_game('human', 'alphabeta')