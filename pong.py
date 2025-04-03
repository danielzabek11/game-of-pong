import pygame
import random
import os
import pickle
import sys
import math

SCREEN_WIDTH, SCREEN_HEIGHT = 740, 580
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 80
BALL_SIZE = 10
PADDLE_SPEED = 7
INITIAL_HP = 2000

DIFFICULTY_SPEEDS = {
    "easy": 3,
    "medium": 4,
    "hard": 5
}

# Q-learning parameters
ALPHA = 0.022
GAMMA = 0.966
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.987
NUM_BINS = 12


ALIGNMENT_THRESHOLD = 30
EXTRA_MOVE_PENALTY = -0.2
WIN_POINTS = 5

class QLearningAgent:
    def __init__(self, actions, alpha, gamma, epsilon):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]

    def choose_action(self, state):
        q_values = self.get_q_values(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        action_index = self.actions.index(action)
        best_next_q = max(next_q_values)
        q_values[action_index] += self.alpha * (reward + self.gamma * best_next_q - q_values[action_index])

class PongGame:
    def __init__(self, training=False, visualize_training=False, difficulty="medium"):
        pygame.init()
        self.training = training
        self.visualize_training = visualize_training
        self.difficulty = difficulty
        self.ball_speed = DIFFICULTY_SPEEDS[difficulty]
        
        if not training or (training and visualize_training):
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Pong with Q-Learning AI")
            self.font = pygame.font.Font(None, 36)
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.left_paddle = pygame.Rect(10, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle = pygame.Rect(SCREEN_WIDTH - 10 - PADDLE_WIDTH, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        
        self.ball = pygame.Rect(SCREEN_WIDTH // 2 - BALL_SIZE // 2, SCREEN_HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.reset_ball()
        
        self.left_hp = INITIAL_HP
        self.right_hp = INITIAL_HP
        
        if training:
            initial_epsilon = EPSILON_START
        else:
            initial_epsilon = EPSILON_END

        self.agent = QLearningAgent(actions=[-1, 0, 1], alpha=ALPHA, gamma=GAMMA, epsilon=initial_epsilon)
        self.num_bins = NUM_BINS

        self.left_score = 0
        self.right_score = 0
        self.running = True

        self.qtable_filename = f"q_table_{difficulty}.pkl"

        if not self.training and os.path.exists(self.qtable_filename):
            with open(self.qtable_filename, "rb") as f:
                self.agent.q_table = pickle.load(f)
            print("Loaded pre-trained Q-table from", self.qtable_filename)

    def reset_ball(self):
        self.ball.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.ball_vel = [
            random.choice([-self.ball_speed, self.ball_speed]),
            random.choice([-self.ball_speed, self.ball_speed])
        ]

    def discretize(self, value, max_value, bins):
        bin_size = max_value / bins
        return int(value / bin_size)

    def get_state(self):
        ball_x_bin = self.discretize(self.ball.x, SCREEN_WIDTH, self.num_bins)
        ball_y_bin = self.discretize(self.ball.y, SCREEN_HEIGHT, self.num_bins)
        paddle_y_bin = self.discretize(self.right_paddle.y, SCREEN_HEIGHT, self.num_bins)
        ball_vel_x_sign = 1 if self.ball_vel[0] > 0 else -1
        ball_vel_y_sign = 1 if self.ball_vel[1] > 0 else -1
        return (ball_x_bin, ball_y_bin, ball_vel_x_sign, ball_vel_y_sign, paddle_y_bin)

    def get_reward(self):
        if self.ball.colliderect(self.right_paddle):
            return 1
        elif self.ball.x > SCREEN_WIDTH:
            return -1
        else:
            return 0

    def handle_ball_bounce(self):
        if self.ball.top <= 0 or self.ball.bottom >= SCREEN_HEIGHT:
            self.ball_vel[1] = -self.ball_vel[1]
            current_angle = math.atan2(self.ball_vel[1], self.ball_vel[0])
            delta_angle = random.uniform(-0.1, 0.1)
            new_angle = current_angle + delta_angle
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel[0] = speed * math.cos(new_angle)
            self.ball_vel[1] = speed * math.sin(new_angle)

    def predict_ball_y(self, target_x):
        """Simulate the ball's trajectory until its x-coordinate reaches target_x.
           Returns the predicted y-coordinate at that x position.
        """
        if self.ball_vel[0] <= 0:
            return self.ball.centery
        
        sim_x = self.ball.x
        sim_y = self.ball.y
        vx = self.ball_vel[0]
        vy = self.ball_vel[1]
        
        while sim_x < target_x:
            sim_x += vx
            sim_y += vy
            if sim_y <= 0:
                sim_y = -sim_y
                vy = -vy
            elif sim_y >= SCREEN_HEIGHT - BALL_SIZE:
                sim_y = 2*(SCREEN_HEIGHT - BALL_SIZE) - sim_y
                vy = -vy
        return sim_y

    def draw(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), self.left_paddle)
        pygame.draw.rect(self.screen, (255, 255, 255), self.right_paddle)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
        left_text = self.font.render(str(self.left_score), True, (255, 255, 255))
        right_text = self.font.render(str(self.right_score), True, (255, 255, 255))
        self.screen.blit(left_text, (SCREEN_WIDTH // 4, 20))
        self.screen.blit(right_text, (SCREEN_WIDTH * 3 // 4, 20))
        left_hp_text = self.font.render(f"HP: {self.left_hp}", True, (255, 255, 255))
        right_hp_text = self.font.render(f"HP: {self.right_hp}", True, (255, 255, 255))
        self.screen.blit(left_hp_text, (50, SCREEN_HEIGHT - 40))
        self.screen.blit(right_hp_text, (SCREEN_WIDTH - 150, SCREEN_HEIGHT - 40))
        pygame.display.flip()

    def show_final_screen(self, message):
        """Display the final message (win/loss) and wait for any key press."""
        self.screen.fill((0, 0, 0))
        final_text = self.font.render(message, True, (255, 255, 255))
        prompt_text = self.font.render("Press any key to continue", True, (255, 255, 255))
        self.screen.blit(final_text, (SCREEN_WIDTH//2 - final_text.get_width()//2, SCREEN_HEIGHT//2 - 50))
        self.screen.blit(prompt_text, (SCREEN_WIDTH//2 - prompt_text.get_width()//2, SCREEN_HEIGHT//2 + 10))
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                    waiting = False

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            keys = pygame.key.get_pressed()
            if (keys[pygame.K_w] or keys[pygame.K_UP]) and self.left_paddle.top > 0 and self.left_hp > 0:
                self.left_paddle.y -= PADDLE_SPEED
                self.left_hp -= 1
            if (keys[pygame.K_s] or keys[pygame.K_DOWN]) and self.left_paddle.bottom < SCREEN_HEIGHT and self.left_hp > 0:
                self.left_paddle.y += PADDLE_SPEED
                self.left_hp -= 1

            state = self.get_state()
            if self.ball_vel[0] > 0 and self.right_hp > 0:
                predicted_y = self.predict_ball_y(self.right_paddle.x)
                if self.right_paddle.centery < predicted_y - ALIGNMENT_THRESHOLD:
                    action = 1
                    if self.right_paddle.bottom < SCREEN_HEIGHT:
                        self.right_paddle.y += PADDLE_SPEED
                        self.right_hp -= 1
                elif self.right_paddle.centery > predicted_y + ALIGNMENT_THRESHOLD:
                    action = -1
                    if self.right_paddle.top > 0:
                        self.right_paddle.y -= PADDLE_SPEED
                        self.right_hp -= 1
                else:
                    action = 0
            else:
                action = 0

            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]
            self.handle_ball_bounce()

            if self.ball.colliderect(self.left_paddle) and self.ball_vel[0] < 0:
                self.ball_vel[0] = -self.ball_vel[0]
            if self.ball.colliderect(self.right_paddle) and self.ball_vel[0] > 0:
                self.ball_vel[0] = -self.ball_vel[0]

            reward = self.get_reward()
            if self.ball_vel[0] <= 0 and action != 0:
                reward += EXTRA_MOVE_PENALTY

            next_state = self.get_state()
            self.agent.update(state, action, reward, next_state)

            if self.ball.x < 0:
                self.right_score += 1
                self.reset_ball()
            elif self.ball.x > SCREEN_WIDTH:
                self.left_score += 1
                self.reset_ball()

            if (self.left_score >= WIN_POINTS or self.right_score >= WIN_POINTS or 
                self.left_hp <= 0 or self.right_hp <= 0):
                self.running = False

            self.draw()
            self.clock.tick(60)

        if not self.training:
            if self.left_score >= WIN_POINTS or self.right_hp <= 0:
                final_message = "You Won!"
            elif self.right_score >= WIN_POINTS or self.left_hp <= 0:
                final_message = "You Lost!"
            else:
                final_message = "Game Over"
            self.show_final_screen(final_message)
        pygame.quit()

def train_agent(episodes=15000, visualize=False, difficulty="medium"):
    game = PongGame(training=True, visualize_training=visualize, difficulty=difficulty)
    for episode in range(episodes):
        game.reset_ball()
        game.left_paddle.y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
        game.right_paddle.y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
        game.left_hp = INITIAL_HP
        game.right_hp = INITIAL_HP
        episode_over = False

        while not episode_over:
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            if game.left_paddle.centery < game.ball.centery and game.left_paddle.bottom < SCREEN_HEIGHT and game.left_hp > 0:
                game.left_paddle.y += PADDLE_SPEED
                game.left_hp -= 1
            elif game.left_paddle.centery > game.ball.centery and game.left_paddle.top > 0 and game.left_hp > 0:
                game.left_paddle.y -= PADDLE_SPEED
                game.left_hp -= 1

            state = game.get_state()
            if game.ball_vel[0] > 0 and game.right_hp > 0:
                predicted_y = game.predict_ball_y(game.right_paddle.x)
                if game.right_paddle.centery < predicted_y - ALIGNMENT_THRESHOLD:
                    action = 1
                    if game.right_paddle.bottom < SCREEN_HEIGHT:
                        game.right_paddle.y += PADDLE_SPEED
                        game.right_hp -= 1
                elif game.right_paddle.centery > predicted_y + ALIGNMENT_THRESHOLD:
                    action = -1
                    if game.right_paddle.top > 0:
                        game.right_paddle.y -= PADDLE_SPEED
                        game.right_hp -= 1
                else:
                    action = 0
            else:
                action = 0

            game.ball.x += game.ball_vel[0]
            game.ball.y += game.ball_vel[1]
            game.handle_ball_bounce()

            if game.ball.colliderect(game.left_paddle) and game.ball_vel[0] < 0:
                game.ball_vel[0] = -game.ball_vel[0]
            if game.ball.colliderect(game.right_paddle) and game.ball_vel[0] > 0:
                game.ball_vel[0] = -game.ball_vel[0]

            reward = game.get_reward()
            if game.ball_vel[0] <= 0 and action != 0:
                reward += EXTRA_MOVE_PENALTY
            next_state = game.get_state()
            game.agent.update(state, action, reward, next_state)

            if game.ball.x < 0 or game.ball.x > SCREEN_WIDTH or game.left_hp <= 0 or game.right_hp <= 0:
                episode_over = True

            if visualize:
                game.draw()
                game.clock.tick(300)
        
        game.agent.epsilon = max(EPSILON_END, game.agent.epsilon * EPSILON_DECAY)

    with open(game.qtable_filename, "wb") as f:
        pickle.dump(game.agent.q_table, f)
    print("Training complete. Q-table saved to", game.qtable_filename)

def show_menu():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pong Game Menu")
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()
    selected_difficulty = None

    while selected_difficulty is None:
        screen.fill((0, 0, 0))
        title = font.render("Pong Game", True, (255, 255, 255))
        option1 = font.render("Press 1: Easy", True, (255, 255, 255))
        option2 = font.render("Press 2: Medium", True, (255, 255, 255))
        option3 = font.render("Press 3: Hard", True, (255, 255, 255))
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 100))
        screen.blit(option1, (SCREEN_WIDTH//2 - option1.get_width()//2, 200))
        screen.blit(option2, (SCREEN_WIDTH//2 - option2.get_width()//2, 250))
        screen.blit(option3, (SCREEN_WIDTH//2 - option3.get_width()//2, 300))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    selected_difficulty = "easy"
                elif event.key == pygame.K_2:
                    selected_difficulty = "medium"
                elif event.key == pygame.K_3:
                    selected_difficulty = "hard"
        clock.tick(60)
    pygame.quit()
    return selected_difficulty

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in [
        "train_easy", "train_medium", "train_hard",
        "train_easy_visual", "train_medium_visual", "train_hard_visual"
    ]:
        visualize = ("visual" in sys.argv[1])
        if "easy" in sys.argv[1]:
            difficulty = "easy"
        elif "medium" in sys.argv[1]:
            difficulty = "medium"
        elif "hard" in sys.argv[1]:
            difficulty = "hard"
        print("Starting training mode for", difficulty, "difficulty", "with visualization" if visualize else "")
        train_agent(episodes=15000, visualize=visualize, difficulty=difficulty)
    else:
        while True:
            difficulty = show_menu()
            game = PongGame(training=False, difficulty=difficulty)
            game.run()
