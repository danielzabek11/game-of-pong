import pygame
import random
import os
import pickle
import sys
import math

import config
from agent import QLearningAgent


class PongGame:
    def __init__(
        self,
        is_master_training_session=False,
        visualize_master_training=False,
        gameplay_difficulty_level="medium",
    ):
        pygame.init()
        self.is_master_training_session = is_master_training_session
        self.visualize_master_training = visualize_master_training
        self.gameplay_difficulty_level = gameplay_difficulty_level

        # Load settings based on game mode (training vs. gameplay)
        if self.is_master_training_session:
            self.current_env_ball_speed = config.TRAINING_ENVIRONMENT_SETTINGS[
                "ball_speed"
            ]
            self.ai_paddle_execution_speed = config.TRAINING_ENVIRONMENT_SETTINGS[
                "ai_paddle_speed"
            ]
            self.initial_ai_hp = config.TRAINING_ENVIRONMENT_SETTINGS["ai_hp"]
            self.ai_move_hp_penalty = config.TRAINING_ENVIRONMENT_SETTINGS[
                "ai_penalty_move"
            ]
            self.ai_still_hp_penalty = config.TRAINING_ENVIRONMENT_SETTINGS[
                "ai_penalty_still"
            ]
            self.ai_still_frames_threshold = config.TRAINING_ENVIRONMENT_SETTINGS[
                "ai_penalty_still_frames"
            ]
            self.ai_hp_regen_on_hit = config.TRAINING_ENVIRONMENT_SETTINGS[
                "ai_penalty_regen_on_hit"
            ]
            env_caption_name = f"MasterTrain ({config.TRAINING_ENVIRONMENT_SETTINGS['ball_speed']} ball, {config.TRAINING_ENVIRONMENT_SETTINGS['ai_paddle_speed']} AI speed)"
        else:
            self.current_env_ball_speed = config.GAMEPLAY_BALL_SPEEDS[
                self.gameplay_difficulty_level
            ]
            self.ai_paddle_execution_speed = config.GAMEPLAY_AI_PADDLE_EXECUTION_SPEED[
                self.gameplay_difficulty_level
            ]
            self.initial_ai_hp = config.GAMEPLAY_AI_HP_CONFIG[
                self.gameplay_difficulty_level
            ]
            self.ai_move_hp_penalty = config.GAMEPLAY_AI_PENALTY_CONFIG[
                self.gameplay_difficulty_level
            ]["move"]
            self.ai_still_hp_penalty = config.GAMEPLAY_AI_PENALTY_CONFIG[
                self.gameplay_difficulty_level
            ]["still"]
            self.ai_still_frames_threshold = config.GAMEPLAY_AI_PENALTY_CONFIG[
                self.gameplay_difficulty_level
            ]["still_frames"]
            self.ai_hp_regen_on_hit = config.GAMEPLAY_AI_PENALTY_CONFIG[
                self.gameplay_difficulty_level
            ]["regen_on_hit"]
            env_caption_name = self.gameplay_difficulty_level.capitalize()

        self.current_ball_speed_multiplier = 1.0

        # Pygame screen setup (visual or headless)
        if not self.is_master_training_session or self.visualize_master_training:
            self.screen = pygame.display.set_mode(
                (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
            )
            pygame.display.set_caption(f"Pong AI - {env_caption_name}")
            try:  # Font loading
                self.font = pygame.font.Font(None, 32)
                self.score_font = pygame.font.Font(None, 48)
            except:
                self.font = pygame.font.SysFont("arial", 30)
                self.score_font = pygame.font.SysFont("arial", 42)
        else:  # Headless for non-visualized training
            self.screen = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            self.font = pygame.font.Font(None, 36)
            self.score_font = pygame.font.Font(None, 36)

        # Initialize game objects and state
        self.clock = pygame.time.Clock()
        self.left_paddle = pygame.Rect(
            20,
            config.SCREEN_HEIGHT // 2 - config.PADDLE_HEIGHT // 2,
            config.PADDLE_WIDTH,
            config.PADDLE_HEIGHT,
        )
        self.right_paddle = pygame.Rect(
            config.SCREEN_WIDTH - 20 - config.PADDLE_WIDTH,
            config.SCREEN_HEIGHT // 2 - config.PADDLE_HEIGHT // 2,
            config.PADDLE_WIDTH,
            config.PADDLE_HEIGHT,
        )
        self.ball = pygame.Rect(
            config.SCREEN_WIDTH // 2 - config.BALL_SIZE // 2,
            config.SCREEN_HEIGHT // 2 - config.BALL_SIZE // 2,
            config.BALL_SIZE,
            config.BALL_SIZE,
        )
        self.reset_ball()

        self.left_hp = config.INITIAL_HP_PLAYER
        self.right_hp = self.initial_ai_hp
        self.ai_frames_still = 0
        self.particles = []
        self.left_score = 0
        self.right_score = 0
        self.running = True
        self.paused = False

        # Initialize the Q-learning agent
        initial_epsilon = (
            config.EPSILON_START
            if self.is_master_training_session
            else config.EPSILON_END
        )
        self.agent = QLearningAgent(
            actions=[-1, 0, 1],  # Down, Stay, Up
            alpha=config.ALPHA,
            gamma=config.GAMMA,
            epsilon=initial_epsilon,
            is_master_training_mode_for_agent=self.is_master_training_session,
        )
        self.num_bins = config.NUM_BINS
        self.qtable_filename = config.MASTER_Q_TABLE_FILENAME
        self.qtable_filename_best = config.MASTER_Q_TABLE_FILENAME_BEST

        # Load Q-table, prioritizing 'best' or resuming training
        if os.path.exists(self.qtable_filename_best):
            self._load_q_table()
        elif os.path.exists(self.qtable_filename) and self.is_master_training_session:
            self._load_q_table(filename_to_load=self.qtable_filename)
        elif not self.is_master_training_session and not os.path.exists(
            self.qtable_filename_best
        ):
            print(
                f"WARNING: Best Q-table '{self.qtable_filename_best}' not found. AI plays sub-optimally."
            )

        # AI reaction delay setup for gameplay
        self.ai_action_pending = 0
        self.ai_current_delay_frames = 0
        self.ai_target_delay_frames = 0
        if not self.is_master_training_session:
            min_d, max_d = config.GAMEPLAY_AI_REACTION_DELAY_FRAMES[
                self.gameplay_difficulty_level
            ]
            self.ai_target_delay_frames = (
                random.randint(min_d, max_d) if min_d <= max_d else min_d
            )

    def _load_q_table(self, filename_to_load=None):
        # Loads Q-table from file
        actual_filename_to_load = filename_to_load
        if actual_filename_to_load is None:
            if os.path.exists(self.qtable_filename_best):
                actual_filename_to_load = self.qtable_filename_best
            elif (
                os.path.exists(self.qtable_filename) and self.is_master_training_session
            ):
                actual_filename_to_load = self.qtable_filename  # For resuming training
            elif not self.is_master_training_session:
                self.agent.q_table = {}
                return

        if actual_filename_to_load is None:
            self.agent.q_table = {}
            return
        try:
            with open(actual_filename_to_load, "rb") as f:
                saved_data = pickle.load(f)
            if isinstance(saved_data, dict):
                self.agent.q_table = saved_data
            elif isinstance(saved_data, tuple) and len(saved_data) == 2:
                self.agent.q_table = saved_data[0]
                if self.is_master_training_session:
                    self.agent.epsilon = saved_data[1]
            if not self.is_master_training_session:
                self.agent.epsilon = config.EPSILON_END
            print(
                f"Loaded Q-table from '{actual_filename_to_load}'. Agent Epsilon: {self.agent.epsilon:.4f}"
            )
        except Exception as e:
            print(
                f"Error loading Q-table from '{actual_filename_to_load}': {e}. New Q-table."
            )
            self.agent.q_table = {}

    def _save_q_table(self, filename_to_save_to):
        # Saves current Q-table and agent's epsilon to a file.
        data_to_save = (self.agent.q_table, self.agent.epsilon)
        try:
            with open(filename_to_save_to, "wb") as f:
                pickle.dump(data_to_save, f)
            print(
                f"Q-table '{filename_to_save_to}' saved. Epsilon: {self.agent.epsilon:.4f}"
            )
        except Exception as e:
            print(f"ERROR: Could not save Q-table to '{filename_to_save_to}': {e}")

    def reset_ball(self):
        # Resets ball to center with random initial velocity.
        self.ball.center = (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2)
        self.current_ball_speed_multiplier = 1.0
        act_speed = self.current_env_ball_speed * self.current_ball_speed_multiplier
        vx = random.choice([-act_speed, act_speed])
        vy_abs = act_speed * random.uniform(0.3, 0.7)  # Ensure some vertical movement
        self.ball_vel = [vx, random.choice([-vy_abs, vy_abs])]

    def discretize(self, value, max_value, bins):
        # Converts a continuous value into a discrete bin index.
        if value >= max_value:
            return bins - 1
        if value <= 0:
            return 0
        bin_size = max_value / bins
        return 0 if bin_size == 0 else min(bins - 1, int(value / bin_size))

    def get_state(self):
        # Generates a discrete state representation for the Q-learning agent.
        paddle_y_bin = self.discretize(
            self.right_paddle.y,
            config.SCREEN_HEIGHT - config.PADDLE_HEIGHT,
            self.num_bins,
        )
        ball_x_bin = self.discretize(self.ball.x, config.SCREEN_WIDTH, self.num_bins)
        ball_y_bin = self.discretize(self.ball.y, config.SCREEN_HEIGHT, self.num_bins)
        ball_vx_sign = (
            1 if self.ball_vel[0] > 0.1 else (-1 if self.ball_vel[0] < -0.1 else 0)
        )
        ball_vy_sign = (
            1 if self.ball_vel[1] > 0.1 else (-1 if self.ball_vel[1] < -0.1 else 0)
        )
        return (ball_x_bin, ball_y_bin, ball_vx_sign, ball_vy_sign, paddle_y_bin)

    def get_reward(
        self, hit, miss, action_taken, ai_scored_this_step, player_scored_this_step
    ):
        # Calculates reward/penalty based on AI's action and game events.
        r = 0.0
        if hit:
            r += 5.0
        if miss:
            r -= 5.0
        if ai_scored_this_step:
            r += 10.0
        elif player_scored_this_step:
            r -= 10.0
        if self.ball_vel[0] < 0 and action_taken != 0:
            r += config.EXTRA_MOVE_PENALTY
        return r

    def _create_particles(self, x_pos, y_pos, color_val):
        # Generates visual particles for collisions.
        for _ in range(config.PARTICLE_COUNT):
            self.particles.append(
                {
                    "x": x_pos
                    + random.uniform(-config.BALL_SIZE / 3, config.BALL_SIZE / 3),
                    "y": y_pos
                    + random.uniform(-config.BALL_SIZE / 3, config.BALL_SIZE / 3),
                    "vx": random.uniform(
                        -config.PARTICLE_SPEED_MAX, config.PARTICLE_SPEED_MAX
                    ),
                    "vy": random.uniform(
                        -config.PARTICLE_SPEED_MAX, config.PARTICLE_SPEED_MAX
                    ),
                    "life": config.PARTICLE_LIFESPAN + random.randint(-5, 5),
                    "color": color_val,
                    "size": random.randint(
                        config.PARTICLE_SIZE_MIN, config.PARTICLE_SIZE_MAX
                    ),
                }
            )

    def handle_ball_bounce(self):
        # Manages ball collisions with top/bottom walls.
        collided_with_wall = False
        if self.ball.top <= 0:
            self.ball.top = 0
            self.ball_vel[1] = abs(self.ball_vel[1])
            collided_with_wall = True
        elif self.ball.bottom >= config.SCREEN_HEIGHT:
            self.ball.bottom = config.SCREEN_HEIGHT
            self.ball_vel[1] = -abs(self.ball_vel[1])
            collided_with_wall = True

        if collided_with_wall and (
            not self.is_master_training_session or self.visualize_master_training
        ):
            self._create_particles(
                self.ball.centerx, self.ball.centery, config.PARTICLE_COLOR_WALL
            )
            # Adds slight randomness to ball's angle after wall bounce
            current_angle = math.atan2(self.ball_vel[1], self.ball_vel[0])
            angle_delta = random.uniform(-0.15, 0.15)
            new_angle = current_angle + angle_delta
            speed_magnitude = (
                math.sqrt(self.ball_vel[0] ** 2 + self.ball_vel[1] ** 2)
                or self.current_env_ball_speed
            )
            self.ball_vel = [
                speed_magnitude * math.cos(new_angle),
                speed_magnitude * math.sin(new_angle),
            ]
            # Ensures ball maintains some vertical momentum
            min_vertical_speed_factor = 0.2
            if (
                abs(self.ball_vel[1])
                < min_vertical_speed_factor * self.current_env_ball_speed
            ):
                self.ball_vel[1] = math.copysign(
                    max(
                        abs(self.ball_vel[1]),
                        min_vertical_speed_factor * self.current_env_ball_speed,
                    ),
                    self.ball_vel[1],
                )

    def _apply_speed_multiplier(self):
        # Adjusts ball's speed based on the current speed multiplier.
        current_speed_magnitude = math.sqrt(
            self.ball_vel[0] ** 2 + self.ball_vel[1] ** 2
        )
        if current_speed_magnitude == 0:
            self.ball_vel = [
                self.current_env_ball_speed
                * self.current_ball_speed_multiplier
                * random.choice([-0.707, 0.707]),
                self.current_env_ball_speed
                * self.current_ball_speed_multiplier
                * random.choice([-0.707, 0.707]),
            ]
            return
        target_speed = self.current_env_ball_speed * self.current_ball_speed_multiplier
        if current_speed_magnitude > 0:
            scale_factor = target_speed / current_speed_magnitude
            self.ball_vel = [
                self.ball_vel[0] * scale_factor,
                self.ball_vel[1] * scale_factor,
            ]

    def _display_get_ready_message(self, duration_ms=1500):
        if not self.is_master_training_session or self.visualize_master_training:
            overlay_surf = pygame.Surface(
                (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA
            )
            overlay_surf.fill((0, 0, 0, 120))
            self.screen.blit(overlay_surf, (0, 0))
            ready_font = self.score_font
            ready_text_surf = ready_font.render("GET READY!", True, (255, 255, 100))
            text_rect = ready_text_surf.get_rect(
                center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2)
            )
            self.screen.blit(ready_text_surf, text_rect)
            pygame.display.flip()
            start_time = pygame.time.get_ticks()
            while pygame.time.get_ticks() - start_time < duration_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                self.clock.tick(30)

    def _handle_point_scored_common_reset(self):
        # Resets AI HP, paddle positions, and ball after a point.
        self.right_hp = self.initial_ai_hp
        self.left_paddle.centery = config.SCREEN_HEIGHT // 2
        self.right_paddle.centery = config.SCREEN_HEIGHT // 2
        self.reset_ball()

    def perform_post_score_sequence(self, get_ready_delay_ms=1500):
        self._handle_point_scored_common_reset()
        if not self.is_master_training_session or self.visualize_master_training:
            self.draw()
            self._display_get_ready_message(duration_ms=get_ready_delay_ms)

    def draw_pause_screen(self):
        overlay_surf = pygame.Surface(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA
        )
        overlay_surf.fill((0, 0, 0, 180))
        self.screen.blit(overlay_surf, (0, 0))
        pause_texts = [
            (self.score_font, "PAUSED", (255, 255, 255), -80),
            (self.font, "Press R to Resume", (200, 200, 200), 0),
            (self.font, "Press M for Menu", (200, 200, 200), 40),
            (self.font, "Press Q to Quit", (200, 200, 200), 80),
        ]
        for font_obj, text_str, color_rgb, y_offset in pause_texts:
            text_surf = font_obj.render(text_str, True, color_rgb)
            self.screen.blit(
                text_surf,
                (
                    config.SCREEN_WIDTH // 2 - text_surf.get_width() // 2,
                    config.SCREEN_HEIGHT // 2 + y_offset,
                ),
            )
        pygame.display.flip()

    def draw(self, current_episode_num=None):
        self.screen.fill((10, 10, 20))
        pygame.draw.line(
            self.screen,
            (50, 50, 70),
            (config.SCREEN_WIDTH // 2, 0),
            (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT),
            2,
        )
        # Paddles and ball
        pygame.draw.rect(
            self.screen, (230, 230, 230), self.left_paddle, border_radius=3
        )
        pygame.draw.rect(
            self.screen, (230, 230, 230), self.right_paddle, border_radius=3
        )
        pygame.draw.ellipse(self.screen, (255, 200, 0), self.ball)
        # Particles
        for p_data in self.particles:
            size_val = int(p_data["size"] * (p_data["life"] / config.PARTICLE_LIFESPAN))
            if size_val > 0:
                pygame.draw.circle(
                    self.screen,
                    p_data["color"],
                    (int(p_data["x"]), int(p_data["y"])),
                    size_val,
                )
        # Scores
        l_score_surf = self.score_font.render(
            str(self.left_score), True, (220, 220, 250)
        )
        r_score_surf = self.score_font.render(
            str(self.right_score), True, (220, 220, 250)
        )
        self.screen.blit(
            l_score_surf, (config.SCREEN_WIDTH // 4 - l_score_surf.get_width() // 2, 15)
        )
        self.screen.blit(
            r_score_surf,
            (config.SCREEN_WIDTH * 3 // 4 - r_score_surf.get_width() // 2, 15),
        )
        # HP display
        l_hp_surf = self.font.render("Player HP: ---", True, (100, 220, 100))
        ai_hp_color = (
            (220, 100, 100)
            if self.right_hp < self.initial_ai_hp * 0.3
            else (100, 220, 100)
        )  # Color changes if AI HP low
        r_hp_surf = self.font.render(f"AI HP: {int(self.right_hp)}", True, ai_hp_color)
        self.screen.blit(
            l_hp_surf, (30, config.SCREEN_HEIGHT - self.font.get_height() - 10)
        )
        self.screen.blit(
            r_hp_surf,
            (
                config.SCREEN_WIDTH - r_hp_surf.get_width() - 30,
                config.SCREEN_HEIGHT - self.font.get_height() - 10,
            ),
        )
        # Training information overlay
        if (
            self.is_master_training_session
            and self.visualize_master_training
            and current_episode_num is not None
        ):
            ep_text_str = f"Ep: {current_episode_num+1} Eps: {self.agent.epsilon:.3f}"
            ep_surf_obj = self.font.render(ep_text_str, True, (200, 200, 220))
            self.screen.blit(
                ep_surf_obj,
                (
                    config.SCREEN_WIDTH * 3 // 4 - ep_surf_obj.get_width() // 2,
                    15 + r_score_surf.get_height() + 5,
                ),
            )
        pygame.display.flip()

    def show_final_screen(self, message_str):
        # Displays the game over/win message.
        if not pygame.display.get_init():
            return
        self.screen.fill((10, 10, 20))
        final_s = self.score_font.render(message_str, True, (220, 220, 250))
        prompt_s = self.font.render("Press any key for Menu", True, (180, 180, 200))
        self.screen.blit(
            final_s,
            (
                config.SCREEN_WIDTH // 2 - final_s.get_width() // 2,
                config.SCREEN_HEIGHT // 2 - 50,
            ),
        )
        self.screen.blit(
            prompt_s,
            (
                config.SCREEN_WIDTH // 2 - prompt_s.get_width() // 2,
                config.SCREEN_HEIGHT // 2 + 20,
            ),
        )
        pygame.display.flip()
        waiting_for_input = True
        while waiting_for_input:
            self.clock.tick(30)
            for game_event in pygame.event.get():
                if game_event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if game_event.type == pygame.KEYDOWN:
                    waiting_for_input = False

    def run(self):
        # Main game loop for executing a single player vs AI game.
        if not self.is_master_training_session:
            self.agent.epsilon = config.EPSILON_END
            min_d, max_d = config.GAMEPLAY_AI_REACTION_DELAY_FRAMES[
                self.gameplay_difficulty_level
            ]
            self.ai_target_delay_frames = (
                random.randint(min_d, max_d) if min_d <= max_d else min_d
            )
            self.ai_current_delay_frames = 0
            self.ai_action_pending = 0

        while self.running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.paused = not self.paused
                    if self.paused:
                        if event.key == pygame.K_r:
                            self.paused = False
                        elif event.key == pygame.K_m:
                            self.running = False
                            return
                        elif event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit()
            if self.paused:
                if (
                    not self.is_master_training_session
                    or self.visualize_master_training
                ):
                    self.draw_pause_screen()
                self.clock.tick(15)
                continue

            # --- Player Input ---
            keys_pressed = pygame.key.get_pressed()
            if (
                keys_pressed[pygame.K_w] or keys_pressed[pygame.K_UP]
            ) and self.left_paddle.top > 0:
                self.left_paddle.y -= config.PLAYER_PADDLE_SPEED
            if (
                keys_pressed[pygame.K_s] or keys_pressed[pygame.K_DOWN]
            ) and self.left_paddle.bottom < config.SCREEN_HEIGHT:
                self.left_paddle.y += config.PLAYER_PADDLE_SPEED

            # --- AI Logic (Gameplay) ---
            current_game_state = self.get_state()
            action_to_be_executed = 0  # Default: AI stays still
            ai_has_moved_this_frame = False
            if self.right_hp > 0:
                # AI reaction delay and mistake probability
                if self.ai_current_delay_frames >= self.ai_target_delay_frames:
                    mistake_chance = config.GAMEPLAY_AI_MISTAKE_PROBABILITY[
                        self.gameplay_difficulty_level
                    ]
                    strategic_move = self.agent.choose_action(
                        current_game_state,
                        make_mistake_prob_for_gameplay=mistake_chance,
                    )
                    self.ai_action_pending = strategic_move
                    action_to_be_executed = self.ai_action_pending
                    self.ai_current_delay_frames = 0
                    min_d, max_d = config.GAMEPLAY_AI_REACTION_DELAY_FRAMES[
                        self.gameplay_difficulty_level
                    ]
                    self.ai_target_delay_frames = (
                        random.randint(min_d, max_d) if min_d <= max_d else min_d
                    )
                else:  # AI "thinking" (delaying)
                    action_to_be_executed = self.ai_action_pending
                    self.ai_current_delay_frames += 1

            # Execute AI move and HP penalties
            if self.right_hp > 0:
                if (
                    action_to_be_executed == 1
                    and self.right_paddle.bottom < config.SCREEN_HEIGHT
                ):  # Move down
                    self.right_paddle.y += self.ai_paddle_execution_speed
                    self.right_hp -= self.ai_move_hp_penalty
                    ai_has_moved_this_frame = True
                elif (
                    action_to_be_executed == -1 and self.right_paddle.top > 0
                ):  # Move up
                    self.right_paddle.y -= self.ai_paddle_execution_speed
                    self.right_hp -= self.ai_move_hp_penalty
                    ai_has_moved_this_frame = True

            # HP penalty for AI being still too long
            self.ai_frames_still = (
                0 if ai_has_moved_this_frame else self.ai_frames_still + 1
            )
            if self.ai_frames_still >= self.ai_still_frames_threshold:
                self.right_hp -= self.ai_still_hp_penalty
                self.ai_frames_still = 0
            self.right_hp = max(0, self.right_hp)

            # --- Ball Movement & Physics ---
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]
            self.handle_ball_bounce()

            # --- Paddle Collisions ---
            ball_hit_by_ai_this_frame = False
            paddle_deflection_angle_factor = 2.5
            # Left (player) paddle
            if self.ball.colliderect(self.left_paddle) and self.ball_vel[0] < 0:
                self.ball_vel[0] = abs(self.ball_vel[0])
                self.current_ball_speed_multiplier = min(
                    config.MAX_BALL_SPEED_MULTIPLIER,
                    self.current_ball_speed_multiplier
                    * config.BALL_SPEED_INCREMENT_FACTOR,
                )
                self._apply_speed_multiplier()
                if (
                    not self.is_master_training_session
                    or self.visualize_master_training
                ):
                    self._create_particles(
                        self.ball.centerx,
                        self.ball.centery,
                        config.PARTICLE_COLOR_PADDLE,
                    )
                offset_from_paddle_center = (
                    self.ball.centery - self.left_paddle.centery
                ) / (config.PADDLE_HEIGHT / 2)
                self.ball_vel[1] += (
                    offset_from_paddle_center
                    * paddle_deflection_angle_factor
                    * (abs(self.ball_vel[0]) * 0.15)
                )
            # Right (AI) paddle
            if self.ball.colliderect(self.right_paddle) and self.ball_vel[0] > 0:
                self.ball_vel[0] = -abs(self.ball_vel[0])
                self.current_ball_speed_multiplier = min(
                    config.MAX_BALL_SPEED_MULTIPLIER,
                    self.current_ball_speed_multiplier
                    * config.BALL_SPEED_INCREMENT_FACTOR,
                )
                self._apply_speed_multiplier()
                if (
                    not self.is_master_training_session
                    or self.visualize_master_training
                ):
                    self._create_particles(
                        self.ball.centerx,
                        self.ball.centery,
                        config.PARTICLE_COLOR_PADDLE,
                    )
                self.right_hp = min(
                    self.initial_ai_hp, self.right_hp + self.ai_hp_regen_on_hit
                )
                offset_from_paddle_center = (
                    self.ball.centery - self.right_paddle.centery
                ) / (config.PADDLE_HEIGHT / 2)
                self.ball_vel[1] += (
                    offset_from_paddle_center
                    * paddle_deflection_angle_factor
                    * (abs(self.ball_vel[0]) * 0.15)
                )

            max_vertical_velocity_ratio = 0.95
            if (
                abs(self.ball_vel[0]) > 0.1
                and abs(self.ball_vel[1])
                > abs(self.ball_vel[0]) * max_vertical_velocity_ratio
            ):
                self.ball_vel[1] = math.copysign(
                    abs(self.ball_vel[0]) * max_vertical_velocity_ratio,
                    self.ball_vel[1],
                )

            # --- Scoring and Post-Score Sequence ---
            point_was_scored_this_frame = False
            if self.ball.x + config.BALL_SIZE < 0:  # AI (right) scores
                self.right_score += 1
                point_was_scored_this_frame = True
            elif self.ball.x > config.SCREEN_WIDTH:  # Player (left) scores
                self.left_score += 1
                point_was_scored_this_frame = True

            if point_was_scored_this_frame:
                self.perform_post_score_sequence()  # Resets HP, paddles, ball; shows "Get Ready!"
                # Reset AI reaction delay logic for next round in gameplay
                if not self.is_master_training_session:
                    min_d, max_d = config.GAMEPLAY_AI_REACTION_DELAY_FRAMES[
                        self.gameplay_difficulty_level
                    ]
                    self.ai_target_delay_frames = (
                        random.randint(min_d, max_d) if min_d <= max_d else min_d
                    )
                    self.ai_current_delay_frames = 0
                    self.ai_action_pending = 0

            # --- Particle Update ---
            active_particles = []
            for particle in self.particles:
                particle["x"] += particle["vx"]
                particle["y"] += particle["vy"]
                particle["life"] -= 1
                if particle["life"] > 0:
                    active_particles.append(particle)
            self.particles = active_particles

            # --- Game End Check ---
            if (
                self.right_hp <= 0
                or self.left_score >= config.WIN_POINTS
                or self.right_score >= config.WIN_POINTS
            ):
                self.running = False

            if not self.is_master_training_session or self.visualize_master_training:
                self.draw()
            self.clock.tick(60)

        # --- Post-Game Loop ---
        if not self.is_master_training_session and pygame.display.get_init():
            final_message = "Game Over"
            if (
                self.right_hp <= 0
                and self.left_score < config.WIN_POINTS
                and self.right_score < config.WIN_POINTS
            ):
                final_message = "YOU WON! (AI HP Depleted)"
            elif self.left_score >= config.WIN_POINTS:
                final_message = "YOU WON! (Score Limit)"
            elif self.right_score >= config.WIN_POINTS:
                final_message = "AI WINS! (Score Limit)"
            self.show_final_screen(final_message)
