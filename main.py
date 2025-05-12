import pygame
import sys
import math

import config
from pong import PongGame


def train_master_agent_session(episodes_to_run=50000, visualize_session=False):
    game_instance = PongGame(
        is_master_training_session=True,
        visualize_master_training=visualize_session,
        gameplay_difficulty_level=None,
    )

    print(
        f"Starting MASTER AI training: {episodes_to_run} episodes, visualize: {visualize_session}"
    )
    print(f"  Regular Q-table checkpoints: '{game_instance.qtable_filename}'")
    print(f"  Best Q-table will be saved to: '{game_instance.qtable_filename_best}'")

    # Training metrics and control variables
    avg_rewards_history_list = []
    log_reporting_interval = 100  # Episodes between progress logs
    q_table_save_interval = 1000  # Episodes between saving Q-table
    max_steps_per_episode = 7000  # Safety break for very long episodes
    visualization_frame_rate = 120  # FPS for visualization

    best_average_reward = -float("inf")  # Tracks best performance for saving model
    patience_for_early_stopping = config.EARLY_STOPPING_PATIENCE

    # Main training loop
    for episode_index in range(episodes_to_run):
        game_instance.reset_ball()
        game_instance.left_paddle.y = game_instance.right_paddle.y = (
            config.SCREEN_HEIGHT // 2 - config.PADDLE_HEIGHT // 2
        )
        game_instance.left_hp = config.INITIAL_HP_PLAYER
        game_instance.right_hp = game_instance.initial_ai_hp
        game_instance.ai_frames_still = 0
        game_instance.left_score = 0
        game_instance.right_score = 0
        game_instance.particles = []

        current_state_for_training = game_instance.get_state()
        current_episode_total_reward = 0.0

        # Loop for steps within a single episode
        for step_num in range(max_steps_per_episode):
            if visualize_session:
                for event_obj in pygame.event.get():
                    if event_obj.type == pygame.QUIT:
                        print(
                            f"Training interrupted. Saving Q-table to '{game_instance.qtable_filename}'..."
                        )
                        game_instance._save_q_table(
                            filename_to_save_to=game_instance.qtable_filename
                        )
                        pygame.quit()
                        sys.exit()

            # Simple rule-based opponent (left paddle)
            if (
                game_instance.left_paddle.centery
                < game_instance.ball.centery - config.PADDLE_HEIGHT * 0.25
                and game_instance.left_paddle.bottom < config.SCREEN_HEIGHT
            ):
                game_instance.left_paddle.y += config.PLAYER_PADDLE_SPEED
            elif (
                game_instance.left_paddle.centery
                > game_instance.ball.centery + config.PADDLE_HEIGHT * 0.25
                and game_instance.left_paddle.top > 0
            ):
                game_instance.left_paddle.y -= config.PLAYER_PADDLE_SPEED

            action_chosen_for_training = game_instance.agent.choose_action(
                current_state_for_training
            )

            ai_moved_this_step = False
            if game_instance.right_hp > 0:
                if (
                    action_chosen_for_training == 1
                    and game_instance.right_paddle.bottom < config.SCREEN_HEIGHT
                ):  # Move down
                    game_instance.right_paddle.y += (
                        game_instance.ai_paddle_execution_speed
                    )
                    game_instance.right_hp -= game_instance.ai_move_hp_penalty
                    ai_moved_this_step = True
                elif (
                    action_chosen_for_training == -1
                    and game_instance.right_paddle.top > 0
                ):  # Move up
                    game_instance.right_paddle.y -= (
                        game_instance.ai_paddle_execution_speed
                    )
                    game_instance.right_hp -= game_instance.ai_move_hp_penalty
                    ai_moved_this_step = True

            game_instance.ai_frames_still = (
                0 if ai_moved_this_step else game_instance.ai_frames_still + 1
            )
            if (
                game_instance.ai_frames_still >= game_instance.ai_still_frames_threshold
            ):  # Penalty for being still
                game_instance.right_hp -= game_instance.ai_still_hp_penalty
                game_instance.ai_frames_still = 0
            game_instance.right_hp = max(0, game_instance.right_hp)

            # Update ball position and handle wall bounces
            game_instance.ball.x += game_instance.ball_vel[0]
            game_instance.ball.y += game_instance.ball_vel[1]
            game_instance.handle_ball_bounce()

            # Handle paddle collisions
            hit_ball_in_training_step, missed_ball_in_training_step = False, False
            paddle_deflection_factor_training = 2.5

            # Left (player) paddle collision
            if (
                game_instance.ball.colliderect(game_instance.left_paddle)
                and game_instance.ball_vel[0] < 0
            ):
                game_instance.ball_vel[0] = abs(game_instance.ball_vel[0])
                game_instance.current_ball_speed_multiplier = min(
                    config.MAX_BALL_SPEED_MULTIPLIER,
                    game_instance.current_ball_speed_multiplier
                    * config.BALL_SPEED_INCREMENT_FACTOR,
                )
                game_instance._apply_speed_multiplier()
                if visualize_session:
                    game_instance._create_particles(
                        game_instance.ball.centerx,
                        game_instance.ball.centery,
                        config.PARTICLE_COLOR_PADDLE,
                    )
                offset_val_training = (
                    game_instance.ball.centery - game_instance.left_paddle.centery
                ) / (config.PADDLE_HEIGHT / 2)
                game_instance.ball_vel[1] += (
                    offset_val_training
                    * paddle_deflection_factor_training
                    * (abs(game_instance.ball_vel[0]) * 0.15)
                )

            # Right (AI) paddle collision
            if (
                game_instance.ball.colliderect(game_instance.right_paddle)
                and game_instance.ball_vel[0] > 0
            ):
                game_instance.ball_vel[0] = -abs(game_instance.ball_vel[0])
                game_instance.current_ball_speed_multiplier = min(
                    config.MAX_BALL_SPEED_MULTIPLIER,
                    game_instance.current_ball_speed_multiplier
                    * config.BALL_SPEED_INCREMENT_FACTOR,
                )
                game_instance._apply_speed_multiplier()
                if visualize_session:
                    game_instance._create_particles(
                        game_instance.ball.centerx,
                        game_instance.ball.centery,
                        config.PARTICLE_COLOR_PADDLE,
                    )
                hit_ball_in_training_step = True
                game_instance.right_hp = min(
                    game_instance.initial_ai_hp,
                    game_instance.right_hp + game_instance.ai_hp_regen_on_hit,
                )  # AI regens HP
                offset_val_training = (
                    game_instance.ball.centery - game_instance.right_paddle.centery
                ) / (config.PADDLE_HEIGHT / 2)
                game_instance.ball_vel[1] += (
                    offset_val_training
                    * paddle_deflection_factor_training
                    * (abs(game_instance.ball_vel[0]) * 0.15)
                )

            # Clamp ball's vertical velocity
            max_y_vel_ratio_training = 0.95
            if (
                abs(game_instance.ball_vel[0]) > 0.1
                and abs(game_instance.ball_vel[1])
                > abs(game_instance.ball_vel[0]) * max_y_vel_ratio_training
            ):
                game_instance.ball_vel[1] = math.copysign(
                    abs(game_instance.ball_vel[0]) * max_y_vel_ratio_training,
                    game_instance.ball_vel[1],
                )

            # Scoring logic for training
            player_scored_this_step, ai_scored_this_step = False, False
            if game_instance.ball.x + config.BALL_SIZE < 0:  # AI scores
                game_instance.right_score += 1
                ai_scored_this_step = True
                game_instance.reset_ball()
            elif game_instance.ball.x > config.SCREEN_WIDTH:  # Player scores
                game_instance.left_score += 1
                missed_ball_in_training_step = True  # AI missed
                player_scored_this_step = True
                game_instance.reset_ball()

            # Determine if episode has ended
            episode_ended_this_step = (
                game_instance.right_hp <= 0
                or game_instance.left_score >= config.WIN_POINTS
                or game_instance.right_score >= config.WIN_POINTS
                or step_num >= max_steps_per_episode - 1
            )

            # Calculate reward for the current step
            reward_for_this_step = game_instance.get_reward(
                hit_ball_in_training_step,
                missed_ball_in_training_step,
                action_chosen_for_training,
                ai_scored_this_step,
                player_scored_this_step,
            )
            # Apply large terminal rewards/penalties if episode ended
            if episode_ended_this_step:
                if (
                    game_instance.right_hp <= 0
                    or game_instance.left_score >= config.WIN_POINTS
                ):  # AI lost
                    reward_for_this_step -= 20.0
                elif game_instance.right_score >= config.WIN_POINTS:  # AI won
                    reward_for_this_step += 20.0

            current_episode_total_reward += reward_for_this_step
            next_state_for_training = game_instance.get_state()

            # Agent learns from the experience
            game_instance.agent.update(
                current_state_for_training,
                action_chosen_for_training,
                reward_for_this_step,
                next_state_for_training,
            )
            current_state_for_training = next_state_for_training

            if visualize_session:
                active_particles_viz = []
                for p_obj_viz in game_instance.particles:
                    p_obj_viz["x"] += p_obj_viz["vx"]
                    p_obj_viz["y"] += p_obj_viz["vy"]
                    p_obj_viz["life"] -= 1
                    if p_obj_viz["life"] > 0:
                        active_particles_viz.append(p_obj_viz)
                game_instance.particles = active_particles_viz
                game_instance.draw(current_episode_num=episode_index)
                game_instance.clock.tick(visualization_frame_rate)

            if episode_ended_this_step:
                break

        # After each episode, decay epsilon for exploration-exploitation balance
        game_instance.agent.epsilon = max(
            config.EPSILON_END, game_instance.agent.epsilon * config.EPSILON_DECAY
        )
        avg_rewards_history_list.append(current_episode_total_reward)

        # Periodically log progress and check for early stopping
        if (episode_index + 1) % log_reporting_interval == 0:
            last_n_rewards = avg_rewards_history_list[-log_reporting_interval:]
            avg_reward_value = (
                sum(last_n_rewards) / len(last_n_rewards) if last_n_rewards else 0.0
            )
            print(
                f"Ep {episode_index+1}/{episodes_to_run}. Eps: {game_instance.agent.epsilon:.4f}. Avg Rwd: {avg_reward_value:.2f}. Best Avg Rwd: {best_average_reward:.2f}. Patience: {patience_for_early_stopping}. Q#: {len(game_instance.agent.q_table)}"
            )

            # Save model if it's the best so far
            if avg_reward_value > best_average_reward + config.MIN_REWARD_IMPROVEMENT:
                print(
                    f"  Improvement! New best: {avg_reward_value:.2f}. Old best: {best_average_reward:.2f}. Resetting patience."
                )
                best_average_reward = avg_reward_value
                patience_for_early_stopping = config.EARLY_STOPPING_PATIENCE
                print(
                    f"  Saving new best Q-table to '{game_instance.qtable_filename_best}'."
                )
                game_instance._save_q_table(
                    filename_to_save_to=game_instance.qtable_filename_best
                )
            else:
                patience_for_early_stopping -= 1

            if patience_for_early_stopping <= 0:  # Trigger early stopping
                print(
                    f"Early stopping at ep {episode_index+1}. No improvement for {config.EARLY_STOPPING_PATIENCE} intervals."
                )
                break

        # Periodically save the current Q-table as a backup
        if (episode_index + 1) % q_table_save_interval == 0:
            print(
                f"Periodic Q-table save for '{game_instance.qtable_filename}' at ep {episode_index+1}."
            )
            game_instance._save_q_table(
                filename_to_save_to=game_instance.qtable_filename
            )

    # After all training episodes (or early stopping)
    print(f"Training finished after {episode_index + 1} episodes.")
    if patience_for_early_stopping <= 0:
        print("Reason: Early stopping.")
    else:
        print(f"Reason: Completed all {episodes_to_run} episodes.")

    print(f"Saving final Q-table state to '{game_instance.qtable_filename}'.")
    game_instance._save_q_table(filename_to_save_to=game_instance.qtable_filename)
    print(
        f"Best avg reward: {best_average_reward:.2f}. Best Q-table in '{game_instance.qtable_filename_best}'."
    )

    if visualize_session and pygame.get_init():
        pygame.quit()


# Displays the main menu and handles user selections.
def show_menu():
    if not pygame.get_init():
        pygame.init()
    screen_surf = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Pong AI Menu")
    try:
        menu_font, title_font = pygame.font.Font(None, 40), pygame.font.Font(None, 60)
    except:
        menu_font, title_font = pygame.font.SysFont("arial", 36), pygame.font.SysFont(
            "arial", 54
        )

    game_clock = pygame.time.Clock()
    menu_items = [
        "Play Easy",
        "Play Medium",
        "Play Hard",
        "Train Master AI (Visual)",
        "Exit",
    ]
    selected_item_index = 0
    key_to_action_map = {
        pygame.K_1 + i: item for i, item in enumerate(menu_items) if i < 9
    }
    if len(menu_items) > 9:
        key_to_action_map[pygame.K_0] = menu_items[9]

    while True:  # Menu loop
        screen_surf.fill((10, 20, 40))
        title_surface = title_font.render("PONG AI CHALLENGE", True, (220, 220, 250))
        screen_surf.blit(
            title_surface,
            (config.SCREEN_WIDTH // 2 - title_surface.get_width() // 2, 70),
        )

        for i, option_text in enumerate(menu_items):
            text_color = (
                (255, 255, 100) if i == selected_item_index else (200, 200, 220)
            )
            number_prefix = f"{i+1 if i<9 else(0 if i==9 else '')}."  # e.g., "1.", "2."
            item_display_text = f"{number_prefix} {option_text}"
            text_render_surface = menu_font.render(item_display_text, True, text_color)
            screen_surf.blit(
                text_render_surface,
                (
                    config.SCREEN_WIDTH // 2 - text_render_surface.get_width() // 2,
                    180 + i * 50,
                ),
            )
        pygame.display.flip()

        # Handle menu input
        for menu_event in pygame.event.get():
            if menu_event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if menu_event.type == pygame.KEYDOWN:
                action_to_take = None
                # Arrow key navigation
                if menu_event.key in (pygame.K_w, pygame.K_UP):
                    selected_item_index = (
                        selected_item_index - 1 + len(menu_items)
                    ) % len(menu_items)
                elif menu_event.key in (pygame.K_s, pygame.K_DOWN):
                    selected_item_index = (selected_item_index + 1) % len(menu_items)
                # Number key selection
                elif (
                    menu_event.key in key_to_action_map
                    and key_to_action_map[menu_event.key] is not None
                ):
                    action_to_take = key_to_action_map[menu_event.key]
                    selected_item_index = menu_items.index(action_to_take)
                # Enter/Space selection
                elif menu_event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    action_to_take = menu_items[selected_item_index]

                # Process the selected action
                if action_to_take:
                    if "Play" in action_to_take:
                        return action_to_take.split(" ")[
                            1
                        ].lower()  # Return difficulty string
                    elif "Train Master AI (Visual)" == action_to_take:
                        train_master_agent_session(
                            episodes_to_run=30000, visualize_session=True
                        )
                        # Re-initialize Pygame for menu if training quit it
                        if not pygame.get_init():
                            pygame.init()
                        screen_surf = pygame.display.set_mode(
                            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
                        )
                        pygame.display.set_caption("Pong AI Menu")
                        try:
                            menu_font, title_font = pygame.font.Font(
                                None, 40
                            ), pygame.font.Font(None, 60)
                        except:
                            menu_font, title_font = pygame.font.SysFont(
                                "arial", 36
                            ), pygame.font.SysFont("arial", 54)
                    elif "Exit" == action_to_take:
                        pygame.quit()
                        sys.exit()
        game_clock.tick(30)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith(
        "train"
    ):  # e.g., "train" or "train_visual"
        is_visual_training_cli = "visual" in sys.argv[1]

        # Default number of episodes for CLI training
        default_episodes_cli_visual = 25000
        default_episodes_cli_novisual = 50000

        num_episodes_for_training = (
            default_episodes_cli_visual
            if is_visual_training_cli
            else default_episodes_cli_novisual
        )

        print(
            f"CLI Training: visualize={is_visual_training_cli}, episodes={num_episodes_for_training}"
        )
        train_master_agent_session(
            episodes_to_run=num_episodes_for_training,
            visualize_session=is_visual_training_cli,
        )

        if pygame.get_init():
            pygame.quit()
        sys.exit()
    else:
        while True:
            selected_difficulty_level = show_menu()

            if selected_difficulty_level:
                game_session = PongGame(
                    is_master_training_session=False,
                    gameplay_difficulty_level=selected_difficulty_level,
                )
                game_session.run()
