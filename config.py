# --- Core Game Constants ---
SCREEN_WIDTH = 740
SCREEN_HEIGHT = 580
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 80
BALL_SIZE = 10
PLAYER_PADDLE_SPEED = 10
WIN_POINTS = 5
INITIAL_HP_PLAYER = 9999

# --- Q-Learning Parameters ---
ALPHA = 0.1  # Learning rate
GAMMA = 0.97  # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.9995  # Rate at which epsilon decreases
NUM_BINS = 12  # For discretizing continuous states
EXTRA_MOVE_PENALTY = -0.25  # Penalty for AI moving when ball is moving away
MASTER_Q_TABLE_FILENAME = "q_table_master.pkl"
MASTER_Q_TABLE_FILENAME_BEST = "q_table_master_best.pkl"
EARLY_STOPPING_PATIENCE = 25  # For training: stop if no improvement
MIN_REWARD_IMPROVEMENT = 0.05  # Threshold for reward improvement

# --- AI Training Environment Specific Settings ---
TRAINING_ENVIRONMENT_SETTINGS = {
    "ball_speed": 4.5,
    "ai_paddle_speed": PLAYER_PADDLE_SPEED,
    "ai_hp": 1500,
    "ai_penalty_move": 0.5,
    "ai_penalty_still": 0.25,
    "ai_penalty_still_frames": 60,  # Frames before "still" penalty applies
    "ai_penalty_regen_on_hit": 25,
}

# --- Gameplay Difficulty Configurations (Player vs AI) ---
GAMEPLAY_BALL_SPEEDS = {"easy": 3.5, "medium": 4.5, "hard": 5.5}
GAMEPLAY_AI_PADDLE_EXECUTION_SPEED = {
    "easy": PLAYER_PADDLE_SPEED - 3,
    "medium": PLAYER_PADDLE_SPEED,
    "hard": PLAYER_PADDLE_SPEED,
}
GAMEPLAY_AI_HP_CONFIG = {"easy": 800, "medium": 900, "hard": 1000}
GAMEPLAY_AI_PENALTY_CONFIG = {
    "easy": {"move": 1, "still": 0.5, "still_frames": 90, "regen_on_hit": 25},
    "medium": {"move": 0.5, "still": 0.5, "still_frames": 60, "regen_on_hit": 25},
    "hard": {"move": 0.3, "still": 0.5, "still_frames": 45, "regen_on_hit": 25},
}
GAMEPLAY_AI_MISTAKE_PROBABILITY = {"easy": 0.10, "medium": 0.03, "hard": 0.01}
GAMEPLAY_AI_REACTION_DELAY_FRAMES = {"easy": (3, 6), "medium": (1, 2), "hard": (0, 1)}

# --- Ball Dynamics & Particles ---
BALL_SPEED_INCREMENT_FACTOR = 1.15
MAX_BALL_SPEED_MULTIPLIER = 3.0
PARTICLE_COUNT = 10
PARTICLE_LIFESPAN = 18
PARTICLE_SPEED_MAX = 3
PARTICLE_SIZE_MIN = 2
PARTICLE_SIZE_MAX = 4
PARTICLE_COLOR_PADDLE = (180, 180, 255)
PARTICLE_COLOR_WALL = (200, 200, 200)
