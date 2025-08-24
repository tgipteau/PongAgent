import pygame
import random 
import math
from agent import Agent
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

#kMode = "play" 
kMode = "train"

FPS = 60
TRAIN_FPS = 240
rewards = {"ball_hit_player": 20, "ball_hit_target": 30, "ball_missed": -100, "step": 0}

kScreenWidth = 800
kScreenHeight = 600

pygame.init()
font = pygame.font.SysFont("Arial", 24)


class Player:
    def __init__(self):
        self.x = 100
        self.y = 300
        self.length = 100
        self.width = 10

    def move(self, action):
        if action == 0:
            self.y += 5
        elif action == 1:
            self.y -= 5
        elif action == 2:
            pass
     

    def draw(self, screen):
        pygame.draw.line(screen, (0, 0, 255), (self.x, self.y), (self.x, self.y+self.length), self.width)

class Ball:
    def __init__(self):
        self.x = kScreenWidth - 100
        self.y = kScreenHeight // 2
        self.vx = -20 # effectively horizontal speed / game speed
        self.vy = random.choice([-1, 1])
        self.radius = 10    
        self.color = (255, 0, 0)

        self.state = "in_play"  # can be "in_play", "missed"

    def move(self):
        self.x += self.vx
        self.y += self.vy

        if self.y <= self.radius or self.y >= kScreenHeight - self.radius:
            self.vy = -self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

class Target:
    def __init__(self):
        self.length = 160
        self.top_y =  random.randint(0, kScreenHeight-self.length)
        self.bottom_y = self.top_y + self.length
        self.color = (0, 255, 0)
        self.width = 30

        self.state = "in_play"  # can be "in_play", "hit"
        

    def move(self):
        self.top_y = random.randint(0, kScreenHeight-self.length)
        self.bottom_y = self.top_y + self.length

        #print(f"New target position: {self.top_y} to {self.bottom_y}")

    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (kScreenWidth - self.width, self.top_y, self.width, self.bottom_y - self.top_y))

class Game:
    def __init__(self):
        self.player = Player()
        self.ball = Ball()
        self.target = Target()
        self.score = 0
        self.done = False

    def reset(self):
        self.player = Player()
        self.ball = Ball()
        self.target = Target()
        self.score = 0
        self.done = False

    def handle_collisions_and_misses(self):

        reward = rewards["step"]

        # Check collision with player
        player_start_y = self.player.y
        player_end_y = self.player.y + self.player.length

        if (self.ball.x - self.ball.radius <= self.player.x + self.player.width and
            player_start_y <= self.ball.y <= player_end_y):

            #print("Ball hit player")
            # reflect ball according to where it hit the player
            hit_pos = (self.ball.y - player_start_y) / self.player.length # 0 to 1
            new_angle = (hit_pos - 0.5) * math.pi / 2  # -pi/4 to pi/4
            speed = math.sqrt(self.ball.vx**2 + self.ball.vy**2)
            self.ball.vx = speed * math.cos(new_angle)
            self.ball.vy = speed * math.sin(new_angle)
            self.ball.x = self.player.x + self.player.width + self.ball.radius  # Prevent sticking
            self.ball.state = "in_play"

            reward = rewards["ball_hit_player"]

        # Check collision with target

        target_start_y = self.target.top_y
        target_end_y = self.target.bottom_y 

        if (self.ball.x + self.ball.radius >= kScreenWidth - self.target.width and
            target_start_y <= self.ball.y <= target_end_y):

            #print("Ball hit target")
            #reflect ball according to where it hit the target
            hit_pos = (self.ball.y - target_start_y) / self.target.length # 0 to 1
            new_angle = (hit_pos - 0.5) * math.pi / 2  # -pi/4 to pi/4
            speed = math.sqrt(self.ball.vx**2 + self.ball.vy**2)
            self.ball.vx = -speed * math.cos(new_angle)
            self.ball.vy = speed * math.sin(new_angle)
            self.ball.x = kScreenWidth - self.target.width - self.ball.radius  # Prevent sticking
            self.target.state = "hit"

            reward = rewards["ball_hit_target"]
            self.target.move()
            self.score += 1

        if self.ball.x < 0 or self.ball.x > kScreenWidth:
            #print("Ball missed")
            self.ball.state = "missed"
            self.reset()
            reward = rewards["ball_missed"]
        
        return reward

        
    def step(self, action):

        self.player.move(action)
        self.ball.move()
        reward = self.handle_collisions_and_misses()
    
        return reward

    def render(self, screen):
        screen.fill((0, 0, 0))
        self.player.draw(screen)
        self.ball.draw(screen)
        self.target.draw(screen)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))


def main():

    if kMode == "play":
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Squash Agent")
        clock = pygame.time.Clock()

        game = Game()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            keys = pygame.key.get_pressed()
            if keys[pygame.K_DOWN]:
                action = 0
            elif keys[pygame.K_UP]:
                action = 1
            elif keys[pygame.K_LEFT]:
                action = 2
            elif keys[pygame.K_RIGHT]:
                action = 3
            else:
                action = -1

            reward = game.step(action)
            if reward != 0:
                print(f"Reward: {reward}")

            game.render(screen)
            pygame.display.flip()

            timeDelta = clock.tick(FPS) / 1000.0

    elif kMode == "train":

        agent = Agent(state_size=6, action_size=3, cfg=config["agent"])  

        clock = pygame.time.Clock()
        game = Game()
        running = True
        episode_reward = 0
        episode = 0

        state = [game.player.y, game.ball.x, game.ball.y, game.ball.vx, game.ball.vy, game.target.top_y]

        while running:

            # Choisir action
            action = agent.act(state)
            # Appliquer action et obtenir reward
            reward = game.step(action)
            # Prochain état
            next_state = [game.player.y, game.ball.x, game.ball.y, game.ball.vx, game.ball.vy, game.target.top_y]
            done = (game.ball.state == "missed")

            if done:
                next_state = None  # convention dans ton Agent
                episode += 1
                print(f"Episode {episode} finished. Reward={episode_reward:.1f}, epsilon={agent.epsilon:.3f}")
                episode_reward = 0
                # game.step s’occupe déjà du reset → donc pas besoin de game.reset()

            # Stocker dans la mémoire
            agent.remember(state, action, next_state, reward)

            # Réentraîner (si assez d’expériences en mémoire)
            agent.replay()

            # Décrémenter epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            # Mettre à jour le réseau cible périodiquement
            agent.step_count += 1
            if agent.step_count % agent.update_target_every == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())

            # Avancer
            state = next_state if next_state is not None else [game.player.y, game.ball.x, game.ball.y, game.ball.vx, game.ball.vy, game.target.top_y]
            episode_reward += reward

            # Tick pygame
            timeDelta = clock.tick(TRAIN_FPS) / 1000.0
        
if __name__ == "__main__":
    main()
        
    
