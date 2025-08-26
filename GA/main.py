import pygame
import random 
import math
from agent import Agent
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
from tqdm import tqdm
import sys

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mode = sys.argv[1].lower() if len(sys.argv) > 1 else "play" # play is default mode / play, trainleft, trainright

FPS = 100
rewards = {"ball_hit_player": 100, "ball_missed": -100, "step": 0, "dist_reward_factor": 3e-3, "static_reward": 0.1}

kScreenWidth = 800
kScreenHeight = 600


class Player:
    def __init__(self, x):
        self.x = x
        self.y = 300
        self.length = 100
        self.width = 10

    def move(self, action):
        if action == 0:
            self.y += 10
        elif action == 1:
            self.y -= 10
        elif action == 2:
            pass
     

    def draw(self, screen):
        pygame.draw.line(screen, (255,255,255), (self.x, self.y), (self.x, self.y+self.length), self.width)

class Ball:
    def __init__(self):
        self.x = kScreenWidth - 200
        self.y = kScreenHeight // 2
        self.vx = -10 # effectively horizontal speed / game speed
        self.vy = random.randrange(-6,6)
        self.radius = 10    
        self.color = (255, 255, 255)

        self.state = "in_play"  # can be "in_play", "missed"

    def move(self):
        self.x += self.vx
        self.y += self.vy


    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

class Game:
    def __init__(self, two_players):
        self.player = Player(x=100)
        self.ball = Ball()
        self.score = 0
        self.done = False

        self.two_players = two_players
        if two_players :
            self.player2 = Player(x=kScreenWidth-100)

    def reset(self):
        self.player = Player(x=100)
        self.ball = Ball()
        self.score = 0
        self.done = False
        if self.two_players:
            self.player2 = Player(kScreenWidth-100)

    def handle_collisions_and_misses(self):


        # Check collision with player1
        player_start_y = self.player.y
        player_end_y = self.player.y + self.player.length

        if (self.ball.x - self.ball.radius <= self.player.x + self.player.width and
            player_start_y <= self.ball.y <= player_end_y):

            # reflect ball according to where it hit the player
            hit_pos = (self.ball.y - player_start_y) / self.player.length # 0 to 1
            new_angle = (hit_pos - 0.5) * math.pi / 2  # -pi/4 to pi/4
            speed = math.sqrt(self.ball.vx**2 + self.ball.vy**2)
            self.ball.vx = speed * math.cos(new_angle)
            self.ball.vy = speed * math.sin(new_angle)
            self.ball.x = self.player.x + self.player.width + self.ball.radius  # Prevent sticking
            self.ball.state = "in_play"
            self.score += 1

            reward = rewards["ball_hit_player"] if mode=="trainleft" else 0
            return reward, "in_play"

        # Check collision with player2
        if self.two_players :
            player2_start_y = self.player2.y
            player2_end_y = self.player2.y + self.player2.length

            if (self.ball.x + self.ball.radius >= self.player2.x and
                player2_start_y <= self.ball.y <= player2_end_y):

                # reflect ball according to where it hit the player2
                hit_pos = (self.ball.y - player2_start_y) / self.player2.length # 0 to 1
                new_angle = (hit_pos - 0.5) * math.pi / 2  # -pi/4 to pi/4
                new_angle = math.pi - new_angle # to the left 
                speed = math.sqrt(self.ball.vx**2 + self.ball.vy**2)
                self.ball.vx = speed * math.cos(new_angle)
                self.ball.vy = speed * math.sin(new_angle)
                self.ball.x = self.player2.x - self.ball.radius  # Prevent sticking
                self.ball.state = "in_play"
                self.score += 1

                reward = rewards["ball_hit_player"] if mode=="trainright" else 0
                return reward, "in_play" 


        # Check if ball left the screen
        if self.ball.x < 0:
            self.ball.state = "missed"
            self.reset()
            return rewards["ball_missed"], "done"
        
        if self.two_players :
            # if 2players, right of the screen is a miss
            if self.ball.x > kScreenWidth:
                self.ball.state == "missed"
                self.reset()
                return rewards["ball_missed"], "done"
        
        # check for collision with front wall
        if not self.two_players :
            # if 1player, right of the screen is a wall
            if self.ball.x > kScreenWidth:
                # reflect ball according to where it hit the wall + small randomization to avoid "locking" strategies
                speed = math.sqrt(self.ball.vx**2 + self.ball.vy**2)
                self.ball.vx = -self.ball.vx + random.random() 
                self.ball.vy = self.ball.vy + random.random()
                corrector_speed = speed / math.sqrt(self.ball.vx**2 + self.ball.vy**2)
                self.ball.vx *= corrector_speed
                self.ball.vy *= corrector_speed

                self.ball.x = kScreenWidth - self.ball.radius  # Prevent sticking
                self.ball.state = "in_play"

        # check collision with side walls (up and down)

        if self.ball.y <= self.ball.radius:
            self.ball.vy = -self.ball.vy 
            self.ball.y = self.ball.y + self.ball.radius
            
        elif self.ball.y >= kScreenHeight - self.ball.radius:
            self.ball.vy = -self.ball.vy 
            self.ball.y = self.ball.y - self.ball.radius
        
        return rewards["step"], "in_play"
 
    def step(self, actionp1, actionp2=-1):

        self.player.move(actionp1)
        if self.two_players :
            self.player2.move(actionp2)
        self.ball.move()
        reward, status = self.handle_collisions_and_misses()

        # adding continuous reward regarding y delta to ball
        if mode=="trainleft":
            player_y_dist_to_ball = abs(self.ball.y - (self.player.y+(self.player.length//2)))
            y_dist_reward = -player_y_dist_to_ball*rewards["dist_reward_factor"]
            reward += y_dist_reward
        elif mode=="trainright":
            player2_y_dist_to_ball = abs(self.ball.y - (self.player2.y+(self.player2.length//2)))
            y_dist_reward = -player2_y_dist_to_ball*rewards["dist_reward_factor"]
            reward += y_dist_reward

        # rewarding continuous movement (unused)
        """if action == 2 :
            reward += rewards["static_reward"]"""
    
        return reward, status

    def render(self, screen):
        screen.fill((0, 0, 0))
        self.player.draw(screen)
        self.ball.draw(screen)
        if self.two_players:
            self.player2.draw(screen)
        

def get_state(game :Game, player=1):

    if player==1:
        state = [game.player.y, game.ball.x, game.ball.y, game.ball.vx, game.ball.vy]
    elif player==2:
        state = [game.player2.y, game.ball.x, game.ball.y, game.ball.vx, game.ball.vy]
    return state

def main():

    if mode=="play":

        left_agent = Agent(state_size=config["game"]["state_size"], action_size=config["game"]["action_size"], cfg=config["agent"])
        left_agent.model.load_state_dict(torch.load("ga_left_player.pth"))
        left_agent.model.eval() 

        right_agent = Agent(state_size=config["game"]["state_size"], action_size=config["game"]["action_size"], cfg=config["agent"])
        right_agent.model.load_state_dict(torch.load("ga_right_player.pth"))
        right_agent.model.eval() 

        pygame.init()
        font = pygame.font.SysFont("Arial", 24)

        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Squash left_agent")
        clock = pygame.time.Clock()

        game = Game(two_players=True)
        best_score = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        
            left_action = left_agent.act(get_state(game, player=1))
            right_action = right_agent.act(get_state(game, player=2))
            reward, status = game.step(actionp1=left_action, actionp2=right_action)
            best_score = max(best_score, game.score)


            game.render(screen)
            score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
            best_score_text = font.render(f"Best Score: {best_score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            screen.blit(best_score_text, (10, 50))




            pygame.display.flip()

            timeDelta = clock.tick(FPS) / 1000.0

    elif mode == "trainleft":

        pygame.init()
        font = pygame.font.SysFont("Arial", 24)

        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Squash Agent")
        clock = pygame.time.Clock()

        population = [Agent(state_size=config["game"]["state_size"], action_size=config["game"]["action_size"], cfg=config["agent"]) for _ in range(config["train"]["population_size"])]
        great_players = population[:config["train"]["nb_of_great_players"]] # collecting best players overall

        for generation in range(config["train"]["nb_generations"]):

            print(f"Generation {generation+1}/{config['train']['nb_generations']}")

            game = Game()

            agent_idx = 0
            for agent in tqdm(population):
                
                agent_idx += 1
                episode = 0 # episode ends when ball is missed
                agent.training_reward = 0

                while episode < config["train"]["nb_episodes_per_agent"]:

                    action = agent.act(get_state(game, player=1))
                    reward, status = game.step(action)
                    agent.training_reward += reward

                    if game.score > 30 : # avoid infinite plays
                        episode += 1
                        game.reset()

                    if status == "done":
                        episode += 1
                    
                    # show succesful agents playing 
                    if game.score > 20 and generation: 
                        game.render(screen)
                        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
                        gen_text = font.render(f"Generation {generation+1} : agent nb {agent_idx} ep. {episode}", True, (255,255,255))
                        screen.blit(score_text, (10, 10))
                        screen.blit(gen_text, (10, kScreenHeight-40))
                        pygame.display.flip()
                        timeDelta = clock.tick(FPS) / 1000.0
  
            
            sorted_population = sorted(population, key=lambda x: x.training_reward, reverse=True)
            threshold = math.floor(config["train"]["percentage_to_keep"]*config["train"]["population_size"])
            best_agents = sorted_population[:threshold] # top x% agents kept for reproduction
            
            print([a.training_reward for a in best_agents])

            # updating great_players
            best_agents += great_players # re-adding greatplayers to genetic pool
            great_players = sorted(best_agents, key=lambda x: x.training_reward, reverse=True) # updating great players
            great_players = great_players[:config["train"]["nb_of_great_players"]]


            # debug info
            mean_reward = sum(agent.training_reward for agent in population) / len(population)
            best_reward = best_agents[0].training_reward
            

            print(f"Generation {generation+1}/{config['train']['nb_generations']} - Best Reward: {best_reward} - Mean Reward: {mean_reward}")
            print(f"Great players have rewards : {[gp.training_reward for gp in great_players]}")

            new_population = []
            while len(new_population) < config["train"]["population_size"]:
                parent1, parent2 = random.sample(best_agents, 2)
                child = Agent(state_size=config["game"]["state_size"], action_size=config["game"]["action_size"], cfg=config["agent"])
                for child_param, param1, param2 in zip(child.model.parameters(), parent1.model.parameters(), parent2.model.parameters()):
                    mask = torch.rand_like(child_param) > 0.5
                    child_param.data.copy_(torch.where(mask, param1.data, param2.data))
                    mutation_mask = torch.rand_like(child_param) < config["train"]["mutation_rate"]
                    mutation_values = torch.randn_like(child_param) * config["train"]["mutation_scale"]
                    child_param.data.add_(mutation_mask.float() * mutation_values)
                new_population.append(child)
            
            population = new_population


        print("Training finished.")
        best_agent = max(best_agents, key=lambda x: x.training_reward)
        print(f"Best agent training reward: {best_agent.training_reward}")
        print("Saving model...")
        torch.save(best_agent.model.state_dict(), "ga_best_model.pth")
        print("Model saved.")

    elif mode=="trainright":

        ### Training right agent requires having a trained left agent
        left_agent = Agent(state_size=config["game"]["state_size"], action_size=config["game"]["action_size"], cfg=config["agent"])
        left_agent.model.load_state_dict(torch.load("ga_left_player.pth"))
        left_agent.model.eval() 

        pygame.init()
        font = pygame.font.SysFont("Arial", 24)

        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Squash Agent")
        clock = pygame.time.Clock()

        population = [Agent(state_size=config["game"]["state_size"], action_size=config["game"]["action_size"], cfg=config["agent"]) for _ in range(config["train"]["population_size"])]
        great_players = population[:config["train"]["nb_of_great_players"]] # collecting best players overall

        for generation in range(config["train"]["nb_generations"]):

            print(f"Generation {generation+1}/{config['train']['nb_generations']}")

            game = Game(two_players=True)

            agent_idx = 0
            for agent in tqdm(population):
                
                agent_idx += 1
                episode = 0 # episode ends when ball is missed
                agent.training_reward = 0

                while episode < config["train"]["nb_episodes_per_agent"]:

                    left_action = left_agent.act(get_state(game, player=1))
                    right_action = agent.act(get_state(game, player=2))
                    reward, status = game.step(actionp1=left_action, actionp2=right_action)
                    agent.training_reward += reward

                    if game.score > 30 : # avoid infinite plays
                        episode += 1
                        game.reset()

                    if status == "done":
                        episode += 1
                    
                    # show succesful agents playing 
                    if game.score >= 20 and generation >= 10: 
                        game.render(screen)
                        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
                        gen_text = font.render(f"Generation {generation+1} : agent nb {agent_idx} ep. {episode}", True, (255,255,255))
                        screen.blit(score_text, (10, 10))
                        screen.blit(gen_text, (10, kScreenHeight-40))
                        pygame.display.flip()
                        timeDelta = clock.tick(FPS) / 1000.0
  
            
            sorted_population = sorted(population, key=lambda x: x.training_reward, reverse=True)
            threshold = math.floor(config["train"]["percentage_to_keep"]*config["train"]["population_size"])
            best_agents = sorted_population[:threshold] # top x% agents kept for reproduction
            
            print([a.training_reward for a in best_agents])

            # updating great_players
            best_agents += great_players # re-adding greatplayers to genetic pool
            great_players = sorted(best_agents, key=lambda x: x.training_reward, reverse=True) # updating great players
            great_players = great_players[:config["train"]["nb_of_great_players"]]


            # debug info
            mean_reward = sum(agent.training_reward for agent in population) / len(population)
            best_reward = best_agents[0].training_reward
            

            print(f"Generation {generation+1}/{config['train']['nb_generations']} - Best Reward: {best_reward} - Mean Reward: {mean_reward}")
            print(f"Great players have rewards : {[gp.training_reward for gp in great_players]}")

            new_population = []
            while len(new_population) < config["train"]["population_size"]:
                parent1, parent2 = random.sample(best_agents, 2)
                child = Agent(state_size=config["game"]["state_size"], action_size=config["game"]["action_size"], cfg=config["agent"])
                for child_param, param1, param2 in zip(child.model.parameters(), parent1.model.parameters(), parent2.model.parameters()):
                    mask = torch.rand_like(child_param) > 0.5
                    child_param.data.copy_(torch.where(mask, param1.data, param2.data))
                    mutation_mask = torch.rand_like(child_param) < config["train"]["mutation_rate"]
                    mutation_values = torch.randn_like(child_param) * config["train"]["mutation_scale"]
                    child_param.data.add_(mutation_mask.float() * mutation_values)
                new_population.append(child)
            
            population = new_population


        print("Training finished.")
        best_agent = max(best_agents, key=lambda x: x.training_reward)
        print(f"Best agent training reward: {best_agent.training_reward}")
        print("Saving model...")
        torch.save(best_agent.model.state_dict(), "ga_right_player.pth")
        print("Model saved.")

if __name__ == "__main__":
    main()
        
    
