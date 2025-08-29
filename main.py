import pygame
import random 
import math
from agent import Agent
import yaml
import torch
import os
from tqdm import tqdm
import sys


#region - loading parameters from yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
rewards_dict = config["rewards"]
train_dict = config["train"]
game_dict = config["game"]
agent_dict = config["agent"]

FPS = game_dict["FPS"]
kScreenWidth = game_dict["screen_width"]
kScreenHeight = game_dict["screen_height"]
# endregion

#region - handling argv
if len(sys.argv) != 3:
    print("usage : main.py MODE TYPE - as in main.py play squash", file=sys.stderr)
    sys.exit(1)
mode = sys.argv[1].lower()
type = sys.argv[2].lower()
print(mode, type)
if mode not in {"play", "train"} or type not in {"squash", "target", "movingtarget"}:
    print("Modes are play and train.\n Types are squash, target and movingtarget", file=sys.stderr)
    sys.exit(1)
# endregion



#### CLASSES DEFINITIONS
class Ball:
    def __init__(self):
        self.x = kScreenWidth - 150
        self.y = random.randint(kScreenHeight // 10, kScreenHeight*9 // 10)
        self.vx = -10 # effectively horizontal speed / game speed
        self.vy = random.randrange(-6,6)
        self.speed = math.sqrt(self.vx**2+self.vy**2)
        self.radius = 10
        self.state = "in_play"  # can be "in_play", "missed"

    def move(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (self.x, self.y), self.radius)


class Player:
    def __init__(self):
        self.x = 100
        self.length = 100
        self.top_y = (kScreenHeight - self.length)//2 # centered
        self.speed =10

    def move(self, action):
        if action == 0:
            self.top_y += self.speed
        elif action == 1:
            self.top_y -= self.speed
        elif action == 2:
            pass
        
        # avoid leaving court
        self.top_y = min(kScreenHeight-self.length, max(0, self.top_y))
     
    def draw(self, screen):
        pygame.draw.line(screen, (255,255,255), (self.x, self.top_y), (self.x, self.top_y+self.length), 10)

    def handle_ball_collision(self, ball:Ball):

        hit_bool = (self.top_y <= ball.y <= self.top_y+self.length) and (self.x - ball.radius <= ball.x <= self.x)
        if hit_bool :
            # collision happened, checking impact position
            impact_pos = (ball.y - self.top_y) / self.length
            impact_pos = 2*impact_pos - 1 # -1 on top, 0 in center, 1 on bottom
            # redirecting ball accordingly
            new_angle = impact_pos * math.pi / 4
            speed = math.sqrt(ball.vx**2 + ball.vy**2)
            ball.vx = speed * math.cos(new_angle)
            ball.vy = speed * math.sin(new_angle)
            # avoid sticking
            ball.x = self.x + ball.radius 

        return hit_bool


class Target:
    def __init__(self, type= None, length=100):
        
        self.x = kScreenWidth-100
        self.length = length
        self.top_y = 0
        self.center = 0
        self.set() # initial positioning

        self.type = type
        if self.type is None :
            print("Choose type from : static, moving")
        elif self.type == "moving":
            self.direction = "up" # initial direction

    def set(self):
        self.top_y = random.randint(0, kScreenHeight-self.length)
        self.center = self.top_y + self.length//2

    def move(self):
        """ for the movement on hit in 'static' mode, see handle_ball_collision"""

        if self.type == "moving":
            if self.direction == "up":
                if self.top_y > 4:
                    self.top_y -= 4 
                else:
                    self.direction="down"
            if self.direction == "down":
                if self.top_y+self.length < kScreenHeight - 4:
                    self.top_y += 4
                else:
                    self.direction = "up"

    def draw(self, screen):
        pygame.draw.line(screen, (0,255,0), (self.x, self.top_y), (self.x, self.top_y+self.length), 10)
        
    def handle_ball_collision(self, ball:Ball):

        hit_bool = (self.top_y <= ball.y <= self.top_y+self.length) and (self.x + ball.radius >= ball.x >= self.x)
        if hit_bool :
            # collision happened, checking impact position
            impact_pos = (ball.y - self.top_y) / self.length
            impact_pos = 2*impact_pos - 1 # -1 on top, 0 in center, 1 on bottom
            # redirecting ball accordingly
            new_angle = -impact_pos * math.pi / 4 + math.pi 
            ball.vx = ball.speed * math.cos(new_angle)
            ball.vy = ball.speed * math.sin(new_angle)
            # avoid sticking
            ball.x = self.x - ball.radius 

            if self.type == "static":
                # in static mode, targets moves after every hit
                # in curriculum, added an intermediate objective by disabling this,
                # meaning : the target only changes with every new episode
                # self.set()

        return hit_bool


class Game:
    def __init__(self, type=None):

        self.player = Player()
        self.ball = Ball()
        self.score = 0
        
        self.type = type
        if type is None:
            print("Choose type from : squash, target, moving_target")
        elif self.type == "squash":
            self.target = None
        elif self.type == "target":
            self.target = Target(type="static")
        elif self.type == "moving_target":
            self.target = Target(type="moving")
        
    def reset(self):
        self.player = Player()
        self.ball = Ball()
        self.score = 0
        if self.target is not None:
            self.target.set()

    def handle_borders(self):
        """returns in_play status"""

        # ball is missed if leaving from left side
        if  self.ball.x + self.ball.radius <= 0:
            self.ball.state = "missed"
            return False
        
        # in modes other than squash, ball is missed if leaving from right side
        if self.ball.x - self.ball.radius >= kScreenWidth and self.type != "squash":
            self.ball.state = "missed"
            return False
        
        # in squash, ball bounces off right wall
        if self.ball.x + self.ball.radius >= kScreenWidth and self.type == "squash":
            self.ball.vx = -self.ball.vx
            self.ball.vy += random.randint(-1,1) if self.ball.vy == 0 else 0 # avoid centering strats

        # ball always bounces off top and bottom
        if self.ball.y - self.ball.radius <= 0:
            self.ball.vy = -self.ball.vy
            self.ball.y = self.ball.radius # prevent sticking

        if self.ball.y + self.ball.radius >= kScreenHeight:
            self.ball.vy = -self.ball.vy
            self.ball.y = kScreenHeight - self.ball.radius # prevent sticking

        return True

    def step(self, action):

        terminated = False

        # moving 
        self.player.move(action)
        self.ball.move()
        if self.target is not None:
            self.target.move()

        # checking states
        player_hit = self.player.handle_ball_collision(self.ball)
        if self.target is not None:
            target_hit = self.target.handle_ball_collision(self.ball)
        in_play_status = self.handle_borders()

        # adding scores
        self.score += int(player_hit)

        # computing step reward
        step_reward = 0
        step_reward += rewards_dict["ball_hit_player"] * player_hit
        if self.target is not None:
            step_reward += rewards_dict["ball_hit_target"] * target_hit
        step_reward += rewards_dict["ball_missed"] * (not in_play_status)
        player_y_dist_norm = 1 - (abs((self.player.top_y + self.player.length//2 ) - self.ball.y) / kScreenHeight)
        step_reward += rewards_dict["y_dist_reward_factor"] * player_y_dist_norm
        step_reward += rewards_dict["stability_reward"] if action == 2 else 0
        step_reward += rewards_dict["centered"] * 1 - (abs((self.player.top_y + self.player.length//2 ) - kScreenHeight//2) / kScreenHeight)
        if not in_play_status:
            terminated = True
            self.reset()

        return step_reward, terminated

    def get_state(self):
        # states are normalized
        state = [
                self.player.top_y / (kScreenHeight - self.player.length),
                self.ball.x / kScreenWidth, 
                self.ball.y / kScreenHeight,
                self.ball.vx / self.ball.speed,
                self.ball.vy / self.ball.speed
                ]
                
        # including intel for target, or normalized placeholders for future curriculum steps
        if self.target is not None:
            state.append(self.target.center / (kScreenHeight - self.target.length))
            if self.target.type == "moving":
                state.append(1 if self.target.direction=="up" else 0)
            else :
                state.append(0.5) # placeholder
        else:
            state.append(0.5) # placeholder
        
            

        return state

    def render(self, screen):
        self.player.draw(screen)
        self.ball.draw(screen)
        if self.target is not None:
            self.target.draw(screen)



#### MAIN
def main():


    if mode=="play" : 
        # initializing pygame and game instance
        pygame.init()
        font = pygame.font.SysFont("Arial", 24)
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption(f"Pong Agent - {type}")
        clock = pygame.time.Clock()
        game = Game(type=type)
        best_score = 0
        running = True

        # getting latest trainee as agent if available
        state_size = len(game.get_state())
        agent = Agent(state_size=state_size, action_size=game_dict["action_size"], cfg=agent_dict)
        path_latest_agent = f"agents/ga_agent_{type}_{len(os.listdir("agents"))}.pth"
        print("Latest agent is in : ", path_latest_agent)
        if os.path.exists(path_latest_agent) :
            print("loading agent from ", path_latest_agent)
            agent.model.load_state_dict(torch.load(path_latest_agent))
        agent.model.eval() 

        # game loop
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return

        
            # step
            state = game.get_state()
            action = agent.act(state)
            game.step(action)
            best_score = max(best_score, game.score)

            # rendering 
            screen.fill((0,0,0))
            game.render(screen)
            score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
            best_score_text = font.render(f"Best Score: {best_score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            screen.blit(best_score_text, (10, 50))
            pygame.display.flip()

            timeDelta = clock.tick(FPS) / 1000.0

    if mode == "train" :

        # initializing pygame and game instance
        """
        pygame.init()
        font = pygame.font.SysFont("Arial", 24)
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption(f"Pong Agent - {type}")
        clock = pygame.time.Clock()
        """
        game = Game(type=type)
        running = True

        # getting agent fleet for genetic algorithm
        state_size = len(game.get_state())
        population = [Agent(state_size=state_size, action_size=game_dict["action_size"], cfg=agent_dict) for _ in range(train_dict[type]["population_size"])]
        agent_save_id = len(os.listdir("agents"))+1
        if type=="target":
            # train from best squasher
            path_latest_squash_agent = f"agents/ga_agent_squash_{agent_save_id-1}.pth"
            if os.path.exists(path_latest_squash_agent) :
                print("training on target from ", path_latest_squash_agent)
                for agent in population:
                    agent.model.load_state_dict(torch.load(path_latest_squash_agent))
                    agent.model.eval()
        
        if type=="movingtarget":
            path_latest_target_agent = f"agents/ga_agent_target_{agent_save_id-1}.pth"
            if os.path.exists(path_latest_agent) :
                print("training on moving target from ", path_latest_target_agent)
                for agent in population:
                    agent.model.load_state_dict(torch.load(path_latest_target_agent))
                    agent.model.eval()
        great_players = population[:train_dict[type]["nb_of_great_players"]] # collecting best players overall

        current_mutation_rate = train_dict[type]["mutation_rate"]

        # training agents
        for generation in range(train_dict[type]["nb_generations"]):

            print(f"Generation {generation+1}/{train_dict[type]['nb_generations']}")

            agent_idx = 0
            for agent in tqdm(population):
                
                episode = 0 # episode ends when ball is missed
                agent.training_reward = 0
                episode_reward = 0

                while episode < train_dict[type]["nb_episodes_per_agent"]:
                    
                    # step
                    state = game.get_state()
                    action = agent.act(state)
                    reward, terminated = game.step(action)
                    agent.training_reward += reward
                    episode_reward += reward

                    if terminated:
                        episode += 1
                        episode_reward = 0

                    # avoid infinite plays
                    if game.score > 30 : 
                        episode += 1
                        episode_reward = 0
                        game.reset()
                    
                    """
                    # render first agent playing on every generation
                    # you are better off setting up a main.py play in another terminal, using last best agent
                    if agent_idx==0 and episode==1 : 
                        screen.fill((0,0,0))
                        game.render(screen)
                        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
                        gen_text = font.render(f"Generation {generation+1} : agent nb {agent_idx+1} ep. {episode+1}", True, (255,255,255))
                        action_text = font.render(f"action : {action}", True, (255,255,255))
                        episode_reward_text = font.render(f"episode reward : {episode_reward}", True, (255,255,255))


                        screen.blit(score_text, (10, 10))
                        screen.blit(gen_text, (10, kScreenHeight-40))
                        screen.blit(action_text, (kScreenWidth - 200, kScreenHeight-40))
                        screen.blit(episode_reward_text, (kScreenWidth - 230, kScreenHeight-80))
                        pygame.display.flip()

                        timeDelta = clock.tick(FPS) / 1000.0
                    """

                agent_idx += 1
  
            
            sorted_population = sorted(population, key=lambda x: x.training_reward, reverse=True)
            threshold = math.floor(train_dict[type]["percentage_to_keep"]*train_dict[type]["population_size"])
            best_agents = sorted_population[:threshold] # top x% agents kept for reproduction
            
            # updating great_players
            best_agents += great_players # re-adding greatplayers to genetic pool
            great_players = sorted(best_agents, key=lambda x: x.training_reward, reverse=True) # updating great players
            great_players = great_players[:train_dict[type]["nb_of_great_players"]]


            # debug info
            mean_reward = sum(agent.training_reward for agent in population) / len(population)
            best_reward = best_agents[0].training_reward
            

            print(f"Generation {generation+1}/{train_dict[type]['nb_generations']} - Best Reward: {best_reward:.1f} - Mean Reward: {mean_reward:.1f}") 
            print(f"Great players have rewards : {[f'{gp.training_reward:.1f}' for gp in great_players]}")

            new_population = []
            while len(new_population) < train_dict[type]["population_size"]:
                parent1, parent2 = random.sample(best_agents, 2)
                child = Agent(state_size=state_size, action_size=game_dict["action_size"], cfg=agent_dict)
                for child_param, param1, param2 in zip(child.model.parameters(), parent1.model.parameters(), parent2.model.parameters()):
                    mask = torch.rand_like(child_param) > 0.5
                    child_param.data.copy_(torch.where(mask, param1.data, param2.data))
                    mutation_mask = torch.rand_like(child_param) < current_mutation_rate
                    mutation_values = torch.randn_like(child_param) * train_dict[type]["mutation_scale"]
                    child_param.data.add_(mutation_mask.float() * mutation_values)
                new_population.append(child)
            
            current_mutation_rate *= train_dict[type]["mutation_rate_decay"]
            print(f"{current_mutation_rate=}")
            print("\n\n")
            population = new_population

            # saving current best agent
            best_agent = max(best_agents, key=lambda x: x.training_reward)
            torch.save(best_agent.model.state_dict(), f"agents/ga_agent_{type}_{agent_save_id}.pth")


if __name__ == "__main__":
    main()
        
    
