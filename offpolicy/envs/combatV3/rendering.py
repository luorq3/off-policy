import os.path as ops
import pygame


BG_COLOR = (135, 206, 235)
WHITE = (255, 255, 255)
# SCREEN_SIZE = (450, 450)
GRID_WIDTH = 50
BLACK = (0, 0, 0)
SHIP_COLOR = (14, 8, 168)
FORT_COLOR = (36, 133, 4)
FONT_PATH = ops.join(ops.dirname(__file__), 'assets/font/Qaz-ZV74m.ttf')

class Viewer:
    def __init__(self, map_size):
        self.map_width = map_size[1]
        self.map_height = map_size[0]

        self.width = self.map_width * GRID_WIDTH
        self.height = self.map_height * GRID_WIDTH

        self.display = None
        self.board = pygame.surface.Surface((self.width * 2 + 50, self.height + 210))
        self.surface_ship = pygame.surface.Surface((self.width, self.height))
        self.surface_fort = pygame.surface.Surface((self.width, self.height))

    def make_display(self):
        self.display = pygame.display.set_mode((self.width + 50, self.height + 210))

    def draw_surface(self, agents, actions=None):
        self.board.fill(WHITE)
        if not pygame.font.get_init():
            pygame.font.init()
        f = pygame.font.Font(FONT_PATH, 50)
        for i in range(self.map_width):
            f_id = f.render(str(i + 1), True, (163, 196, 120))
            rect = f_id.get_rect()
            rect.x = i * 50 + 10
            rect.y = 160
            self.board.blit(f_id, rect)
        for i in range(self.map_height):
            f_id = f.render(str(i + 1), True, (163, 196, 120))
            rect = f_id.get_rect()
            rect.y = i * 50 + 210
            rect.x = self.width * 2
            self.board.blit(f_id, rect)

        self.surface_ship.fill(BG_COLOR)
        self.surface_fort.fill(BG_COLOR)

        interval = 50
        for i in range(1, self.map_width):
            pygame.draw.line(self.surface_ship, BLACK, (i * interval, 0), (i * interval, self.height))
            pygame.draw.line(self.surface_fort, BLACK, (i * interval, 0), (i * interval, self.height))

        for i in range(1, self.map_height):
            pygame.draw.line(self.surface_ship, BLACK, (0, i * interval), (self.height, i * interval))
            pygame.draw.line(self.surface_ship, BLACK, (0, i * interval), (self.height, i * interval))

        for agent in agents:
            if agent.is_dead:
                continue

            rect_center = (agent.y * interval + interval//2, agent.x * interval + interval//2)
            f_id = f.render(str(agent.idx), True, WHITE)
            rect = f_id.get_rect()
            rect.center = rect_center
            if agent.type == 'ship':
                pygame.draw.ellipse(self.surface_ship, SHIP_COLOR, (agent.y * interval, agent.x * interval+5, interval, interval-10))
                self.surface_ship.blit(f_id, rect)
            elif agent.type == 'fort':
                pygame.draw.circle(self.surface_fort, FORT_COLOR, rect_center, interval//2)
                self.surface_fort.blit(f_id, rect)

        if actions is not None:
            y = 0
            f = pygame.font.Font(FONT_PATH, 26)
            for agent, action in zip(agents, actions):
                if agent.is_dead:
                    continue
                if agent.type == 'ship':
                    if 0 < action < 5:
                        direction = ['up', 'down', 'left', 'right']
                        memo = f.render(f"Agent_{str(agent.idx)}'s HP is {agent.hp}, moving towards {direction[action-1]}", True, BLACK)
                    else:
                        real_action = action - 5
                        fired_agent = agents[real_action + 3]
                        fired_x = fired_agent.x
                        fired_y = fired_agent.y
                        memo = f.render(f"Agent_{str(agent.idx)}'s HP is {agent.hp}, firing position: {fired_x + 1, fired_y + 1}", True, BLACK)
                        pygame.draw.rect(self.surface_ship, (150, 5, 5), (fired_y*50, fired_x*50, 50, 50))
                else:
                    fired_x = action % 9
                    fired_y = action // 9
                    memo = f.render(
                        f"Agent_{str(agent.idx)}'s HP is {agent.hp}, firing position: {fired_x + 1, fired_y + 1}", True,
                        BLACK)
                    pygame.draw.rect(self.surface_fort, (150, 5, 5), (fired_y * 50, fired_x * 50, 50, 50))

                self.board.blit(memo, (0, y))
                y += 30

        self.board.blit(self.surface_ship, [0, 210])
        self.board.blit(self.surface_fort, [self.width, 210])

    def update_display(self):
        if self.display is None:
            raise RuntimeError(
                "Tried to update the display, but a display hasn't been "
                "created yet! To create a display for the renderer, you must "
                "call the `make_display()` method."
            )

        self.display.blit(self.board, [0, 0])
        pygame.display.update()
