import os.path as ops
import pygame


BG_COLOR = (135, 206, 235)
white_color = (255, 255, 255)
SCREEN_SIZE = (450, 450)
map_size = (9, 9)
BLACK = (0, 0, 0)
ship_color = (14, 8, 168)
fort_color = (36, 133, 4)
font_path = ops.join(ops.dirname(ops.dirname(__file__)), 'combat/assets/font/Qaz-ZV74m.ttf')
n_ship = 3

class Viewer:
    def __init__(self):
        self.width = SCREEN_SIZE[0]
        self.height = SCREEN_SIZE[1]

        self.map_width = map_size[0]
        self.map_height = map_size[1]

        self.display = None
        self.board = pygame.surface.Surface((self.width + 50, self.height + 210))
        self.surface = pygame.surface.Surface((self.width, self.height))

    def make_display(self):
        self.display = pygame.display.set_mode((self.width + 50, self.height + 210))

    def draw_surface(self, agents, actions=None):
        self.board.fill(white_color)
        if not pygame.font.get_init():
            pygame.font.init()
        f = pygame.font.Font(font_path, 50)
        for i in range(9):
            f_id = f.render(str(i + 1), True, (163, 196, 120))
            rect = f_id.get_rect()
            rect.x = i * 50 + 10
            rect.y = 160
            self.board.blit(f_id, rect)
        for i in range(9):
            f_id = f.render(str(i + 1), True, (163, 196, 120))
            rect = f_id.get_rect()
            rect.y = i * 50 + 210
            rect.x = 450
            self.board.blit(f_id, rect)

        self.surface.fill(BG_COLOR)
        interval = self.width // self.map_width
        for i in range(1, 9):
            pygame.draw.line(self.surface, BLACK, (i * interval, 0), (i * interval, self.height))

        interval = self.height // self.map_height
        for i in range(1, 9):
            pygame.draw.line(self.surface, BLACK, (0, i * interval), (self.height, i * interval))

        for agent in agents:
            if agent.is_dead:
                continue

            rect_center = (agent.y * interval + interval//2, agent.x * interval + interval//2)
            if agent.type == 'ship':
                pygame.draw.ellipse(self.surface, ship_color, (agent.y * interval, agent.x * interval+5, interval, interval-10))
            elif agent.type == 'fort':
                pygame.draw.circle(self.surface, fort_color, rect_center, interval//2)

            f_id = f.render(str(agent.idx), True, white_color)
            rect = f_id.get_rect()
            rect.center = rect_center
            self.surface.blit(f_id, rect)

        if actions is not None:
            y = 0
            f = pygame.font.Font(font_path, 26)
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
                        pygame.draw.rect(self.surface, (150, 5, 5), (fired_y*50, fired_x*50, 50, 50))
                else:
                    fired_x = action % 9
                    fired_y = action // 9
                    memo = f.render(
                        f"Agent_{str(agent.idx)}'s HP is {agent.hp}, firing position: {fired_x + 1, fired_y + 1}", True,
                        BLACK)
                    pygame.draw.rect(self.surface, (150, 5, 5), (fired_y * 50, fired_x * 50, 50, 50))

                self.board.blit(memo, (0, y))
                y += 30

        self.board.blit(self.surface, [0, 210])

    def update_display(self):
        if self.display is None:
            raise RuntimeError(
                "Tried to update the display, but a display hasn't been "
                "created yet! To create a display for the renderer, you must "
                "call the `make_display()` method."
            )

        self.display.blit(self.board, [0, 0])
        pygame.display.update()
