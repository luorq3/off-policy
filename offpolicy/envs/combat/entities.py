
class AgentState:
    def __init__(self):
        super(AgentState, self).__init__()
        self.x = None
        self.y = None
        self.hp = None


class Action:
    def __init__(self):
        self.u = None


class Entity:
    def __init__(self):
        self.name = ''
        self.idx = None
        self.hp = 20
        self.size = 0.050
        self.movable = False
        self.collide = False
        self.color = None
        self.state = AgentState()

    @property
    def is_dead(self):
        return self.hp <= 0


class Fort(Entity):
    def __init__(self, x, y):
        super(Fort, self).__init__()
        self.x = x
        self.y = y
        self.dmg = 1
        self.type = 'fort'


class Ship(Entity):
    def __init__(self, x, y):
        super(Ship, self).__init__()
        self.movable = True
        self.blind = False
        self.action = Action()
        self.action_callback = None
        self.x = x
        self.y = y
        self.dmg = 2
        self.type = 'ship'
