class Entity:
    def __init__(self):
        self.type = None
        self.idx = None
        self.hp = 5
        self.movable = False
        self.color = None

    @property
    def is_dead(self):
        return self.hp <= 0


class Fort(Entity):
    def __init__(self, idx, x, y):
        super(Fort, self).__init__()
        self.idx = idx
        self.x = x
        self.y = y
        self.dmg = 1
        self.type = 'fort'
        self.action = None
        self.can_fire = True


class Ship(Entity):
    def __init__(self, idx, x, y):
        super(Ship, self).__init__()
        self.idx = idx
        self.movable = True
        self.x = x
        self.y = y
        self.dmg = 1
        self.type = 'ship'
