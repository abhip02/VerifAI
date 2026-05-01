from composed_wander import *

scenario GraphTraceEntry_TurnRight():
    setup:
        ego = new Car with behavior TurnRight(), with speed Range(8, 14)
    compose:
        while True:
            wait
