from composed_wander import *

scenario GraphTraceEntry_TurnLeft():
    setup:
        ego = new Car with behavior TurnLeft(), with speed Range(8, 14)
    compose:
        while True:
            wait
