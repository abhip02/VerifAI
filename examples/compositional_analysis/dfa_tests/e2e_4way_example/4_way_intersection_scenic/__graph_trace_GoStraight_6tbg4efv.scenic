from composed_wander import *

scenario GraphTraceEntry_GoStraight():
    setup:
        ego = new Car with behavior GoStraight(), with speed Range(8, 14)
    compose:
        while True:
            wait
