from composed_wander import *

scenario GraphTraceEntry_Wander():
    setup:
        ego = new Car with behavior Wander(), with speed Range(8, 14)
    compose:
        while True:
            wait
