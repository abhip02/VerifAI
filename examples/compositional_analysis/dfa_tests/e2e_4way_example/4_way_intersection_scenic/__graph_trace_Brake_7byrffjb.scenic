from composed_wander import *

scenario GraphTraceEntry_Brake():
    setup:
        ego = new Car with behavior Brake(), with speed Range(8, 14)
    compose:
        while True:
            wait
