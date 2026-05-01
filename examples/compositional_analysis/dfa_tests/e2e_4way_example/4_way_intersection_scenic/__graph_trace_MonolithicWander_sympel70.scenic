from composed_wander import *

scenario GraphTraceEntry_MonolithicWander():
    setup:
        ego = new Car
    compose:
        do MonolithicWander()
