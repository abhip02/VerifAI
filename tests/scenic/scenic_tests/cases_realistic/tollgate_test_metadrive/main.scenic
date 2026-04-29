"""Tollgate test — MetaDrive physics variant.

Same non-Markovian mandatory-wait DFA (K=3) as tollgate_test/, but purpose-
built for MetaDrive so the scene sampler gets a MetaDrive-native Vehicle as
the ego instead of a base Object (which broke when the newtonian driving
model was combined with a --model override).
"""

param map = localPath('../../../Town01.xodr')
param timestep = 0.1
param use2DMap = True
param render = 0
param render3D = 0
param real_time = 0

model scenic.simulators.metadrive.model

from tollgate_phases import *

param stop_threshold = STOP_THRESHOLD_MS
param required_wait_steps = REQUIRED_WAIT_STEPS

terminate after 60 steps

scenario Main():
    precondition: True
    invariant: True
    setup:
        ego = new Car with behavior TollgateArrival()
    compose:
        do choose {
            TollgateCruise(): 1,
            TollgateArrival(): 3,
        }
