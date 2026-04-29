"""Tollgate test (MetaDrive variant) — same DFA, MetaDrive physics.

Same compliance patterns as tollgate_test/, but specialized for MetaDrive:
the model imports below bind the Car class from scenic.simulators.metadrive
so the scene sampler accepts our ego object without the `'Object' has no
attribute isCar` mismatch that bit us when we combined newtonian.driving_model
with a --model override.
"""

from scenic.domains.driving.actions import (
    SetThrottleAction,
    SetBrakeAction,
    SetReverseAction,
    SetHandBrakeAction,
)

STOP_THRESHOLD_MS = 1.0
REQUIRED_WAIT_STEPS = 3

CRUISE_THROTTLE = 0.6
HOLD_BRAKE = 1.0


behavior TollgateCruise():
    while True:
        take (
            SetThrottleAction(CRUISE_THROTTLE),
            SetBrakeAction(0),
            SetReverseAction(False),
            SetHandBrakeAction(False),
        )

behavior TollgateCompliantStop():
    for i in range(8):
        take (SetThrottleAction(CRUISE_THROTTLE), SetBrakeAction(0),
              SetReverseAction(False), SetHandBrakeAction(False))
    for i in range(8):
        take (SetThrottleAction(0), SetBrakeAction(HOLD_BRAKE))
    while True:
        take (SetThrottleAction(CRUISE_THROTTLE), SetBrakeAction(0))

behavior TollgateEarlyResume():
    for i in range(8):
        take (SetThrottleAction(CRUISE_THROTTLE), SetBrakeAction(0),
              SetReverseAction(False), SetHandBrakeAction(False))
    for i in range(3):
        take (SetThrottleAction(0), SetBrakeAction(HOLD_BRAKE))
    while True:
        take (SetThrottleAction(CRUISE_THROTTLE), SetBrakeAction(0))

behavior TollgateAlmostWaitedEnough():
    for i in range(8):
        take (SetThrottleAction(CRUISE_THROTTLE), SetBrakeAction(0),
              SetReverseAction(False), SetHandBrakeAction(False))
    for i in range(5):
        take (SetThrottleAction(0), SetBrakeAction(HOLD_BRAKE))
    while True:
        take (SetThrottleAction(CRUISE_THROTTLE), SetBrakeAction(0))

behavior TollgateCreep():
    while True:
        take (SetThrottleAction(0.15), SetBrakeAction(0),
              SetReverseAction(False), SetHandBrakeAction(False))


behavior TollgateArrival():
    do choose {
        TollgateCompliantStop(): 3,
        TollgateEarlyResume(): 1,
        TollgateAlmostWaitedEnough(): 1,
        TollgateCreep(): 1,
    }
