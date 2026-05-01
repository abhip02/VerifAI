import os
import pandas as pd
import random
from verifai.samplers import ScenicSampler
from verifai.falsifier import generic_falsifier
from verifai.scenic_server import ScenicServer
from verifai.monitor import specification_monitor
from dotmap import DotMap

global path

# Spec: fail if speed > 20
class SpeedSpec(specification_monitor):
    global path
    def __init__(self):
        def spec(sim_result):
            # print(dict(sim_result).keys())
            # Get recorded speeds from Scenic result
            ego_speed = sim_result.records.get("ego_speed", [])
            ego_vx = sim_result.records.get("ego_vx", [])
            ego_vy = sim_result.records.get("ego_vy", [])
            ego_heading = sim_result.records.get("ego_heading", [])
            ego_position = sim_result.records.get("ego_position", [])
            
            # values at the end of the sim
            final_ego_speed = ego_speed[-1][-1]
            
            final_ego_vx = ego_vx[-1][-1]
            final_ego_vy = ego_vy[-1][-1]
            
            final_ego_heading = ego_heading[-1][-1]
            final_ego_position = ego_position[-1][-1]
            final_ego_position_x = final_ego_position[0]
            final_ego_position_y = final_ego_position[1]
            
            
            rho = 0 if final_ego_speed < 6 else 1
            
            # Final values for CSV
            final_data = {
                "ego_heading": [final_ego_heading],
                "ego_x": [final_ego_position_x],
                "ego_y": [final_ego_position_y],
                "ego_speed": [final_ego_speed],
                "ego_vx": [final_ego_vx],
                "ego_vy": [final_ego_vy],
                "rho": [rho]
            }

            # Convert to DataFrame
            df = pd.DataFrame(final_data)

            # Append to CSV, write header only if file doesn't exist
            df.to_csv(path + "_post_conditions.csv", mode='a', header=not os.path.exists(path + "_post_conditions.csv"), index=False)
            
            # Falsification: Fail if sampled speed > 3
            return rho
        super().__init__(spec)


def falsify_main(iter, falsification_iters):
    global path
    
    # 1. Load scenario
    if iter < 1:
        scenario_path = os.path.abspath("subscenario1")
    else:
        subscenario2s = ["subscenario2L", "subscenario2S", "subscenario2R"]
        chosen_subscenario = random.choice(subscenario2s)
        
        scenario_path = os.path.abspath(chosen_subscenario)
        
    path = scenario_path
    sampler = ScenicSampler.fromScenario(scenario_path + ".scenic")

    # 3. Falsifier params
    params = DotMap(
        n_iters=falsification_iters,
        save_error_table=True,
        # fal_thres=1
    )

    # 4. Create falsifier
    falsifier = generic_falsifier(
        sampler=sampler,
        monitor=SpeedSpec(),
        falsifier_params=params,
        server_class=ScenicServer
    )

    os.environ["PANDA3D_SHOWBASE_WINDOWTYPE"] = "offscreen"
    # 5. Run falsifier (launches MetaDrive simulation each time)
    falsifier.run_falsifier()

    # DEBUG: Print column titles of error table
    df = falsifier.error_table
    # print(df.table.columns)


if __name__ == "__main__":
    # PARAMS
    iters = 5
    falsification_iters = 2
    
    # delete existing file to restart it
    filename1 = "subscenario1_post_conditions.csv"
    filename2 = "subscenario2S_post_conditions.csv"
    filename3 = "subscenario2L_post_conditions.csv"
    filename4 = "subscenario2R_post_conditions.csv"
    
    filenames = [filename1, filename2, filename3, filename4]
    
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"{filename} deleted.")
        
    for i in range(iters):
        falsify_main(iter=i, falsification_iters=falsification_iters)
