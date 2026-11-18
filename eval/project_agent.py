"""Leaderboard agent for closed-loop evaluation of any Policy (see policy/policy_interface.py).

Put this next to your HW1 test_agent.py (leaderboard/team_code/) and run the leaderboard with
    --agent=leaderboard/team_code/project_agent.py
    --agent-config=dummy            (or the path to a checkpoint)

Environment variables:
    PROJECT_ROOT     path to the ec518-project repo (so `policy` can be imported)
    PROJECT_WEATHER  optional, e.g. HardRainNoon -> overrides the route weather
    PROJECT_LOG      optional, json file that receives per-route timing stats
"""
import json
import os
import sys
import time

import numpy as np

import carla
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

sys.path.append(os.environ.get("PROJECT_ROOT", "."))
from policy.policy_interface import load_policy  # noqa: E402

# Keep identical to collect/collect_data.py
W, H, FOV = 320, 240, 90
CAM_X, CAM_Z = 1.5, 2.4

# Post-processing applied identically to every variant (set PROJECT_POSTPROC=0 to disable):
#  - brake is binarized at 0.5 and throttle is zeroed while braking (same as the HW2 il_agent)
#  - anti-stall "creep": after STUCK_STEPS ticks below 0.1 m/s, force CREEP_THROTTLE for
#    CREEP_STEPS ticks. Behavior cloning learns "speed 0 -> brake" because the expert mostly
#    stands still at red lights (inertia problem); creep events are counted in PROJECT_LOG.
POSTPROC = os.environ.get("PROJECT_POSTPROC", "1") != "0"
STUCK_STEPS = int(os.environ.get("PROJECT_STUCK_STEPS", "100"))  # 5 s at 20 Hz
CREEP_STEPS, CREEP_THROTTLE = 20, 0.5
DEBUG = os.environ.get("PROJECT_DEBUG", "0") == "1"


def get_entry_point():
    return "ProjectAgent"


class ProjectAgent(autonomous_agent.AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.policy = load_policy(path_to_conf_file, device="cuda")
        self.policy.reset()
        self._route = None
        self._weather_set = False
        self._act_ms, self._step_ms = [], []
        self._t_last = None
        self.step = -1  # the course leaderboard's statistics_manager reads agent.step
        self._stopped, self._creep, self._n_creep = 0, 0, 0
        self._raw_brake_stopped = []

    def sensors(self):
        cam = {"x": CAM_X, "y": 0.0, "z": CAM_Z, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
               "width": W, "height": H, "fov": FOV}
        s = [dict(cam, type="sensor.camera.rgb", id="rgb"),
             {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"}]
        if getattr(self.policy, "needs_gt", False):
            # GT-Explicit only. If the leaderboard says "Illegal sensor used", add these two
            # types to the allowed sensor list in leaderboard/autoagents/agent_wrapper.py.
            s.append(dict(cam, type="sensor.camera.semantic_segmentation", id="seg"))
            s.append(dict(cam, type="sensor.camera.depth", id="depth"))
        return s

    # -- navigation command from the route the leaderboard gives us --------------
    def _command(self):
        if self._route is None:
            self._route = list(self._global_plan_world_coord)
        if not self._route:
            return 4
        loc = CarlaDataProvider.get_hero_actor().get_location()
        window = self._route[:10]
        dists = [tf.location.distance(loc) for tf, _ in window]
        i = int(np.argmin(dists))
        self._route = self._route[i:]
        if dists[i] < 3.0 and len(self._route) > 1:
            self._route.pop(0)  # reached it, look at the next one
        opt = self._route[0][1]
        return int(opt.value) if opt.value >= 1 else 4

    def run_step(self, input_data, timestamp):
        t0 = time.perf_counter()
        self.step += 1
        if not self._weather_set:
            w = os.environ.get("PROJECT_WEATHER")
            if w:
                CarlaDataProvider.get_world().set_weather(getattr(carla.WeatherParameters, w))
            self._weather_set = True

        obs = {
            "rgb": np.ascontiguousarray(input_data["rgb"][1][:, :, 2::-1]),  # BGRA -> RGB
            "speed": float(input_data["speed"][1]["speed"]),
            "command": self._command(),
        }
        if "seg" in input_data:
            obs["seg"] = np.ascontiguousarray(input_data["seg"][1][:, :, 2])
            d = input_data["depth"][1].astype(np.float32)
            obs["depth"] = (d[:, :, 2] + d[:, :, 1] * 256.0 + d[:, :, 0] * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0

        ta = time.perf_counter()
        steer, throttle, brake = self.policy.act(obs)
        self._act_ms.append((time.perf_counter() - ta) * 1000.0)

        speed = obs["speed"]
        if speed < 0.1:
            self._raw_brake_stopped.append(float(brake))
        if POSTPROC:
            brake = 1.0 if brake > 0.5 else 0.0
            if brake > 0:
                throttle = 0.0
            self._stopped = self._stopped + 1 if speed < 0.1 else 0
            if self._creep > 0:
                throttle, brake = CREEP_THROTTLE, 0.0
                self._creep -= 1
            elif self._stopped >= STUCK_STEPS:
                self._creep, self._stopped = CREEP_STEPS, 0
                self._n_creep += 1
        if DEBUG and self.step % 20 == 0:
            print(f"step {self.step} speed {speed:.2f} cmd {obs['command']} "
                  f"steer {steer:+.3f} throttle {throttle:.2f} brake {brake:.2f}", flush=True)

        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.throttle = float(np.clip(throttle, 0.0, 1.0))
        control.brake = float(np.clip(brake, 0.0, 1.0))
        self._step_ms.append((time.perf_counter() - t0) * 1000.0)
        return control

    def destroy(self):
        log = os.environ.get("PROJECT_LOG")
        if log and self._act_ms:
            stats = {"steps": len(self._act_ms),
                     "policy_ms_mean": float(np.mean(self._act_ms)),
                     "policy_ms_p95": float(np.percentile(self._act_ms, 95)),
                     "agent_ms_mean": float(np.mean(self._step_ms)),
                     "postproc": POSTPROC,
                     "creep_events": self._n_creep,
                     "raw_brake_mean_when_stopped": (float(np.mean(self._raw_brake_stopped))
                                                     if self._raw_brake_stopped else None)}
            os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
            with open(log, "a") as f:
                f.write(json.dumps(stats) + "\n")
