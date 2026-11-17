"""Collect expert driving data in CARLA (synchronous mode).

Each episode saves RGB / semantic segmentation / depth from three cameras that share
one pose, plus speed, navigation command and expert control. The output layout follows
docs/interface.md.

The expert is CARLA's BehaviorAgent (PythonAPI/carla/agents). Works with the 0.9.10 agents
API (our setup: CARLA 0.9.10.1, Python 3.7) and with 0.9.12+; see set_destination,
route_done and update_agent for the version differences.

Usage (start CarlaUE4.exe first, in the conda env "carla"):
    python collect_data.py --town Town01 --weather ClearNoon --frames 1200 --seed 0
"""
import argparse
import csv
import json
import math
import os
import queue
import random
import sys

import numpy as np
from PIL import Image

import carla

CARLA_ROOT = os.environ.get("CARLA_ROOT", r"D:\WindowsNoEditor")
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402

# Camera settings: keep these identical to the evaluation agent (eval/project_agent.py).
W, H, FOV = 320, 240, 90
CAM_X, CAM_Z = 1.5, 2.4  # TODO: change to the camera pose your HW1 agent used, if different

FPS_SIM = 20  # simulation tick rate (fixed_delta_seconds = 0.05), same as the leaderboard


def to_bgra(image):
    """carla.Image -> uint8 array of shape (H, W, 4) in BGRA order."""
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)


def decode_depth(bgra):
    """CARLA depth camera (raw BGRA encoding) -> depth in meters, float32."""
    b = bgra[:, :, 0].astype(np.float32)
    g = bgra[:, :, 1].astype(np.float32)
    r = bgra[:, :, 2].astype(np.float32)
    return (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0


def get_frame(q, frame, timeout=5.0):
    """Pop sensor messages until the one for `frame` arrives."""
    while True:
        data = q.get(timeout=timeout)
        if data.frame == frame:
            return data


def set_destination(agent, ego, location):
    """Plan a route from the ego's current position. Call only after at least one world.tick():
    in synchronous mode a freshly spawned actor reports location (0, 0, 0) until the first tick."""
    if hasattr(agent, "done"):
        agent.set_destination(location)  # CARLA >= 0.9.12: set_destination(end, start=None)
    else:
        agent.set_destination(ego.get_location(), location, clean=True)  # 0.9.10 / 0.9.11


def route_done(agent):
    if hasattr(agent, "done"):
        return agent.done()  # CARLA >= 0.9.12
    lp = agent._local_planner
    q = getattr(lp, "_waypoints_queue", None)
    if q is None:
        q = getattr(lp, "waypoints_queue", [])
    buf = getattr(lp, "_waypoint_buffer", [])  # 0.9.10 keeps the next few waypoints here
    return len(q) + len(buf) < 5


class _WorldShim(object):
    """0.9.9-style update_information(world) only needs world.player."""
    def __init__(self, ego):
        self.player = ego


def update_agent(agent, ego):
    """CARLA 0.9.10/0.9.11 BehaviorAgent needs update_information() before every run_step();
    0.9.12+ does it inside run_step() and has no public update_information."""
    if not hasattr(agent, "update_information"):
        return
    try:
        agent.update_information()
    except TypeError:
        agent.update_information(_WorldShim(ego))


def current_command(agent):
    """Navigation command as an int (CARLA RoadOption): 1 LEFT, 2 RIGHT, 3 STRAIGHT,
    4 LANEFOLLOW, 5 CHANGELANELEFT, 6 CHANGELANERIGHT. VOID/unknown -> 4."""
    opt = getattr(agent._local_planner, "target_road_option", None)
    if opt is None or opt.value < 1:
        return 4
    return int(opt.value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default="Town01")
    p.add_argument("--weather", default="ClearNoon", help="name in carla.WeatherParameters")
    p.add_argument("--vehicles", type=int, default=50, help="background traffic")
    p.add_argument("--frames", type=int, default=1200, help="simulation ticks (20 Hz)")
    p.add_argument("--save_every", type=int, default=2, help="2 -> saved data is 10 Hz")
    p.add_argument("--warmup", type=int, default=40, help="ticks to skip at the start")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="dataset")
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)

    client = carla.Client(a.host, a.port)
    client.set_timeout(60.0)
    world = client.load_world(a.town)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS_SIM
    world.apply_settings(settings)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    if hasattr(tm, "set_random_device_seed"):
        tm.set_random_device_seed(a.seed)
    world.set_weather(getattr(carla.WeatherParameters, a.weather))

    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)

    actors, sensors = [], []
    ep_name = f"{a.town.lower()}_{a.weather.lower()}_v{a.vehicles}_seed{a.seed}"
    ep_dir = os.path.join(a.out, ep_name)
    for d in ("rgb", "seg", "depth"):
        os.makedirs(os.path.join(ep_dir, d), exist_ok=True)

    try:
        # the blueprint id is "mkz2017" in CARLA 0.9.10 and "mkz_2017" in newer versions
        ego_ids = [b.id for b in bp_lib.filter("vehicle.lincoln.mkz*")]
        ego_bp = bp_lib.find(ego_ids[0]) if ego_ids else bp_lib.filter("vehicle.*")[0]
        ego_bp.set_attribute("role_name", "hero")
        ego = world.spawn_actor(ego_bp, spawns[0])
        actors.append(ego)

        for sp in spawns[1:1 + a.vehicles]:
            v = world.try_spawn_actor(random.choice(bp_lib.filter("vehicle.*")), sp)
            if v is not None:
                v.set_autopilot(True, tm.get_port())
                actors.append(v)

        cam_tf = carla.Transform(carla.Location(x=CAM_X, z=CAM_Z))
        queues = {}
        for name, bp_id in [("rgb", "sensor.camera.rgb"),
                            ("seg", "sensor.camera.semantic_segmentation"),
                            ("depth", "sensor.camera.depth")]:
            cbp = bp_lib.find(bp_id)
            cbp.set_attribute("image_size_x", str(W))
            cbp.set_attribute("image_size_y", str(H))
            cbp.set_attribute("fov", str(FOV))
            s = world.spawn_actor(cbp, cam_tf, attach_to=ego)
            q = queue.Queue()
            s.listen(q.put)
            sensors.append(s)
            queues[name] = q

        collisions = []
        col = world.spawn_actor(bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        col.listen(lambda e: collisions.append(e.frame))
        sensors.append(col)

        # one tick so the ego (and everything just spawned) has its real transform
        world.tick()
        agent = BehaviorAgent(ego, behavior="normal")
        set_destination(agent, ego, random.choice(spawns[1:]).location)

        rows, idx = [], 0
        for t in range(a.frames):
            frame = world.tick()
            imgs = {k: get_frame(q, frame) for k, q in queues.items()}

            if route_done(agent):
                set_destination(agent, ego, random.choice(spawns[1:]).location)
            update_agent(agent, ego)
            command = current_command(agent)
            control = agent.run_step()
            ego.apply_control(control)

            if t < a.warmup or t % a.save_every != 0:
                continue

            vel = ego.get_velocity()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            tf = ego.get_transform()

            rgb = np.ascontiguousarray(to_bgra(imgs["rgb"])[:, :, 2::-1])  # BGRA -> RGB
            seg = np.ascontiguousarray(to_bgra(imgs["seg"])[:, :, 2])      # tag lives in R channel
            depth = decode_depth(to_bgra(imgs["depth"])).astype(np.float16)

            Image.fromarray(rgb).save(os.path.join(ep_dir, "rgb", f"{idx:06d}.png"))
            Image.fromarray(seg).save(os.path.join(ep_dir, "seg", f"{idx:06d}.png"))
            np.save(os.path.join(ep_dir, "depth", f"{idx:06d}.npy"), depth)

            rows.append({
                "frame": idx, "speed": round(speed, 4), "command": command,
                "steer": round(float(control.steer), 5), "throttle": round(float(control.throttle), 5),
                "brake": round(float(control.brake), 5),
                "x": round(tf.location.x, 3), "y": round(tf.location.y, 3), "yaw": round(tf.rotation.yaw, 3),
                "collision": int(len(collisions) > 0),
            })
            idx += 1
            if idx % 100 == 0:
                print(f"saved {idx} frames")

        with open(os.path.join(ep_dir, "measurements.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        meta = {
            "carla_version": client.get_server_version(),
            "town": a.town, "weather": a.weather, "vehicles": a.vehicles, "seed": a.seed,
            "fps_saved": FPS_SIM / a.save_every, "num_frames": idx,
            "camera": {"x": CAM_X, "y": 0.0, "z": CAM_Z, "pitch": 0.0, "fov": FOV, "width": W, "height": H},
            "num_collisions": len(collisions),
        }
        with open(os.path.join(ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"done: {ep_dir}, {idx} frames, collisions={len(collisions)}")

    finally:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
        for s in sensors:
            s.stop()
            s.destroy()
        for v in actors:
            v.destroy()


if __name__ == "__main__":
    main()
