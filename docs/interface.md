# Interface spec (Z <-> Yuhao)

Agreed on: ____ (fill in date). Any change goes through both of us and this file.

## 1. Dataset (Z produces, Yuhao reads)

```
dataset/
  town01_clearnoon_v50_seed0/       one folder per episode: town_weather_v<vehicles>_seed<seed>
    rgb/000000.png                  uint8 (240, 320, 3), RGB order
    seg/000000.png                  uint8 (240, 320), raw CARLA semantic tag (not a color image)
    depth/000000.npy                float16 (240, 320), meters
    measurements.csv                one row per saved frame
    meta.json                       episode config
```

measurements.csv columns:

| column    | unit / range | note |
|-----------|--------------|------|
| frame     | int          | matches the file index |
| speed     | m/s          | |
| command   | int 1-6      | CARLA RoadOption: 1 LEFT, 2 RIGHT, 3 STRAIGHT, 4 LANEFOLLOW, 5 CHANGELANELEFT, 6 CHANGELANERIGHT |
| steer     | [-1, 1]      | expert control (label) |
| throttle  | [0, 1]       | expert control (label) |
| brake     | [0, 1]       | expert control (label) |
| x, y, yaw | m, m, deg    | ego pose, for analysis only, never a policy input |
| collision | 0/1          | 1 once the ego has collided in this episode |

meta.json: carla_version, town, weather, vehicles, seed, fps_saved (10), num_frames,
camera {x, y, z, pitch, fov, width, height}, num_collisions.

Fixed choices:
- Simulation 20 Hz synchronous; every 2nd tick saved -> 10 Hz data.
- RGB, seg and depth cameras share one pose: x=1.5, z=2.4, pitch=0, FOV 90, 320x240
  (change only if the HW1 agent used another pose, and update this line).
- seg: label taken from the R channel of CARLA's semantic camera. Label IDs changed in
  CARLA 0.9.14, so always read carla_version from meta.json.
- depth: decoded on Z's side, depth_m = (R + G*256 + B*256^2) / (256^3 - 1) * 1000.
- Episodes with collision frames are kept; training code may drop frames after the first collision.

## 2. Policy (Yuhao produces, Z calls)

See policy/policy_interface.py.

```python
policy = load_policy(path, device="cuda")   # "dummy" -> DummyPolicy
policy.reset()                              # start of each route
steer, throttle, brake = policy.act(obs)
# obs["rgb"]     uint8 (240, 320, 3) RGB
# obs["speed"]   float m/s
# obs["command"] int 1-6
# obs["seg"], obs["depth"]   only when policy.needs_gt is True (GT-Explicit), same format as the dataset
```

- The checkpoint stores its variant name, so load_policy needs only the path.
- Frozen VFMs, preprocessing and the class mapping live inside the Policy. Training-time
  feature extraction must call the same preprocessing code.

## 3. Frozen models (must be identical at train and eval time)

| role | model | Hugging Face id |
|------|-------|-----------------|
| implicit features | DINOv2 ViT-S/14 | facebook/dinov2-small |
| depth | Depth Anything V2 Small | depth-anything/Depth-Anything-V2-Small-hf |
| segmentation | SegFormer-B1, Cityscapes | nvidia/segformer-b1-finetuned-cityscapes-1024-1024 |

| model | input size (H x W) | cached output |
|-------|--------------------|---------------|
| DINOv2 | 224 x 294 (16 x 21 patches) | patch tokens (384, 16, 21), float16, CLS dropped |
| Depth Anything V2 | 280 x 378 | relative depth upsampled to 240 x 320, float16 |
| SegFormer-B1 | 480 x 640 | argmax upsampled to 240 x 320, mapped to 10 shared classes, uint8 |

- ImageNet mean/std normalization, resizing done with F.interpolate in policy/vfm.py
  (no Hugging Face image processor). Inference in fp16 on GPU.
- Depth representation, same for GT and prediction: GT inv = 1 / clip(depth_m, 1, 100);
  both GT inv and predicted relative depth are min-max normalized per frame to [0, 1].
- Segmentation: both CARLA tags and Cityscapes trainIds are mapped to 10 shared classes
  (0 other, 1 road, 2 sidewalk, 3 building/pole, 4 vegetation/terrain, 5 sky, 6 vehicle,
  7 person, 8 traffic light, 9 traffic sign). The CARLA table depends on carla_version (>= 0.9.14 or not).
- Commands 5 and 6 (lane changes) are treated as 4 in both training and act().

## 4. Handoff

- Data: Z uploads tar.gz to /projectnb/rlvn/students/<user>/ec518/dataset/ on SCC.
- Checkpoints: Yuhao puts .pt files in the shared drive folder; file name = variant_seed.pt.
- Code: everything else through git.

## 5. Round-trip checks (do both before Tuesday)

1. Data: Yuhao runs `python collect/check_episode.py <episode>` on Z's test episode and gets "OK".
2. Policy: Z runs the leaderboard with `--agent-config=dummy` and the car drives a route.
