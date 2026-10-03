# RoboHub data spec

A generated clip becomes training data only if it passes every gate below. The thresholds live in
`src/rohub/gates.py` (`SPEC`); `tests/test_gates.py` fails if this page and the code disagree.
A rejected clip keeps the first failing gate's reason, and the site shows it.

Task: "put the red block in the bowl". One hand, pinch grasp of a ~3 cm cube, lift, carry, release into a bowl,
tabletop, about 30 cm reach. Robot: LeRobot SO-101 (5 DOF arm + parallel gripper), MuJoCo, 30 fps recording.

## Clip gates (from the generated video)

| id | gate | pass if |
|---|---|---|
| fps | frame rate | 23 to 31 fps |
| length | clip length | 3 to 12 s |
| scene | scene audit by Claude (first frame) | exactly 1 red block, 1 bowl, 1 hand |
| one_object | one red object, pixel check | second-largest red blob / largest <= 0.30 |
| no_duplicate | no second red object appears, any frame | the two largest red blobs are never more than 2 block widths apart while the larger is >= 0.3 and the smaller >= 0.05 of the block's area, on more than 3 frames |
| hand_visible | hand and wrist visible | MediaPipe hand score >= 0.5 on >= 0.95 of frames in the window (clip start to release + 0.5 s) |
| block_visible | block tracked | red block found on >= 0.95 of frames in the same window |
| events | grasp, lift, release | all three found, in that order (from the block's motion) |
| coupled | block moves with the hand | while carried, the block drifts <= 1.0 block widths from the fingers |
| lift | lift height | >= 0.03 m |
| carry | carry distance | >= 0.05 m |
| video_in_bowl | video ends with block in bowl (Claude, last frame) | true, and one red block |

## Robot gates (after retargeting, in a MuJoCo physics replay)

| id | gate | pass if |
|---|---|---|
| ik | IK inside SO-101 joint limits | 0 control ticks pinned at a joint limit, IK error <= 5 mm (0.005 m) on every tick |
| vel | joint velocity | <= 3 rad/s on every arm joint (30 fps commands) |
| acc | joint acceleration | <= 60 rad/s^2 |
| jerk | joint jerk | <= 2500 rad/s^3 |
| close_before_lift | gripper closes before lift | the gripper command is fully closed before the cube rises 1 cm |
| open_over_bowl | gripper opens over the bowl | when the gripper opens the cube is < 0.05 m from the bowl centre |
| sim_success | cube ends in bowl | the physics replay ends with the cube resting inside the bowl, after a lift |
| self_collision | no self-collision | 0 contacts between non-adjacent arm links |
| episode_len | episode length | 2 to 30 s |

## Why these numbers

- 3 rad/s is under the STS3215's no-load speed at 7.4 V (about 4.7 rad/s), so the real arm could follow.
  60 rad/s^2 and 2500 rad/s^3 were set above what a smooth human carry produces after retargeting
  (v01: 0.9 rad/s, 26 rad/s^2, 855 rad/s^3) and below the spikes of a wrist-roll flip in the scripted expert
  (24 rad/s, 734 rad/s^2), which is exactly the kind of motion a real servo should not be asked to do.
- 0.95 visibility: a gap longer than a few frames means the grasp point is invented, not measured.
- The duplicate test catches the most common generated-video failure we saw: after the release the hand leaves
  holding a second copy of the block while the first sits in the bowl. Three frames of tolerance absorb wood knots
  and skin that the red threshold picks up for a frame or two.
- The block-width coupling test catches generated videos where the object slides, morphs or teleports on its
  own while the hand only pretends to hold it.

## Output format

LeRobot dataset format v3.0 written with `LeRobotDataset.create` (lerobot 0.6.1, Python 3.12, uv), robot_type
`so101_follower`, 30 fps. Features: `observation.state` and `action` (the six `{motor}.pos` values: five arm
joints in degrees, gripper 0 to 100), `observation.images.front` (320x240 AV1 video from the sim's front camera)
and `observation.environment_state` (cube x, y, z in metres). `rohub_episodes.json` records which clip each
episode came from and whether it is the retargeted clip itself or a re-anchored copy.
