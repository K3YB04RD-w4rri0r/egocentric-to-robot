# Simulator

`push_env.py` is the whole simulator: robosuite's Lift scene with a Panda arm, changed to match my setup.

| | Value | Why |
|---|---|---|
| Box | 12.8 × 12.8 × 5.7 cm, 0.2 kg | the real box's size; the mass is a guess, and slow pushes barely depend on it |
| Friction | box–table 0.4, ball–box 0.8 | a guess: the physics check couldn't pin it down |
| Pusher | an 8 mm-radius ball on a 7.7 cm stick under the closed gripper | touches at one point, like a fingertip; the gripper is 5.6 cm wide and would push like a ruler |
| Push | at the box's mid-height, 3 cm/s, in a straight line | slow enough that the box stops when the ball stops, like my pushes |
| Control | position control (OSC_POSITION), 20 steps a second | near the table's centre, the ball follows its path to within about 2 mm |

`push(env, side, offset, angle, length)`: rise, move above a point 2 cm outside the chosen face, lower, push in a straight line, lift. A push is described from the box's point of view (which face, where on it, at what angle, how far), the same way as my finger's pushes.

Nothing to run on its own: the planner (`plan/`) and the physics check (`wm/physics_check.py`) use it.
