# scikit-robot-hsr

## Installation

```bash
pip install git+https://github.com/iory/scikit-robot-hsr.git
```

## Quick-Start

```
import skrobot
import skrobot_hsr
robot_model = skrobot_hsr.HSRB()
robot_model.reset_pose()

viewer = skrobot.viewers.TrimeshSceneViewer()
viewer.add(robot_model)
viewer.show()

robot_model.rarm.move_end_pos((0.1, 0.0, 0.0), use_base=False)
robot_model.rarm.move_end_pos((0.1, 0.0, 0.0), use_base=True)
```
