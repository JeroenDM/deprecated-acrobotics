#todo for iterative grid refining

- simple heuristic to detect configuration jumps
  without knowing the robot type:
  monitor cost between two configurations
  outside 2 stadard deviations could be considered jump?
- specify tolerance on euler angles
- cone constraints
- other cost functions (deviation from nominal?)
- implement joint limits for robot inverse kinematics
- fix tests since I changed a lot in path.py
- redundant joint tolerance reduction
 - add weights to cost function

#issues

- memory usage seems to be an issue... for 10000 samples per point and saving
thre results. possible **memory leak** in c++ code??
 +/- 2 GiB for 10 000 samples for 15 points
- How to sample quaternions around a nominal value
 -> look at ompl
 => create random number generator classes to support
 all this stuff
 add this to pyquaternion instead of implementing everything from scratch
 **But**
 then I also have to add halton sampling and so on
 to this library...


Note on math vs numpy
https://stackoverflow.com/questions/3650194/are-numpys-math-functions-faster-than-pythons

matplotlib.font_manager._rebuild()
