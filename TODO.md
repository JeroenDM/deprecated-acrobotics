#todo for iterative grid refining

- implement joint limits for robot inverse kinematics

#issues

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
