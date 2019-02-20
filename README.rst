[![Documentation Status](https://readthedocs.org/projects/acrobotics/badge/?version=latest)](https://acrobotics.readthedocs.io/en/latest/?badge=latest)

Acrobotics
==========
I want to be able to quickly test motion planning ideas, and Python seems like a great language for rapid prototyping. There are great libraries for robot simulation and related task, but installing them is can be a hassle and very dependent on operating system and python version.
The drawback is that I have to write a lot of stuff myself. I'm not sure if it is usefull to do this. But it will be fun and I will learn a buch.
In addition, for performance reasons, I add some c++ code wrapped using SWIG_.
(Unfortunatly more difficult to make platform independent that just Python.)

The name comes from the research group ACRO_, where I currently work as a PhD student.

The sampling based motion planning algorithms are based on the ROS package Descartes_.

I `2D version of this package`_ was the basis for this one.

.. _SWIG:     http://www.swig.org/
.. _ACRO:     https://iiw.kuleuven.be/onderzoek/acro
.. _Descartes http://wiki.ros.org/descartes
.. _2D version of this package: http://u0100037.pages.mech.kuleuven.be/planar_python_robotics
