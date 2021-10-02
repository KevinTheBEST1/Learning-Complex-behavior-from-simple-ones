# Learning-Complex-behavior-from-simple-ones
We present an approach that learns a complex behavior by combining existing simple behaviors. Our approach draws inspiration from behavior-based robotics where the outputs of sub-behaviors are combined in
pre-defined ways to form the desired behavior. In contrast,
the desired behavior in our approach is created by learning
a dynamic profile from interacting with the environment in a
reinforcement learning framework, which determines the mixtures
of sub-behaviors in different situations. More specifically,
our approach outputs n weights for n sub-behaviors for a
linear combination of their outputs. Unlike traditional transfer
learning and multi-task learning methods, the combination is
modular in the sense that the sub-behaviors to be combined
are frozen during both learning and testing. We tested our
approach with both discrete and continuous environments with
varying complexity, including Maze World, Lunar Lander, and
Bipedal Walker. Results show that our approach is able to
combine sub-behaviors to learn the desired behavior while
significantly improving sample efficiency. Furthermore, the
resulting behavior is shown to demonstrate novel combinations
of characteristics from the sub-behaviors in different situations
that contribute to the task success.

# Outputs
## Lunar Lander
The output of Lunar Lander is shown below
 - C 
 
 ![ezgif com-gif-maker](https://user-images.githubusercontent.com/74253717/135729417-6ca33ac4-3cf9-4e24-832f-c753de891f7f.gif)
 - C1

 ![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/74253717/135729415-6854e880-4301-45f7-a3d8-4f76352a5bd9.gif)
 - C2
 
 ![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/74253717/135729413-4dfd226f-2ddf-412b-b94c-5013976f0ecd.gif)
 
## Maze World
We tested our approach with both discrete and continuous environments with
Maze World which is shown below:

 - C 

 ![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/74253717/135729793-6c777fad-0a3f-48f7-8afa-c39df58e30db.gif)
 - C1
 
 ![ezgif com-gif-maker (4)](https://user-images.githubusercontent.com/74253717/135729792-15f8aa4d-5e6e-4932-bbc9-09ddff87cf45.gif)
 - C2

 ![ezgif com-gif-maker (5)](https://user-images.githubusercontent.com/74253717/135729790-afb3088c-29a3-4460-abc0-7c4224d1c0a4.gif)
 
