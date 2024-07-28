**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Runze Wang
* Tested on: Windows 22, VS2022, CUDA12.2, GeForce1650

This project is a CUDA implementation of Boid, an artificial life program that simulates fishes or birds’ flocking behaviors. The simulation is visualized by OpenGL.

<img src="assets/flocking-1722166514425-2.gif" style="zoom:50%;" />

## Introduction: Flocking Simulation

Flocking is defined as the action of a crowd. In nature, flocking often happens on a crowd of birds or a school of fish. Birds, for example, often fly together as a whole in the sky, moving from one position to another. Although the shape of the crowd might change a lot, it is very amazing that each bird flies as if they knew the next steps of all other birds, so that it would never diverge from the crowd and they always stay together.

Biologists have been studying the behavior of flocking for a long time. In such context, we would also call each individual a **boid**. One might very easily start to wonder whether there is any type of communications taking place within the crowd so that they could maintain as a whole. Unfortunately, however, there is no such communication mechanism between each two individuals. In fact, according to the [notes from Conrad Parker](http://www.vergenet.net/~conrad/boids/), each individual would be able to stay close to other boids as long as they follow 3 simple rules:

1. Boids try to fly towards the center of mass of neighboring boids.
2. Boids try to keep a small distance away from other objects (including other boids).
3. Boids try to match velocity with near boids.

The objective of this project would be to build a flocking simulation using CUDA with these 3 simple rules. A demo of the final result has been showed right above this section.

## Algorithm: CUDA Acceleration

The simulation is based on  the **Reynolds Boids algorithm**, along with three levels of optimization. More details is in [INSTRUCTION](./INSTRUCTION.md).

- ping-pong buffers: **avoid read & write conflict and hide latency with the SIMT of CUDA.**
  - While one buffer provides output, the other buffer can be written asynchronously. 
  - Switch over when required.
- uniform spatial grid: **avoid global loop for checking every other boid.**
  - Label every boid with an index key representing its enclosing cell.
  - Sort the key & value array.
  - Create the start & end array representing the border of the two different cells.
- semi-coherent memory access: **spatial locality that mead load data to the warp chuck by chuck.**
  -  Rearranging the boid data so that all the velocities and positions of boids in one cell are also
    contiguous in memory. 

<img src="assets/Boids Ugrids buffers naive.png" alt="buffers for generating a uniform grid using index sort" style="zoom:50%;" />

## Performance Analysis

**For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

- The FPS would decrease as the number of boids increases. This is because GPU needs to compute more boid states and thus needs more threads to finish simulation per time step.

**For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

- Mostly, the more block counts 

**For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

**Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?** 
