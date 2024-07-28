#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
#include "device_launch_parameters.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

//! Block size used for CUDA kernel launch. 
#define blockSize 32

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

//! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
//! These are called ping-pong buffers.
glm::vec3* dev_pos = nullptr;
glm::vec3* dev_vel1 = nullptr;
glm::vec3* dev_vel2 = nullptr;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int* dev_particleArrayIndices = nullptr; // What index in dev_pos and dev_velX represents this particle?
int* dev_particleGridIndices = nullptr; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;


int *dev_gridCellStartIndices = nullptr; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices = nullptr;   // to this cell?
int2* dev_gridCellRanges = nullptr;

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_pos_gathered = nullptr;
glm::vec3* dev_vel_gathered = nullptr;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellRanges, gridCellCount * sizeof(int) * 2);
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
  
  cudaMalloc((void**)&dev_pos_gathered, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_gathered failed!");

  cudaMalloc((void**)&dev_vel_gathered, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel_gathered failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids

  glm::vec3 center(0.0f, 0.0f, 0.0f);
  glm::vec3 separate(0.0f, 0.0f, 0.0f);
  glm::vec3 cohesion(0.0f, 0.0f, 0.0f);

  int rule1NumNeighbors = 0;
  int rule2NumNeighbors = 0;
  glm::vec3 res_vel = glm::vec3(0.0f, 0.0f, 0.0f);

  glm::vec3 thisPos = pos[iSelf];

  // compute the velocity change based on the three rules
  for (int i = 0; i < N; i++) {
    if (i == iSelf) {
      continue;
    }

    glm::vec3 otherPos = pos[i];
    float distance = glm::length(otherPos - thisPos);

    if (distance < rule1Distance) {
      center += otherPos;
      rule1NumNeighbors++;
    }

    if (distance < rule2Distance) {
      separate -= otherPos - thisPos;
    }

    if (distance < rule3Distance) {
      cohesion += vel[i];
      rule2NumNeighbors++;
    }
  }

  // update velocity
  if (rule1NumNeighbors > 0) {
    center /= rule1NumNeighbors;
    res_vel += (center - thisPos) * rule1Scale;
  }
  res_vel += separate * rule2Scale;
  if (rule2NumNeighbors > 0) {
    cohesion /= rule2NumNeighbors;
    res_vel += cohesion * rule3Scale;
  }


  return res_vel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  //! Ping-pong the velocity buffers: avoid read and write conflicts, reduce latency
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. 
  //? Question: why NOT vel1?

  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  // Compute a new velocity based on pos and vel1
  glm::vec3 delta_vel = computeVelocityChange(N, index, pos, vel1);
  
  // Ping-pong the velocity buffers
  // Update the velocity2
  glm::vec3 new_vel = vel1[index] + delta_vel;
  // Clamp the speed
  float speed = glm::length(new_vel);
  if (speed > maxSpeed) {
    new_vel = new_vel * maxSpeed / speed;
  }

  vel2[index] = new_vel;


}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(const int N, const int gridResolution,
  const glm::vec3 gridMin, const float inverseCellWidth,
  const glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    int thisParticleIndex = threadIdx.x + (blockIdx.x * blockDim.x);
    if (thisParticleIndex >= N) {
      return;
    }
  
    glm::vec3 thisPos = pos[thisParticleIndex];
    glm::vec3 thisCellPos = (thisPos - gridMin) * inverseCellWidth;
    glm::ivec3 thisCellIndex3{ thisCellPos.x, thisCellPos.y, thisCellPos.z };
    int thisCellIndex = gridIndex3Dto1D(thisCellIndex3.x, thisCellIndex3.y, thisCellIndex3.z, gridResolution);

    indices[thisParticleIndex] = thisParticleIndex;
    gridIndices[thisParticleIndex] = thisCellIndex;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetInt2Buffer(int N, int2 *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = make_int2(value, value);
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int2 *gridCellRanges) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  int particleIndex = threadIdx.x + (blockIdx.x * blockDim.x);
  if (particleIndex >= N) {
    return;
  }
  int gridIndex = particleGridIndices[particleIndex];
  if(particleIndex == 0){
    gridCellRanges[gridIndex].x = 0;
  }else{
    int prevGridIndex = particleGridIndices[particleIndex-1];
    if(gridIndex != prevGridIndex){
      gridCellRanges[gridIndex].x = particleIndex;
      gridCellRanges[prevGridIndex].y = particleIndex;
    }
    if(particleIndex == N-1){
      gridCellRanges[gridIndex].y = N;
    }
  }

}

__global__ void kernUpdateVelNeighborSearchScattered(
  const int N, const int gridResolution, const glm::vec3 gridMin,
  const float inverseCellWidth, const float cellWidth,
  const int2 *gridCellRanges,
  const int *particleArrayIndices,
  const glm::vec3 *pos,const glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  int thisParticleIndex = threadIdx.x + (blockIdx.x * blockDim.x);
  if (thisParticleIndex >= N) {
    return;
  }

  glm::vec3 thisPos = pos[thisParticleIndex];
  glm::vec3 thisCellPos = (thisPos - gridMin) * inverseCellWidth;
  glm::ivec3 thisCellIndex3{ thisCellPos.x, thisCellPos.y, thisCellPos.z };
  
  glm::vec3 absolutePos = thisCellPos - glm::floor(thisCellPos) - 0.5f;
  glm::ivec3 quadrant;
  quadrant.x = (absolutePos.x > 0) ? 1 : -1;
  quadrant.y = (absolutePos.y > 0) ? 1 : -1;
  quadrant.z = (absolutePos.z > 0) ? 1 : -1;

  int rule1NumNeighbors = 0;
  int rule2NumNeighbors = 0;
  
  glm::vec3 delta_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 center(0.0f, 0.0f, 0.0f);
  glm::vec3 separate(0.0f, 0.0f, 0.0f);
  glm::vec3 cohesion(0.0f, 0.0f, 0.0f);

  for(int k=0; k < 8; k++){
    glm::ivec3 offset {k & 1, (k & 2) >> 1, (k & 4) >> 2};
    glm::ivec3 neighborCellIndex3 = thisCellIndex3 + offset * quadrant;
    int neighborCellIndex = gridIndex3Dto1D(neighborCellIndex3.x, neighborCellIndex3.y, neighborCellIndex3.z, gridResolution);
    if (neighborCellIndex < 0 || neighborCellIndex >= gridResolution * gridResolution * gridResolution) {
      continue;
    }
    int2 range = gridCellRanges[neighborCellIndex];
    for (int i = range.x; i < range.y; i++) {
      int otherParticleIndex = particleArrayIndices[i];
      if (otherParticleIndex == thisParticleIndex) {
        continue;
      }
      glm::vec3 otherPos = pos[otherParticleIndex];
      float distance = glm::length(otherPos - thisPos);

      if(distance > cellWidth)
        continue;
      if (distance < rule1Distance) {
        center += otherPos;
        rule1NumNeighbors++;
      }
      if (distance < rule2Distance) {
        separate -= otherPos - thisPos;
      }
      if (distance < rule3Distance) {
        cohesion += vel1[otherParticleIndex];
        rule2NumNeighbors++;
      }
    }
  }

  if (rule1NumNeighbors > 0) {
    center /= rule1NumNeighbors;
    delta_vel += (center - thisPos) * rule1Scale;
  }

  delta_vel += separate * rule2Scale;

  if (rule2NumNeighbors > 0) {
    cohesion /= rule2NumNeighbors;
    delta_vel += cohesion * rule3Scale;
  }

  glm::vec3 new_vel = vel1[thisParticleIndex] + delta_vel;
  // Clamp the speed
  float speed = glm::length(new_vel);
  if (speed > maxSpeed) {
    new_vel = new_vel * maxSpeed / speed;
  }
  vel2[thisParticleIndex] = new_vel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  const int N, const int gridResolution, const glm::vec3 gridMin,
  const float inverseCellWidth, const float cellWidth,
  const int2 *gridCellRanges,
  const glm::vec3 *pos,const glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  int thisParticleIndex = threadIdx.x + (blockIdx.x * blockDim.x);
  if (thisParticleIndex >= N) {
    return;
  }

  glm::vec3 thisPos = pos[thisParticleIndex];
  glm::vec3 thisCellPos = (thisPos - gridMin) * inverseCellWidth;
  glm::ivec3 thisCellIndex3{ thisCellPos.x, thisCellPos.y, thisCellPos.z };
  
  glm::vec3 absolutePos = thisCellPos - glm::floor(thisCellPos) - 0.5f;
  glm::ivec3 quadrant;
  quadrant.x = (absolutePos.x > 0) ? 1 : -1;
  quadrant.y = (absolutePos.y > 0) ? 1 : -1;
  quadrant.z = (absolutePos.z > 0) ? 1 : -1;

  int rule1NumNeighbors = 0;
  int rule2NumNeighbors = 0;
  
  glm::vec3 delta_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 center(0.0f, 0.0f, 0.0f);
  glm::vec3 separate(0.0f, 0.0f, 0.0f);
  glm::vec3 cohesion(0.0f, 0.0f, 0.0f);

  for(int k=0; k < 8; k++){
    glm::ivec3 offset {k & 1, (k & 2) >> 1, (k & 4) >> 2};
    glm::ivec3 neighborCellIndex3 = thisCellIndex3 + offset * quadrant;
    int neighborCellIndex = gridIndex3Dto1D(neighborCellIndex3.x, neighborCellIndex3.y, neighborCellIndex3.z, gridResolution);
    if (neighborCellIndex < 0 || neighborCellIndex >= gridResolution * gridResolution * gridResolution) {
      continue;
    }
    int2 range = gridCellRanges[neighborCellIndex];
    for (int i = range.x; i < range.y; i++) {
      glm::vec3 otherPos = pos[i];
      glm::vec3 otherVel = vel1[i];
      float distance = glm::length(otherPos - thisPos);

      if(distance > cellWidth || distance == 0)
        continue;
      if (distance < rule1Distance) {
        center += otherPos;
        rule1NumNeighbors++;
      }
      if (distance < rule2Distance) {
        separate -= otherPos - thisPos;
      }
      if (distance < rule3Distance) {
        cohesion += otherVel;
        rule2NumNeighbors++;
      }
    }
  }

  if (rule1NumNeighbors > 0) {
    center /= rule1NumNeighbors;
    delta_vel += (center - thisPos) * rule1Scale;
  }

  delta_vel += separate * rule2Scale;

  if (rule2NumNeighbors > 0) {
    cohesion /= rule2NumNeighbors;
    delta_vel += cohesion * rule3Scale;
  }

  glm::vec3 new_vel = vel1[thisParticleIndex] + delta_vel;
  // Clamp the speed
  float speed = glm::length(new_vel);
  if (speed > maxSpeed) {
    new_vel = new_vel * maxSpeed / speed;
  }
  vel2[thisParticleIndex] = new_vel;



}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  // Update the velocity first
  kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, dev_vel1, dev_vel2);


  // TODO-1.2 ping-pong the velocity buffers
  // Swap the velocity buffers
  std::swap(dev_vel1, dev_vel2);
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);

}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // Label each particle with its array index as well as its grid index.
  kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
      numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, 
      dev_pos, 
      dev_particleArrayIndices, dev_particleGridIndices);
  
  // Unstable key sort using Thrust
  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
  thrust::sort_by_key(dev_thrust_particleGridIndices, 
                      dev_thrust_particleGridIndices + numObjects, 
                      dev_thrust_particleArrayIndices);

  // Reset the grid cell start and end indices
  kernResetInt2Buffer << <fullBlocksPerGrid, blockSize >> >(gridCellCount, dev_gridCellRanges, -1);

  // Identify the start point of each cell in the gridIndices array.
  kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleGridIndices, dev_gridCellRanges);

  // Update a boid's velocity using the uniform grid to reduce the number of boids that need to be checked.
  kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >(
      numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
      dev_gridCellRanges, dev_particleArrayIndices, 
      dev_pos, dev_vel1, dev_vel2);

  // Swap the velocity buffers as Ping-pong buffers
  std::swap(dev_vel1, dev_vel2);

  // Update the position
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);

}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // Label each particle with its array index as well as its grid index.
  kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
      numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, 
      dev_pos, 
      dev_particleArrayIndices, dev_particleGridIndices);
  
  // Unstable key sort using Thrust
  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
  thrust::sort_by_key(dev_thrust_particleGridIndices, 
                      dev_thrust_particleGridIndices + numObjects, 
                      dev_thrust_particleArrayIndices);

  // Reset the grid cell start and end indices
  kernResetInt2Buffer << <fullBlocksPerGrid, blockSize >> >(gridCellCount, dev_gridCellRanges, -1);

  // Identify the start point of each cell in the gridIndices array.
  kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleGridIndices, dev_gridCellRanges);

  // gather the position and velocity data according to the sorted indices
  auto thrust_pos = thrust::device_pointer_cast(dev_pos);
  auto thrust_vel1 = thrust::device_pointer_cast(dev_vel1);
  auto thrust_pos_gathered = thrust::device_pointer_cast(dev_pos_gathered);
  auto thrust_vel_gathered = thrust::device_pointer_cast(dev_vel_gathered);

  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle
  // all the particle data in the simulation array.
  thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, thrust_pos,
                 thrust_pos_gathered);
  thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, thrust_vel1,
                 thrust_vel_gathered);

  // Update a boid's velocity using the uniform grid to reduce the number of boids that need to be checked.
  kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >(
      numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
      dev_gridCellRanges, 
      dev_pos_gathered, dev_vel_gathered, dev_vel2);

  // Swap the velocity buffers as Ping-pong buffers
  std::swap(dev_vel1, dev_vel2);
  std::swap(dev_pos, dev_pos_gathered);
  // Update the position
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellRanges);

  cudaFree(dev_pos_gathered);
  cudaFree(dev_vel_gathered);

  checkCUDAErrorWithLine("cudaFree failed!");
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
