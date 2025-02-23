#include <cstdio>
#include <cuda.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "pathtrace.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "sampler.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

// Random number generator
// Iter and index are used to generate a seed for the random number generator
// To make sure that each pixel has a different seed, we use the pixel index
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

// Stream Compaction Valid Path
struct is_valid{
	__host__ __device__
		bool operator()(const PathSegment& path) {
		return path.remainingBounces > 0;
	}
};

// Intersection Compare
struct compareIntersection {
	__host__ __device__
		bool operator()(const Intersection& i1, const Intersection& i2) {
		return i1.materialId < i2.materialId;
	}
};


//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static BVHNode* dev_scene_bvh = NULL;
static Triangle** dev_trimesh_ptr = NULL;
static BVHNode** dev_tribvh_ptr = NULL;
static Material* dev_materials = NULL;
static Bitmap* dev_bmp_ptr = NULL;
static PathSegment* dev_paths = NULL;
static Intersection* dev_intersections = NULL;
static cudaTextureObject_t* hst_texs = NULL;
static cudaTextureObject_t* dev_texs = NULL;


void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_intersections, pixelcount * sizeof(Intersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(Intersection));


	checkCUDAError("pathtraceInit");
}

void resourceInit(Scene *scene) {

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	if (!scene->trimeshes.empty()) {
		//cudaMalloc((void**)&dev_trimeshes, scene->trimeshes.size() * sizeof(TriangleMesh));
		cudaMalloc((void**)&dev_trimesh_ptr, scene->trimeshes.size() * sizeof(Triangle*));
		for (int i = 0; i < scene->trimeshes.size(); i++)
		{
			int numTriangles = scene->trimeshes[i].num;
			Triangle* dev_triangle = NULL;
			cudaMalloc(&(dev_triangle), numTriangles * sizeof(Triangle));
			cudaMemcpy(dev_triangle, scene->trimeshes[i].triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

			cudaMemcpy((void**)&(dev_trimesh_ptr[i]), &dev_triangle, sizeof(unsigned char*), cudaMemcpyHostToDevice);
		}
	}
	else {
		dev_trimesh_ptr = NULL;
	}

	if (!scene->tri_bvhs.empty()){
		cudaMalloc((void**)&dev_tribvh_ptr, scene->tri_bvhs.size() * sizeof(BVHNode*));

		for (int i = 0; i < scene->tri_bvhs.size(); i++)
		{
			int numNodes = scene->tri_bvhs[i].bvh_nodes.size();
			BVHNode* dev_tribvh_nodes = NULL;
			cudaMalloc((void**)&dev_tribvh_nodes, numNodes * sizeof(BVHNode));
			cudaMemcpy(dev_tribvh_nodes, scene->tri_bvhs[i].bvh_nodes.data(), numNodes * sizeof(BVHNode), cudaMemcpyHostToDevice);

			cudaMemcpy((void**)&(dev_tribvh_ptr[i]), &dev_tribvh_nodes, sizeof(BVHNode*), cudaMemcpyHostToDevice);

		}
	} else {
		dev_tribvh_ptr = NULL;
	}
	int numNodeds = scene->scene_bvh.bvh_nodes.size();
	if (numNodeds > 0) {
		cudaMalloc(&dev_scene_bvh, sizeof(BVHNode) * numNodeds);
		cudaMemcpy(dev_scene_bvh, scene->scene_bvh.bvh_nodes.data(), sizeof(BVHNode) * numNodeds, cudaMemcpyHostToDevice);
	}
	else {
		dev_scene_bvh = NULL;
	}

	if (!scene->bitmaps.empty()) {
		//cudaMalloc(&dev_bmp_ptr, scene->bitmaps.size() * sizeof(Bitmap));
		hst_texs = new cudaTextureObject_t[scene->bitmaps.size()];
		for (int i = 0; i < scene->bitmaps.size(); i++)
		{
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
			cudaArray_t curArray;
			cudaMallocArray(&curArray, &channelDesc, scene->bitmaps[i].width, scene->bitmaps[i].height);
			cudaMemcpy2DToArray(curArray, 0, 0,
				scene->bitmaps[i].pixels, scene->bitmaps[i].width * sizeof(uchar4),
				scene->bitmaps[i].width * sizeof(uchar4), scene->bitmaps[i].height, cudaMemcpyHostToDevice);

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = curArray;

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeNormalizedFloat;
			texDesc.normalizedCoords = 1;

			cudaCreateTextureObject(&hst_texs[i], &resDesc, &texDesc, NULL);

			//int numPixel = scene->bitmaps[i].width * scene->bitmaps[i].height;
			//unsigned char* dev_pixels = NULL;
			//cudaMalloc(&dev_pixels, numPixel * sizeof(unsigned char) * 4);
			//cudaMemcpy(dev_pixels, scene->bitmaps[i].pixels, numPixel * sizeof(unsigned char) * 4, cudaMemcpyHostToDevice);

			//cudaMemcpy(&(dev_bmp_ptr[i].pixels), &dev_pixels, sizeof(unsigned char*), cudaMemcpyHostToDevice);
			//cudaMemcpy(&(dev_bmp_ptr[i].width), &(scene->bitmaps[i].width), sizeof(int), cudaMemcpyHostToDevice);
			//cudaMemcpy(&(dev_bmp_ptr[i].height), &(scene->bitmaps[i].height), sizeof(int), cudaMemcpyHostToDevice);
		}


		cudaMalloc(&dev_texs, scene->bitmaps.size() * sizeof(cudaTextureObject_t));
		cudaMemcpy(dev_texs, hst_texs, scene->bitmaps.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
	}
	else {
		dev_bmp_ptr = NULL;
		dev_texs = NULL;
	}
	// TODO: initialize any extra device memeory you need
	checkCUDAError("resourceInit");
}

void resourceFree() {

	cudaFree(dev_geoms);
	cudaFree(dev_materials);

	if (hst_scene == NULL)
		return;

	if (dev_trimesh_ptr != NULL) {
		for (int i = 0; i < hst_scene->trimeshes.size(); i++) {
			cudaFree(dev_trimesh_ptr[i]);
		}
		cudaFree(dev_trimesh_ptr);
	}

	if (dev_tribvh_ptr != NULL) {
		for (int i = 0; i < hst_scene->tri_bvhs.size(); i++) {
			cudaFree(dev_tribvh_ptr[i]);
		}
		cudaFree(dev_tribvh_ptr);
	}

	if (dev_scene_bvh != NULL) {
		cudaFree(dev_scene_bvh);
	}

	// TODO: clean up any extra device memory you created
	int numBitmaps = hst_scene->bitmaps.size();
	if (dev_bmp_ptr != NULL && numBitmaps > 0) {
		for (int i = 0; i < hst_scene->bitmaps.size(); i++) {
			cudaFree(dev_bmp_ptr[i].pixels);
		}
		cudaFree(dev_bmp_ptr);
	}
	if (dev_texs != NULL)
		cudaFree(dev_texs);

	for (int i = 0; i < numBitmaps; i++)
	{
		cudaTextureObject_t texObj = hst_texs[i];

		// Get the cudaArray from the texture object
		cudaResourceDesc resDesc;
		cudaGetTextureObjectResourceDesc(&resDesc, texObj);
		cudaArray_t array = resDesc.res.array.array;

		// Destroy the texture object and free the cudaArray
		cudaDestroyTextureObject(texObj);
		cudaFreeArray(array);
	}
	delete[] hst_texs;
	checkCUDAError("resourceFree");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_intersections);
	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// ! implement antialiasing by jittering the ray

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		glm::vec2 bias = glm::vec2(u01(rng)-0.5f, u01(rng)-0.5f);
		//glm::vec2 bias = glm::vec2(0,0);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + bias.x)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + bias.y)
		);

		// ! Physics-based depth of field
		if (cam.aperture > 0.0f)
		{
			// Generate a random point on the lens
			glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
			glm::vec2 lensPoint = cam.aperture * sample;

			// Compute the point on the focal plane
			float focalDistance = glm::abs(cam.focalLength / segment.ray.direction.z);
			glm::vec3 focalPoint = segment.ray.origin + focalDistance * segment.ray.direction;

			// Update the ray origin
			segment.ray.origin += cam.right * lensPoint.x + cam.up * lensPoint.y;
			segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
		}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}


// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment* pathSegments,
	Geom* geoms,
	BVHNode* geomBVHs,
	Triangle** trimeshes_ptr,
	BVHNode** tribvhs_ptr,
	Intersection* intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float final_t = FLT_MAX;
		float t_min = FLT_MAX;
		bool outside = true;

		Intersection tmp_intersection;
		Intersection min_intersection;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		bool is_intersect = false;

		// naive parse through global geoms
		bool anyhit = false;


		BVHNode* stack[STACK_SIZE];
		BVHNode** stackPtr = stack;
		*stackPtr = NULL;

		int stack_size = 0;
		float t_root_max = FLT_MAX;
		float t_root_min = 0;
		if (!geomBVHs[0].bbox.intersect(pathSegment.ray, t_root_min, t_root_max)) {
			intersections[path_index].t = -1.0f;
			return;
		}

		stack_size++;
		*(++stackPtr) = &geomBVHs[0];

		while (stack_size > 0 && stack_size < STACK_SIZE) {
			BVHNode* node = *(stackPtr--);
			stack_size--;
			if (node == NULL)
				break;
			else {

				if (node->isLeaf()) {
					Ray local_ray = pathSegment.ray;
					Geom& geom = geoms[node->primId];
					local_ray.origin = multiplyMV(geom.transform.inverseTransform, glm::vec4(local_ray.origin, 1.0f));
					local_ray.direction = glm::normalize(multiplyMV(geom.transform.inverseTransform, glm::vec4(local_ray.direction, 0.0f)));

					tmp_intersection.t = -1.0f;
					is_intersect = false;

					if (geom.type == Primitive::CUBE)
					{
						is_intersect = boxIntersectionTest(local_ray, final_t, tmp_intersection, outside);
					}
					else if (geom.type == Primitive::SPHERE)
					{
						is_intersect = sphereIntersectionTest(local_ray, final_t, tmp_intersection, outside);
					}
					else if (geom.type == Primitive::TRIANGLE) {
						is_intersect = trimeshIntersectionTest(local_ray, final_t, tmp_intersection, outside, trimeshes_ptr[geom.trimeshId], tribvhs_ptr[geom.trimeshId]);
					}

					if (is_intersect) {
						intersect_point = multiplyMV(geom.transform.transform, glm::vec4(getPointOnRay(local_ray, tmp_intersection.t), 1.0f));
						normal = glm::normalize(multiplyMV(geom.transform.invTranspose, glm::vec4(tmp_intersection.surfaceNormal, 0.0f)));
						tmp_intersection.t = glm::length(intersect_point - pathSegment.ray.origin);
						tmp_intersection.surfaceNormal = normal;
						tmp_intersection.materialId = geom.materialId;

						// Compute the minimum t from the intersection tests to determine 
						// what scene geometry object was hit first.
						if (tmp_intersection.t < final_t) {
							final_t = tmp_intersection.t;
							min_intersection = tmp_intersection;
							anyhit = true;
						}
					}

				}
				else {
					float tl_min = 0;
					float tl_max = final_t;
					float tr_min = 0;
					float tr_max = final_t;

					bool hit_left = false, hit_right = false;
					if(node->leftId != -1)
						hit_left = geomBVHs[node->leftId].bbox.intersect(pathSegment.ray, tl_min, tl_max);
					if (node->rightId != -1)
						hit_right = geomBVHs[node->rightId].bbox.intersect(pathSegment.ray, tr_min, tr_max);
					if (hit_left && hit_right) {
						if (tl_min < tr_min) {
							*(++stackPtr) = &geomBVHs[node->rightId];
							stack_size++;
							*(++stackPtr) = &geomBVHs[node->leftId];
							stack_size++;
						}
						else {
							*(++stackPtr) = &geomBVHs[node->leftId];
							stack_size++;
							*(++stackPtr) = &geomBVHs[node->rightId];
							stack_size++;
						}
					}
					else if (hit_left) {
						*(++stackPtr) = &geomBVHs[node->leftId];
						stack_size++;
					}
					else if (hit_right) {
						*(++stackPtr) = &geomBVHs[node->rightId];
						stack_size++;
					}
				}
			}
		}

		if (!anyhit){
			intersections[path_index].t = -1.0f;
		} else {
			intersections[path_index] = min_intersection;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a Intersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter, 
	int num_paths, 
	Intersection* shadeableIntersections, 
	PathSegment* pathSegments, 
	Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{	
		Intersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.texture.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
				pathSegments[idx].remainingBounces = 0;
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

__global__ void shadeMaterial(
	int iter,
	int num_paths,
	Intersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	//Bitmap* bitmaps,
	cudaTextureObject_t* texObjs
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{	
		Intersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];

			Texture& texture = material.texture;
			glm::vec3 materialColor;

			if (texture.type == TextureType::BITMAP)
			{
				//materialColor = getPixel(bitmaps[texture.bitmapId], intersection.uv);
				float4 texColor = tex2D<float4>(texObjs[texture.bitmapId], intersection.uv.x, intersection.uv.y);
				materialColor = glm::vec3(texColor.x, texColor.y, texColor.z);
			}
			else {
				materialColor = texture.color;
			}


			// If the material indicates that the object was a light, "light" the ray
			if (material.type == MaterialType::LIGHT) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}else if (pathSegments[idx].remainingBounces > 0) {
				// Scatter the ray
				Sample sample = scatterRay(pathSegments[idx], intersection, material, materialColor, rng);
				float cosineterm = glm::dot(sample.ray.direction, intersection.surfaceNormal);

				pathSegments[idx].ray = sample.ray;
				pathSegments[idx].color *= sample.BSDF * cosineterm / sample.pdf;
				pathSegments[idx].remainingBounces--;
			}else {
				pathSegments[idx].color = glm::vec3(0.0f);
				pathSegments[idx].remainingBounces = 0;
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = BACKGROUND_COLOR;
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

void pathtrace(uchar4* pbo, int frame, int iter) {
	/*
	* 1.  Ray Generation
	* 2.  Intersection with Scene
	* 3.  Sample and Shading (BSDF Evaluation)
	* 4.  Stream Compaction
	* ->  Go to 2 until max depth
	* 5.  Gather results
	*/

	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	//-------------------------------------------------------------------------

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// 	 * Finally, add this iteration's results to the image. 

	// --- 1. Generating Camera Rays ---
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks: intersections info
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(Intersection));

		// --- 2. PathSegment Intersection Stage ---
		// path tracing to get the intersections with the scene
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			num_paths,
			dev_paths,
			dev_geoms,
			dev_scene_bvh,
			dev_trimesh_ptr,
			dev_tribvh_ptr,
			dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		// --- 3. Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.

		 //thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareIntersection());
		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_texs
			);
		
		// --- 4. Stream Compaction Stage ---
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, is_valid());
		num_paths = dev_path_end - dev_paths;
		iterationComplete = num_paths == 0 || depth >= traceDepth;

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// --- 5. PathSegment Final Gather Stage ---
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	//-------------------------------------------------------------------------

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
