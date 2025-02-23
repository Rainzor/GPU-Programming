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
#include "intersections.h"
#include "sampler.h"
#include "../utilities.h"


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
//static Geom* dev_geoms = NULL;
static GeomGPU* dev_geoms = NULL;
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

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	if (!scene->trimeshes.empty()) {
		//cudaMalloc((void**)&dev_trimeshes, scene->trimeshes.size() * sizeof(TriangleMesh));
		cudaMalloc((void**)&dev_trimesh_ptr, scene->trimeshes.size() * sizeof(Triangle*));
		for (int i = 0; i < scene->trimeshes.size(); i++)
		{
			int numTriangles = scene->trimeshes[i].num;
			Triangle* host_trimesh_ptr = NULL;
			cudaMalloc(&(host_trimesh_ptr), numTriangles * sizeof(Triangle));
			cudaMemcpy(host_trimesh_ptr, scene->trimeshes[i].triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

			cudaMemcpy((void**)&(dev_trimesh_ptr[i]), &host_trimesh_ptr, sizeof(unsigned char*), cudaMemcpyHostToDevice);
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
			BVHNode* host_tribvh_ptr = NULL;
			cudaMalloc((void**)&host_tribvh_ptr, numNodes * sizeof(BVHNode));
			cudaMemcpy(host_tribvh_ptr, scene->tri_bvhs[i].bvh_nodes.data(), numNodes * sizeof(BVHNode), cudaMemcpyHostToDevice);

			cudaMemcpy((void**)&(dev_tribvh_ptr[i]), &host_tribvh_ptr, sizeof(BVHNode*), cudaMemcpyHostToDevice);
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

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(GeomGPU));
	for (int i = 0; i < scene->geoms.size(); i++) {
		cudaMemcpy(&dev_geoms[i].type, &scene->geoms[i].type, sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(&dev_geoms[i].transform, &scene->geoms[i].transform, sizeof(Transform), cudaMemcpyHostToDevice);
		//cudaMemcpy((void**)&dev_geoms[i].dev_material, &dev_materials[scene->geoms[i].materialId], sizeof(Material*), cudaMemcpyHostToDevice);
		cudaMemcpy(&dev_geoms[i].materialId, &scene->geoms[i].materialId, sizeof(int), cudaMemcpyHostToDevice);

		if (scene->geoms[i].type == TRIANGLE) {
			cudaMemcpy(&dev_geoms[i].dev_triangles, &dev_trimesh_ptr[scene->geoms[i].trimeshId], sizeof(Triangle*), cudaMemcpyDeviceToDevice);
			cudaMemcpy(&dev_geoms[i].dev_bvh_nodes, &dev_tribvh_ptr[scene->geoms[i].trimeshId], sizeof(BVHNode*), cudaMemcpyDeviceToDevice);
		}
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
		PathSegment& new_path = pathSegments[index];

		new_path.ray.origin = cam.position;
		new_path.color = glm::vec3(0.0f, 0.0f, 0.0f);
		new_path.throughput = glm::vec3(1.0f, 1.0f, 1.0f);

		// ! implement antialiasing by jittering the ray

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		glm::vec2 bias = glm::vec2(u01(rng)-0.5f, u01(rng)-0.5f);
		//glm::vec2 bias = glm::vec2(0,0);
		new_path.ray.direction = glm::normalize(cam.view
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
			float focalDistance = glm::abs(cam.focalLength / new_path.ray.direction.z);
			glm::vec3 focalPoint = new_path.ray.origin + focalDistance * new_path.ray.direction;

			// Update the ray origin
			new_path.ray.origin += cam.right * lensPoint.x + cam.up * lensPoint.y;
			new_path.ray.direction = glm::normalize(focalPoint - new_path.ray.origin);
		}

		new_path.pixelIndex = index;
		new_path.remainingBounces = traceDepth;
	}
}


// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment* pathSegments,
	GeomGPU* geoms,
	BVHNode* geomBVHs,
	//Triangle** trimeshes_ptr,
	//BVHNode** tribvhs_ptr,
	Intersection* intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		Intersection test_intersection;
		bool outside;
		bool anyhit = worldIntersectionTest(
			pathSegments[path_index].ray,
			FLT_MAX,
			test_intersection,
			geoms,
			geomBVHs);
							//trimeshes_ptr, 
							//tribvhs_ptr);

		if (!anyhit){
			intersections[path_index].t = -1.0f;
		} else {
			intersections[path_index] = test_intersection;
		}
	}
}

/**
* Compute the color of the ray after intersection with the scene.
* It is like the "shader" function in OpenGL
*/
__global__ void shadeMaterialMIS(
	int iter,
	int num_paths,
	Intersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	cudaTextureObject_t* texObjs
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{	
		Intersection intersection = shadeableIntersections[idx];
		PathSegment& cur_path = pathSegments[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material* material = &materials[intersection.materialId];

			Texture& texture = material->texture;
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
			if (material->type == MaterialType::LIGHT) {
				cur_path.color += cur_path.throughput * materialColor * material->emittance;
				cur_path.remainingBounces = 0;// terminate the path

			}else if (cur_path.remainingBounces > 0) {
				// Scatter the ray
				Sample sample = scatterRay(cur_path, intersection, material, materialColor, rng);
				float cosineterm = glm::dot(sample.ray.direction, intersection.surfaceNormal);

				cur_path.ray = sample.ray;
				cur_path.throughput *= sample.BSDF * cosineterm / sample.pdf;
				cur_path.remainingBounces--;
			}
		}
		else {// If there was no intersection, return background color
			cur_path.color += BACKGROUND_COLOR * cur_path.throughput;
			cur_path.remainingBounces = 0;
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
			//dev_trimesh_ptr,
			//dev_tribvh_ptr,
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
		shadeMaterialMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
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
