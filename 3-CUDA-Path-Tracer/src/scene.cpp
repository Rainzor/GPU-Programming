#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>


Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;

	this->workdir = filename.substr(0, filename.find_last_of("\\") + 1);
	cout << "Working directory: " << this->workdir << endl;

    string type = filename.substr(filename.find_last_of(".") + 1);

    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }

	if (strcmp(type.c_str(), "json") == 0) {
        json sceneData;
        fp_in >> sceneData;
        this->state.imageName = sceneData["name"];
        cout << "Scene Name: " << this->state.imageName << endl;
        this->state.traceDepth = sceneData["integrator"]["maxdepth"];
        loadCamera(sceneData["sensor"]);
        for (const auto& object : sceneData["shape"]){
            loadGeom(object);
        }
        for (const auto& material : sceneData["bsdf"]){
            loadMaterial(material);
        }
    }
    fp_in.close();
}

Scene::~Scene() {
	cout << "Cleaning up scene..." << endl;
	for (int i = 0; i < bitmaps.size(); i++) {
		delete[] bitmaps[i].pixels;
	}
}


int Scene::loadCamera(const json& cameraData) {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;

    const json filmData = cameraData["film"];
    camera.resolution = glm::ivec2(filmData["resolution"][0], filmData["resolution"][1]);

    state.iterations = filmData["spp"];

    camera.position = glm::vec3(cameraData["eye"][0], cameraData["eye"][1], cameraData["eye"][2]);
    camera.lookAt = glm::vec3(cameraData["lookat"][0], cameraData["lookat"][1], cameraData["lookat"][2]);
    camera.up = glm::vec3(cameraData["up"][0], cameraData["up"][1], cameraData["up"][2]);

    camera.focalLength = cameraData["focal"];
    camera.aperture = cameraData["aperture"];
    
    // Calculate fov based on resolution
    float fovy = cameraData["fovy"];
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    // Set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadGeom(const json& shapeData) {
    Geom newGeom;
    
    string type = shapeData["type"];
    if (type == "sphere") {
        cout << "Creating new sphere..." << endl;
        newGeom.type = Primitive::SPHERE;
    } else if (type == "cube") {
        cout << "Creating new cube..." << endl;
        newGeom.type = Primitive::CUBE;
    } else if (type == "triangle") {
        cout << "Creating new triangle..." << endl;
        newGeom.type = Primitive::TRIANGLE;
    } else {
        cout << "Unknown shape type: " << type << endl;
        return -1;
    }

    // Link material (bsdf)
    newGeom.materialId = shapeData["bsdf"];
    cout << "Connecting Geom to Material " << newGeom.materialId << "..." << endl;

    // Load transformations
    auto transform = shapeData["transform"];
    newGeom.translation = glm::vec3(transform["translate"][0], transform["translate"][1], transform["translate"][2]);
    newGeom.rotation = glm::vec3(transform["rotate"][0], transform["rotate"][1], transform["rotate"][2]);
    newGeom.scale = glm::vec3(transform["scale"][0], transform["scale"][1], transform["scale"][2]);

    newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
    newGeom.inverseTransform = glm::inverse(newGeom.transform);
    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    geoms.push_back(newGeom);
    return 1;
}

int Scene::loadMaterial(const json& materialData) {
    Material newMaterial;
    string type = materialData["type"];
    
    if (type == "light") {
        newMaterial.type = MaterialType::LIGHT;
    } else if (type == "diffuse") {
        newMaterial.type = MaterialType::DIFFUSE;
    } else if (type == "specular") {
        newMaterial.type = MaterialType::SPECULAR;
    } else if (type == "dielectric") {
        newMaterial.type = MaterialType::DIELECTRIC;
    }

    // Load color and other properties
    if (materialData.contains("rgb")) {
		newMaterial.texture.color = glm::vec3(materialData["rgb"][0], materialData["rgb"][1], materialData["rgb"][2]);
	} else if (materialData.contains("bitmap")) {
		newMaterial.texture.color = glm::vec3(1.0f);
		newMaterial.texture.type = TextureType::BITMAP;
        newMaterial.texture.bitmapId = bitmaps.size();
		string bitmapPath = workdir + string(materialData["bitmap"]);
		cout << "Loading bitmap texture from " << bitmapPath << "..." << endl;
		int w, h, n;
		unsigned char* data = stbi_load(bitmapPath.c_str(), &w, &h, &n, 0);
        if (data == nullptr) {
            cout << "Error loading bitmap texture!" << endl;
            return -1;
        }
		glm::u8vec4* dataCopy = new glm::u8vec4[w * h];
		cout << "Width: " << w << " Height: " << h << " Channels: " << n << endl;
		for (int i = 0; i < w * h; i++) {
			unsigned char r = data[i * n];
			unsigned char g = data[i * n + 1];
			unsigned char b = data[i * n + 2];
			unsigned char a = n == 4 ? data[i * n + 3] : 255;
			dataCopy[i] = glm::u8vec4(r, g, b, a);
		}
		Bitmap newBitmap;
		newBitmap.width = w;
		newBitmap.height = h;
		newBitmap.pixels = dataCopy;
		bitmaps.push_back(newBitmap);
        stbi_image_free(data);

    }
    if (materialData.contains("emission")) {
        newMaterial.emittance = materialData["emission"];
    }
    if (materialData.contains("ior")) {
        newMaterial.indexOfRefraction = materialData["indexOfRefraction"];
    }

    materials.push_back(newMaterial);
    return 1;
}