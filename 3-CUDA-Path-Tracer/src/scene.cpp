#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>


Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;

    string type = filename.substr(filename.find_last_of(".") + 1);

    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }

	if (strcmp(type.c_str(), "txt") == 0) {
        while (fp_in.good()) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            if (!line.empty()) {
                vector<string> tokens = utilityCore::tokenizeString(line);
                if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                    loadMaterial(tokens[1]);
                    cout << " " << endl;
                } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                    loadGeom(tokens[1]);
                    cout << " " << endl;
                } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                    loadCamera();
                    cout << " " << endl;
                }
            }
        }
	}
	else if (strcmp(type.c_str(), "json") == 0) {
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
    newGeom.materialid = shapeData["bsdf"];
    cout << "Connecting Geom to Material " << newGeom.materialid << "..." << endl;

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
    newMaterial.color = glm::vec3(materialData["rgb"][0], materialData["rgb"][1], materialData["rgb"][2]);
    if (materialData.contains("emission")) {
        newMaterial.emittance = materialData["emission"];
    }
    if (materialData.contains("ior")) {
        newMaterial.indexOfRefraction = materialData["indexOfRefraction"];
    }

    materials.push_back(newMaterial);
    return 1;
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "FOCAL") == 0) {
            camera.focalLength = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "APERTURE") == 0) {
			camera.aperture = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FAR_PLANE") == 0) {
		    camera.farClip = atof(tokens[1].c_str());
		} else if (strcmp(tokens[0].c_str(), "NEAR_PLANE") == 0) {
			camera.nearClip = atof(tokens[1].c_str());
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}



int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = Primitive::SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = Primitive::CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}


int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;
        string line;
        //load material type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "light") == 0) {
                newMaterial.type = MaterialType::LIGHT;
            } else if (strcmp(line.c_str(), "diffuse") == 0) {
                newMaterial.type = MaterialType::DIFFUSE;
            } else if (strcmp(line.c_str(), "specular") == 0) {
                newMaterial.type = MaterialType::SPECULAR;
            } else if (strcmp(line.c_str(), "dielectric") == 0) {
                newMaterial.type = MaterialType::DIELECTRIC;
            }
        }

        //load static properties
        for (int i = 0; i < 3; i++) {
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
