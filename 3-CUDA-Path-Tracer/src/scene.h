#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <stb_image.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_obj_loader.h"

using namespace std;
using json = nlohmann::json;

class Scene {
private:
	string workdir;
    ifstream fp_in;
    int loadBitmap(const string& bitmapPath);
    int loadMaterial(const json& materialData);
    int loadGeom(const json& geomData);
    int loadObj(const string& obj_file,const Transform& transform, bool usemtl = false);
    int loadCamera(const json& cameraData);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Bitmap> bitmaps;
	std::vector<TriangleMesh> trimeshes;
    RenderState state;
};
