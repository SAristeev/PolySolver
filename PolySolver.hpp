#pragma once
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include "external/json/json.hpp"
#include <fstream>
#include "Solvers/implementation.hpp"
#include "Solvers/pardiso.hpp"

using namespace nlohmann;

void CreateSolvers(const json& settings, std::set<std::unique_ptr<LinearSolver>>& solvers);