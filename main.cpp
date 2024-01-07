#include "PolySolver.hpp"

int main(int argc, char** argv)
{
    std::unordered_map<std::string, std::string> parsed_params;//in the pair {key,param} param may be empty

    for (int pos = 1; pos < argc; ++pos) {
        if (argv[pos][0] == '-') {//key is found
            std::string key(argv[pos]);

            if (pos + 1 < argc && argv[pos + 1][0] != '-') {//value is found
                std::string value(argv[pos + 1]);

                parsed_params.insert(std::make_pair(key, value));
                ++pos;
                continue;
            }
            parsed_params.insert(std::make_pair(key, std::string()));
        }
    }

    json settings;
    {
        const auto it = parsed_params.find("--input");
        if (it != parsed_params.end() && !it->second.empty()) {
            std::string settingsFileName = it->second; // read file name
            parsed_params.erase(it); // delete
            std::ifstream settingFile(settingsFileName, std::ios::in);
            if (!settingFile) {
                throw std::runtime_error("cannot open setting file: " + settingsFileName);
            }
            settings = json::parse(settingFile);
            settingFile.close();
        }
        else {
            printf("Error: setting file not found!\n");
        }
    }

    // all Derived solvers by ptr to Base Solver
    std::set<std::unique_ptr<LinearSolver>> solvers;
    CreateSolvers(settings, solvers);

    for (std::string case_name : settings["LinearProblem"]["cases"]) {
        std::string path = settings["LinearProblem"]["path"];
        std::string full_path = path + '/' + case_name;

        SPARSE::SparseMatrix<MKL_INT, double> A;
        SPARSE::SparseVector<double> b;
        SPARSE::SparseVector<double> x;
        
        A.freadCSR(full_path + "/A.txt");
        b.fread(full_path + "/B.vec");
        x = b;
        
        for (auto& solve_ptr : solvers) {
            solve_ptr->Solve(A,b,x);
        }
    }
    return 0;
}