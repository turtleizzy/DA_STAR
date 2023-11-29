#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <chrono>

#include "include/graph.h"
#include "include/diff_graph.h"
#include "include/dynamic_graph.h"
#include "include/a_star.h"

#ifdef __NVCC__
#include <cuda.h>
#include "include/graph.cuh"
#include "include/a_star.cuh"
#endif

using WeightType = float;

void read_solver_info(std::string solver_filename, unsigned int &start, std::vector<unsigned int> &ends, unsigned int *shape)
{
    std::FILE *infile = std::fopen(solver_filename.c_str(), "rb");
    if (!infile)
    {
        std::cout << "[ERROR] Couldn't open solver info file\n";
        exit(1);
    }
    std::fread(&start, sizeof(start), 1, infile);
    unsigned int n_ends;
    std::fread(&n_ends, sizeof(n_ends), 1, infile);
    std::cout << n_ends << std::endl;

    for (int i = 0; i < n_ends; i++)
    {
        unsigned int end;
        std::fread(&end, sizeof(end), 1, infile);
        ends.push_back(end);
    }
    std::fread(shape, sizeof(shape[0]), 3, infile);
    std::fclose(infile);
}

inline unsigned int ravel_index(unsigned int i, unsigned int j, unsigned int k, unsigned int *shape)
{
    return i * shape[1] * shape[2] + j * shape[2] + k;
}

inline void unravel_index(unsigned int idx, unsigned int *shape, unsigned int &i, unsigned int &j, unsigned int &k)
{
    unsigned int t;
    i = idx / (shape[1] * shape[2]);
    t = idx % (shape[1] * shape[2]);
    j = t / shape[2];
    k = t % shape[2];
}

int main()
{
    Graph<float> graph;
    std::string filename = "/tmp/a.bin";
    graph.read_graph_binary(filename);

    Dynamic_Graph<WeightType> dgraph(&graph);
    unsigned int start;
    std::vector<unsigned int> ends;
    unsigned int *shape = (unsigned int *)malloc(sizeof(unsigned int *) * 3);
    read_solver_info("/tmp/a_solver.bin", start, ends, shape);
    std::cout << start << std::endl;

    WeightType *hx = (WeightType *)malloc(sizeof(WeightType) * graph.get_num_nodes());
    memset(hx, 0, sizeof(WeightType) * graph.get_num_nodes());
    bool first_iter = 1;
    // for (std::vector<unsigned int>::const_iterator target = ends.begin(); target != ends.end(); target++)
    // {
    //     unsigned int t_i, t_j, t_k;
    //     unravel_index(*target, shape, t_i, t_j, t_k);
    //     std::cout << t_i << "," << t_j << "," << t_k << std::endl;
    //     for (int i = 0; i < shape[0]; i++)
    //         for (int j = 0; j < shape[1]; j++)
    //             for (int k = 0; k < shape[2]; k++)
    //             {
    //                 float val = std::sqrt((i - t_i) * (i - t_i) + (j - t_j) * (j - t_j) + (k - t_k) * (k - t_k));
    //                 if (first_iter)
    //                     hx[ravel_index(i, j, k, shape)] = val;
    //                 else
    //                     hx[ravel_index(i, j, k, shape)] = hx[ravel_index(i, j, k, shape)] > val ? val : hx[ravel_index(i, j, k, shape)];
    //             };
    //     first_iter = 0;
    //     break;
    // };

#ifdef __NVCC__
    auto stTime = std::chrono::system_clock::now();
    GPU_Graph<WeightType> g(&graph);
    printf("allocated\n");
    GPU_Dynamic_Graph<WeightType> dg(&g);

    // for (std::vector<unsigned int>::const_iterator target = ends.begin(); target != ends.end(); target++)
    // {

    GPU_A_Star<WeightType, WeightType> g_algo(&dg, start, ends[0], 32 * 1024);
    g_algo.set_heuristics(hx);

    int N = g.get_num_nodes();
    int *parent_array = (int *)malloc(sizeof(int) * N);
    WeightType *cost_array = (WeightType *)malloc(sizeof(WeightType) * N);

    g_algo.get_path(cost_array, parent_array);
    auto edTime = std::chrono::system_clock::now();
    std::chrono::duration<float, std::ratio<1, 1>> duration(edTime - stTime);
    std::cout << start << " " << graph.offsets[start] << " " << graph.offsets[start + 1] << std::endl;
    std::cout << ends[0] << " " << cost_array[ends[0]] << " " << parent_array[ends[0]] << std::endl;
    std::cout << duration.count() << std::endl;
    g_algo.free_gpu();
    g_algo.free_memory();
#endif

    return 0;
}