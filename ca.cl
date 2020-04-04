inline uint index1D(uint2 i2d, uint2 size) {
    return (uint)(i2d.y * size.x + i2d.x);
}

inline bool check_neighbour_state(__global uint* domain_state, uint2 size, int2 pos, uint n) {
    uint2 i2d = (uint2) (pos.x, pos.y);
    uint next_state = (domain_state[index1D(i2d, size)] + 1) % n;
    uint2 posN = (uint2)(pos.x, (pos.y - 1 < 0) ? size.y : pos.y - 1);
    uint2 posS = (uint2)(pos.x, (pos.y + 1 >= size.y) ? 0 : pos.y + 1);
    uint2 posW = (uint2)((pos.x - 1 < 0 ) ? size.x : pos.x - 1, pos.y);
    uint2 posE = (uint2)((pos.x + 1 >= size.x) ? 0 : pos.x + 1, pos.y);
    
    uint north = index1D(posN, size);
    uint south = index1D(posS, size);
    uint west = index1D(posW, size);
    uint east = index1D(posE, size);

    return (domain_state[north] == next_state) ||
        (domain_state[south] == next_state) ||
        (domain_state[east] == next_state) ||
        (domain_state[west] == next_state);
}

__kernel void parity_cellular_automata(__global uint *domain_state_r, __global uint *domain_state_w, uint2 size, uint n) {
    
}

__kernel void cyclic_cellular_automata(__global uint *domain_state_r, __global uint *domain_state_w, uint2 size, uint n) {
    uint2 i2d = (uint2)(get_global_id(0), get_global_id(1));
    if (check_neighbour_state(domain_state_r, size, (int2)(i2d.x, i2d.y), n)) {
        uint pos = index1D(i2d, size);
        domain_state_w[pos] = (domain_state_r[pos] + 1) % n;
    }
}