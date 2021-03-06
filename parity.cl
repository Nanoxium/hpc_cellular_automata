inline uint index1D(uint2 i2d, uint2 size) {
    return (uint)(i2d.y * size.x + i2d.x);
}

inline uint get_state(__global uint* domain_state, uint2 size, int2 pos) {
    uint2 posN = (uint2)(pos.x, (pos.y - 1 < 0) ? size.y - 1 : pos.y - 1);
    uint2 posS = (uint2)(pos.x, (pos.y + 1 >= size.y) ? 0 : pos.y + 1);
    uint2 posW = (uint2)((pos.x - 1 < 0 ) ? size.x - 1 : pos.x - 1, pos.y);
    uint2 posE = (uint2)((pos.x + 1 >= size.x) ? 0 : pos.x + 1, pos.y);
    
    uint north = index1D(posN, size);
    uint south = index1D(posS, size);
    uint west = index1D(posW, size);
    uint east = index1D(posE, size);

    return domain_state[north] ^ domain_state[south] ^ domain_state[east] ^ domain_state[west];
}

__kernel void cellular_automaton(__global uint* domain_state_r, __global uint* domain_state_w, uint2 size, uint n) {
    uint2 i2d = (uint2)(get_global_id(0), get_global_id(1));
    uint pos = index1D(i2d, size);
    domain_state_w[pos] = get_state(domain_state_r, size, (int2)(i2d.x, i2d.y));
}