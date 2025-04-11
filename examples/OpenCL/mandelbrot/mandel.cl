// Taken from https://github.com/JuliaGPU/OpenCL.jl/blob/master/examples/notebooks/mandelbrot_fractal.ipynb

// codesnippet
__kernel void mandelbrot(
  __global float2 *q,
  __global ushort *output,
  ushort const maxit) {
  int gid = get_global_id(0), it;
  float tmp, r = 0, i = 0;
  output[gid] = 0;
  for(it = 0; it < maxit; it++) {
    tmp = r*r - i*i + q[gid].x;
    i = 2*r*i + q[gid].y;
    r = tmp;
    if (r*r + i*i > 4.0f)
        output[gid] = it;
// codesnippet
  }
}
