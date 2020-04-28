// Includes tricubic interpolation code with the following licence:

/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2009, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.
When using this code in a scientific project, please cite one or all of the
following papers:
*  Daniel Ruijters and Philippe Thévenaz,
   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
   The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
*  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
   Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
\*--------------------------------------------------------------------------*/

var vertShader =
`#version 300 es
#line 4
layout(location=0) in vec3 pos;
uniform mat4 proj_view;
uniform vec3 eye_pos;
uniform vec3 volume_scale;

out vec3 vray_dir;
flat out vec3 transformed_eye;

void main(void) {
	// TODO: For non-uniform size volumes we need to transform them differently as well
	// to center them properly
	vec3 volume_translation = vec3(0.5) - volume_scale * 0.5;
	gl_Position = proj_view * vec4(pos * volume_scale + volume_translation, 1);
	transformed_eye = (eye_pos - volume_translation) / volume_scale;
	vray_dir = pos - transformed_eye;
}`;

var fragShader =
`#version 300 es
#line 24
precision highp int;
precision highp float;
uniform highp sampler3D volume;
uniform highp sampler2D colormap;
uniform ivec3 volume_dims;
uniform float dt_scale;

in vec3 vray_dir;
flat in vec3 transformed_eye;
out vec4 color;

vec2 intersect_box(vec3 orig, vec3 dir) {
	const vec3 box_min = vec3(0);
	const vec3 box_max = vec3(1);
	vec3 inv_dir = 1.0 / dir;
	vec3 tmin_tmp = (box_min - orig) * inv_dir;
	vec3 tmax_tmp = (box_max - orig) * inv_dir;
	vec3 tmin = min(tmin_tmp, tmax_tmp);
	vec3 tmax = max(tmin_tmp, tmax_tmp);
	float t0 = max(tmin.x, max(tmin.y, tmin.z));
	float t1 = min(tmax.x, min(tmax.y, tmax.z));
	return vec2(t0, t1);
}

// Pseudo-random number gen from
// http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
// with some tweaks for the range of values
float wang_hash(int seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return float(seed % 2147483647) / float(2147483647);
}

float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}

float interpolate_tricubic_fast(sampler3D tex, vec3 coord)
{
        // shift the coordinate from [0,1] to [-0.5, nrOfVoxels-0.5]
        vec3 nrOfVoxels = vec3(textureSize(tex, 0));
        vec3 coord_grid = coord * nrOfVoxels - 0.5;
        vec3 index = floor(coord_grid);
        vec3 fraction = coord_grid - index;
        vec3 one_frac = 1.0 - fraction;

        vec3 w0 = 1.0/6.0 * one_frac*one_frac*one_frac;
        vec3 w1 = 2.0/3.0 - 0.5 * fraction*fraction*(2.0-fraction);
        vec3 w2 = 2.0/3.0 - 0.5 * one_frac*one_frac*(2.0-one_frac);
        vec3 w3 = 1.0/6.0 * fraction*fraction*fraction;

        vec3 g0 = w0 + w1;
        vec3 g1 = w2 + w3;
        vec3 mult = 1.0 / nrOfVoxels;
        vec3 h0 = mult * ((w1 / g0) - 0.5 + index);  //h0 = w1/g0 - 1, move from [-0.5, nrOfVoxels-0.5] to [0,1]
        vec3 h1 = mult * ((w3 / g1) + 1.5 + index);  //h1 = w3/g1 + 1, move from [-0.5, nrOfVoxels-0.5] to [0,1]

        // fetch the eight linear interpolations
        // weighting and fetching is interleaved for performance and stability reasons
        float tex000 = texture(tex, h0).r;
        float tex100 = texture(tex, vec3(h1.x, h0.y, h0.z)).r;
        tex000 = mix(tex100, tex000, g0.x);  //weigh along the x-direction
        float tex010 = texture(tex, vec3(h0.x, h1.y, h0.z)).r;
        float tex110 = texture(tex, vec3(h1.x, h1.y, h0.z)).r;
        tex010 = mix(tex110, tex010, g0.x);  //weigh along the x-direction
        tex000 = mix(tex010, tex000, g0.y);  //weigh along the y-direction
        float tex001 = texture(tex, vec3(h0.x, h0.y, h1.z)).r;
        float tex101 = texture(tex, vec3(h1.x, h0.y, h1.z)).r;
        tex001 = mix(tex101, tex001, g0.x);  //weigh along the x-direction
        float tex011 = texture(tex, vec3(h0.x, h1.y, h1.z)).r;
        float tex111 = texture(tex, h1).r;
        tex011 = mix(tex111, tex011, g0.x);  //weigh along the x-direction
        tex001 = mix(tex011, tex001, g0.y);  //weigh along the y-direction

        return mix(tex001, tex000, g0.z);  //weigh along the z-direction
}

/* central difference */
vec3 gradient(in sampler3D s, vec3 p, float dt)
{
        vec2 e = vec2(dt, 0.0);

        return vec3(interpolate_tricubic_fast(s, p - e.xyy) - interpolate_tricubic_fast(s, p + e.xyy),
                interpolate_tricubic_fast(s, p - e.yxy) - interpolate_tricubic_fast(s, p + e.yxy),
                interpolate_tricubic_fast(s, p - e.yyx) - interpolate_tricubic_fast(s, p + e.yyx));
}

void main(void) {
        vec3 ray_dir = normalize(vray_dir);
        vec2 t_hit = intersect_box(transformed_eye, ray_dir);
        if (t_hit.x > t_hit.y) {
                discard;
        }
        t_hit.x = max(t_hit.x, 0.0);
        vec3 dt_vec = 0.5 / (vec3(volume_dims) * abs(ray_dir));
        float dt = dt_scale * min(dt_vec.x, min(dt_vec.y, dt_vec.z));
        vec3 p = transformed_eye + t_hit.x * ray_dir;

        float prev_val = 0.0;
        for (float t = t_hit.x; t < t_hit.y; t += dt) {
                float val = interpolate_tricubic_fast(volume, p);
                vec4 val_color = vec4(0.0,0.0,0.0,0.0); //vec4(texture(colormap, vec2(val, 0.5)).rgb, val);
                // Opacity correction
                val_color.a = 1.0 - pow(1.0 - val_color.a, dt_scale);
                color.rgb += (1.0 - color.a) * val_color.a * val_color.rgb;
                color.a += (1.0 - color.a) * val_color.a;
                if (color.a >= 0.95) {
                        break;
                }

                if (sign(val - 0.10) != sign(prev_val - 0.10)) {
                        vec3 prev_p = p - ray_dir * dt;
                        float a = (0.10 - prev_val)/(val - prev_val);
                        vec3 inter_p = (1.0 - a)*(p - dt*ray_dir) + a*p;
                        color = vec4(gradient(volume, inter_p, 0.02), 1.0);
                }

                prev_val = val;
                p += ray_dir * dt;
        }
    color.r = linear_to_srgb(color.r);
    color.g = linear_to_srgb(color.g);
    color.b = linear_to_srgb(color.b);
}`;

