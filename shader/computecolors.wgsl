// 3D RGB Colorspace Projection with Quaternion Rotations
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct Params {
    rotation_speed: f32, intensity: f32,
    rot_x: f32, rot_y: f32, rot_z: f32, rot_w: f32,
    scale: f32, _padding: u32,
}
@group(1) @binding(0) var<uniform> params: Params;
@group(2) @binding(0) var input_texture: texture_2d<f32>;
@group(2) @binding(1) var tex_sampler: sampler;
@group(2) @binding(2) var output: texture_storage_2d<rgba16float, write>;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<i32>>;

const PI = 3.14159;

// Quaternion structure for rotations
struct Quaternion {
    w: f32,  // Real part
    x: f32,  // i
    y: f32,  // j
    z: f32,  // k
}

fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> Quaternion {
    let half_angle = angle * .5;
    let s = sin(half_angle);
    let axis_n = normalize(axis);
    
    var q: Quaternion;
    q.w = cos(half_angle);
    q.x = axis_n.x * s;
    q.y = axis_n.y * s;
    q.z = axis_n.z * s;
    
    return q;
}

fn quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion {
    var result: Quaternion;
    
    result.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
    result.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
    result.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
    result.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
    return result;
}

fn quat_rotate_vector(q: Quaternion, v: vec3<f32>) -> vec3<f32> {
    let v_quat = Quaternion(0., v.x, v.y, v.z);
    let q_conj = Quaternion(q.w, -q.x, -q.y, -q.z);
    // Perform the rotation: q * v * q^-1
    let result = quat_multiply(quat_multiply(q, v_quat), q_conj);
    // Return the vector part of the result
    return vec3<f32>(result.x, result.y, result.z);
}

fn rot(p: vec3<f32>) -> vec3<f32> {
    let t = time_data.time * params.rotation_speed;
    
    // Create time-varying axes for a completely unique rotation behavior
    let axis1 = normalize(vec3<f32>(
        sin(t * 0.3) + params.rot_x,
        cos(t * 0.4) + params.rot_y,
        sin(t * 0.5 + cos(t * 0.2)) + params.rot_z
    ));
    
    let axis2 = normalize(vec3<f32>(
        cos(t * 0.6),
        sin(t * 0.7),
        cos(t * 0.8)
    ));
    
    let q1 = quat_from_axis_angle(axis1, t * 0.9);
    let q2 = quat_from_axis_angle(axis2, t * 1.1);
    let q_combined = quat_multiply(q1, q2);
    return quat_rotate_vector(q_combined, p);
}

// Convert to screen coordinates with our unique rotation
fn toScreen(color: vec3<f32>, scr: vec2<f32>) -> vec2<f32> {
    let centered = color - .5;
    
    let rotated = rot(centered);
    
    return (rotated.xy + .5) * scr * vec2<f32>(scr.y/scr.x, 1.) * params.scale;
}

@compute @workgroup_size(16, 16, 1)
fn clear_buffer(@builtin(global_invocation_id) id: vec3<u32>) {
    let scr = textureDimensions(output);
    
    if (id.x >= scr.x || id.y >= scr.y) { return; }
    
    let i = id.y * scr.x + id.x;
    atomicStore(&atomic_buffer[i*4],   0);
    atomicStore(&atomic_buffer[i*4+1], 0);
    atomicStore(&atomic_buffer[i*4+2], 0);
    atomicStore(&atomic_buffer[i*4+3], 0);
}

// Project colors
@compute @workgroup_size(16, 16, 1)
fn project_colors(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    let scr = textureDimensions(output);
    
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2<f32>(id.xy) + .5) / vec2<f32>(dims);
    let color = textureSampleLevel(input_texture, tex_sampler, uv, 0.).xyz;
    
    if (length(color) < .05) { return; }
    
    let pos = sqrt(color);
    
    let scr_pos = toScreen(pos, vec2<f32>(scr));
    
    let x = min(max(0, i32(scr_pos.x)), i32(scr.x) - 1);
    let y = min(max(0, i32(scr_pos.y)), i32(scr.y) - 1);
    
    let idx = y * i32(scr.x) + x;
    atomicAdd(&atomic_buffer[idx*4],   i32(256. * color.x));
    atomicAdd(&atomic_buffer[idx*4+1], i32(256. * color.y));
    atomicAdd(&atomic_buffer[idx*4+2], i32(256. * color.z));
    atomicAdd(&atomic_buffer[idx*4+3], 1);
}

@compute @workgroup_size(16, 16, 1)
fn generate_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let scr = textureDimensions(output);
    
    if (id.x >= scr.x || id.y >= scr.y) { return; }
    
    let idx = i32(id.y * scr.x + id.x);
    let cnt = atomicLoad(&atomic_buffer[idx*4+3]);
    
    if (cnt > 0) {
        let r = f32(atomicLoad(&atomic_buffer[idx*4])) / (f32(cnt) * 256.);
        let g = f32(atomicLoad(&atomic_buffer[idx*4+1])) / (f32(cnt) * 256.);
        let b = f32(atomicLoad(&atomic_buffer[idx*4+2])) / (f32(cnt) * 256.);
        
        textureStore(output, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(r,g,b) * params.intensity, 1.));
    } else {
        let uv = (vec2<f32>(id.xy) + .5) / vec2<f32>(scr);
        let bg_color = textureSampleLevel(input_texture, tex_sampler, uv, 0.).xyz * params.rot_w;
        textureStore(output, vec2<i32>(id.xy), vec4<f32>(bg_color, 1.));    
    }
}