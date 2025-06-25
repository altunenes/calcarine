// Calcarine: Simple Texture Processing Shader
// Basic image adjustments: brightness, contrast, saturation, etc.

struct TimeUniform { 
    time: f32, 
    delta: f32, 
    frame: u32, 
    _padding: u32 
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct Params {
    brightness: f32,    // -1.0 to 1.0 (brightness adjustment)
    contrast: f32,      // 0.0 to 2.0 (contrast multiplier) 
    saturation: f32,    // 0.0 to 2.0 (saturation multiplier)
    hue_shift: f32,     // 0.0 to 2π (hue rotation in radians)
    gamma: f32,         // 0.1 to 3.0 (gamma correction)
    vignette: f32,      // 0.0 to 1.0 (vignette strength)
    noise: f32,         // 0.0 to 1.0 (noise amount)
    _padding: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

@group(2) @binding(0) var input_texture: texture_2d<f32>;
@group(2) @binding(1) var tex_sampler: sampler;
@group(2) @binding(2) var output: texture_storage_2d<rgba16float, write>;

const PI = 3.14159265359;

// Convert RGB to HSV
fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    let delta = max_val - min_val;
    
    var hue = 0.0;
    let saturation = select(0.0, delta / max_val, max_val > 0.0);
    let value = max_val;
    
    if (delta > 0.0) {
        if (max_val == rgb.r) {
            hue = ((rgb.g - rgb.b) / delta) % 6.0;
        } else if (max_val == rgb.g) {
            hue = (rgb.b - rgb.r) / delta + 2.0;
        } else {
            hue = (rgb.r - rgb.g) / delta + 4.0;
        }
        hue = hue * 60.0;
        if (hue < 0.0) { hue += 360.0; }
    }
    
    return vec3<f32>(hue, saturation, value);
}

// Convert HSV to RGB
fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let hue = hsv.x;
    let saturation = hsv.y;
    let value = hsv.z;
    
    let c = value * saturation;
    let x = c * (1.0 - abs((hue / 60.0) % 2.0 - 1.0));
    let m = value - c;
    
    var rgb = vec3<f32>(0.0);
    
    if (hue >= 0.0 && hue < 60.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (hue >= 60.0 && hue < 120.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (hue >= 120.0 && hue < 180.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (hue >= 180.0 && hue < 240.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (hue >= 240.0 && hue < 300.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else if (hue >= 300.0 && hue < 360.0) {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m);
}

// Simple noise function
fn noise(coord: vec2<f32>) -> f32 {
    return fract(sin(dot(coord, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Main texture processing
@compute @workgroup_size(16, 16, 1)
fn process_texture(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims);
    var color = textureSampleLevel(input_texture, tex_sampler, uv, 0.0).rgb;
    
    // Apply brightness
    color = color + vec3<f32>(params.brightness);
    
    // Apply contrast
    color = (color - 0.5) * params.contrast + 0.5;
    
    // Apply gamma correction
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / params.gamma));
    
    // Apply saturation and hue shift
    if (params.saturation != 1.0 || params.hue_shift != 0.0) {
        var hsv = rgb_to_hsv(color);
        hsv.y = hsv.y * params.saturation;  // Saturation
        hsv.x = (hsv.x + params.hue_shift * 180.0 / PI) % 360.0;  // Hue shift
        if (hsv.x < 0.0) { hsv.x += 360.0; }
        color = hsv_to_rgb(hsv);
    }
    
    // Apply vignette effect
    if (params.vignette > 0.0) {
        let center = vec2<f32>(0.5);
        let dist = distance(uv, center);
        let vignette_factor = 1.0 - smoothstep(0.3, 0.8, dist * params.vignette);
        color = color * vignette_factor;
    }
    
    // Add noise
    if (params.noise > 0.0) {
        let noise_coord = uv * 1000.0 + time_data.time * 0.1;
        let noise_val = (noise(noise_coord) - 0.5) * params.noise * 0.1;
        color = color + vec3<f32>(noise_val);
    }
    
    // Clamp final color
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}