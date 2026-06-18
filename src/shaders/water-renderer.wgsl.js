export const waterRendererWgsl = /* wgsl */ `

const SIM_SIZE: u32 = 1024u;
const RENDER_SIZE: u32 = 256u;
const RENDER_SIZE_F: f32 = 256.0;
const SIM_SIZE_F: f32 = 1024.0;

struct CameraUniforms {
    viewProj: mat4x4<f32>,
    invViewProj: mat4x4<f32>,
    viewPos: vec4<f32>,
    viewDir: vec4<f32>,
    nearFar: vec2<f32>,
    pad0: vec2<f32>
};

struct RenderUniforms {
    worldSize: vec2<f32>,
    heightAmplify: f32,
    time: f32,
    fresnelPower: f32,
    refractStrength: f32,
    showWireframe: f32,
    pad0: f32,
    pad1: f32
};

struct LightUniforms {
    sunDir: vec4<f32>,
    sunColor: vec4<f32>,
    ambientColor: vec4<f32>,
    waterColor: vec4<f32>,
    deepColor: vec4<f32>
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render: RenderUniforms;
@group(0) @binding(2) var<uniform> light: LightUniforms;

@group(1) @binding(0) var<storage, read> height_buf: array<f32>;
@group(1) @binding(1) var<storage, read> normal_buf: array<vec2<f32>>;
@group(1) @binding(2) var<storage, read> terrain_buf: array<f32>;
@group(1) @binding(3) var<storage, read> obstacle_buf: array<f32>;

fn sim_idx(x: u32, y: u32) -> u32 {
    return clamp(y, 0u, SIM_SIZE - 1u) * SIM_SIZE + clamp(x, 0u, SIM_SIZE - 1u);
}

fn sample_height(uv: vec2<f32>) -> f32 {
    let texel: vec2<f32> = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0 - 0.0009765625)) * SIM_SIZE_F;
    let x0: u32 = u32(floor(texel.x));
    let y0: u32 = u32(floor(texel.y));
    let x1: u32 = min(x0 + 1u, SIM_SIZE - 1u);
    let y1: u32 = min(y0 + 1u, SIM_SIZE - 1u);
    let tx: f32 = fract(texel.x);
    let ty: f32 = fract(texel.y);
    return mix(
        mix(height_buf[sim_idx(x0, y0)], height_buf[sim_idx(x1, y0)], tx),
        mix(height_buf[sim_idx(x0, y1)], height_buf[sim_idx(x1, y1)], tx),
        ty
    );
}

fn sample_normal(uv: vec2<f32>) -> vec2<f32> {
    let texel: vec2<f32> = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0 - 0.0009765625)) * SIM_SIZE_F;
    let x0: u32 = u32(floor(texel.x));
    let y0: u32 = u32(floor(texel.y));
    let x1: u32 = min(x0 + 1u, SIM_SIZE - 1u);
    let y1: u32 = min(y0 + 1u, SIM_SIZE - 1u);
    let tx: f32 = fract(texel.x);
    let ty: f32 = fract(texel.y);
    return mix(
        mix(normal_buf[sim_idx(x0, y0)], normal_buf[sim_idx(x1, y0)], tx),
        mix(normal_buf[sim_idx(x0, y1)], normal_buf[sim_idx(x1, y1)], tx),
        ty
    );
}

fn sample_terrain(uv: vec2<f32>) -> f32 {
    let texel: vec2<f32> = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0 - 0.0009765625)) * SIM_SIZE_F;
    let x0: u32 = u32(floor(texel.x));
    let y0: u32 = u32(floor(texel.y));
    let x1: u32 = min(x0 + 1u, SIM_SIZE - 1u);
    let y1: u32 = min(y0 + 1u, SIM_SIZE - 1u);
    let tx: f32 = fract(texel.x);
    let ty: f32 = fract(texel.y);
    return mix(
        mix(terrain_buf[sim_idx(x0, y0)], terrain_buf[sim_idx(x1, y0)], tx),
        mix(terrain_buf[sim_idx(x0, y1)], terrain_buf[sim_idx(x1, y1)], tx),
        ty
    );
}

fn sample_obstacle(uv: vec2<f32>) -> f32 {
    let cx: u32 = u32(clamp(floor(uv.x * SIM_SIZE_F), 0.0, SIM_SIZE_F - 1.0));
    let cy: u32 = u32(clamp(floor(uv.y * SIM_SIZE_F), 0.0, SIM_SIZE_F - 1.0));
    return obstacle_buf[sim_idx(cx, cy)];
}

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) viewDir: vec3<f32>,
    @location(4) waterDepth: f32,
    @location(5) terrainH: f32
};

@vertex
fn vs_water(@builtin(vertex_index) vid: u32) -> VSOut {
    let face_count: u32 = (RENDER_SIZE - 1u) * (RENDER_SIZE - 1u);
    let face: u32 = vid / 6u;
    let local: u32 = vid % 6u;
    
    var li: u32 = 0u;
    if local == 0u { li = 0u; }
    else if local == 1u { li = 1u; }
    else if local == 2u { li = 2u; }
    else if local == 3u { li = 2u; }
    else if local == 4u { li = 1u; }
    else { li = 3u; }
    
    let faceX: u32 = face % (RENDER_SIZE - 1u);
    let faceY: u32 = face / (RENDER_SIZE - 1u);
    
    let ox: u32 = faceX + (li & 1u);
    let oy: u32 = faceY + ((li >> 1u) & 1u);
    
    let u: f32 = f32(ox) / (RENDER_SIZE_F - 1.0);
    let v: f32 = f32(oy) / (RENDER_SIZE_F - 1.0);
    let uv: vec2<f32> = vec2<f32>(u, v);
    
    let rawHeight: f32 = sample_height(uv);
    let terrainH: f32 = sample_terrain(uv);
    let amplifiedH: f32 = (rawHeight - terrainH) * render.heightAmplify + terrainH;
    
    let worldX: f32 = (u - 0.5) * render.worldSize.x;
    let worldZ: f32 = (v - 0.5) * render.worldSize.y;
    let worldPos: vec3<f32> = vec3<f32>(worldX, amplifiedH, worldZ);
    
    let grad: vec2<f32> = sample_normal(uv);
    let nx: f32 = -grad.x * render.heightAmplify;
    let nz: f32 = -grad.y * render.heightAmplify;
    let N: vec3<f32> = normalize(vec3<f32>(nx, 1.0, nz));
    
    var out: VSOut;
    out.position = camera.viewProj * vec4<f32>(worldPos, 1.0);
    out.worldPos = worldPos;
    out.uv = uv;
    out.normal = N;
    out.viewDir = normalize(camera.viewPos.xyz - worldPos);
    out.waterDepth = rawHeight - terrainH;
    out.terrainH = terrainH;
    return out;
}

@fragment
fn fs_water(input: VSOut) -> @location(0) vec4<f32> {
    let obs: f32 = sample_obstacle(input.uv);
    
    if obs > 0.5 {
        let terrainColor: vec3<f32> = mix(
            vec3<f32>(0.35, 0.28, 0.2),
            vec3<f32>(0.5, 0.42, 0.35),
            smoothstep(0.0, 0.5, input.terrainH)
        );
        let noise: f32 = fract(sin(dot(input.uv * 500.0, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        let rock: vec3<f32> = terrainColor * (0.75 + 0.25 * noise);
        let ndl: f32 = max(dot(input.normal, light.sunDir.xyz), 0.0);
        return vec4<f32>(pow(rock * (light.ambientColor.rgb + light.sunColor.rgb * ndl), vec3<f32>(1.0 / 2.2)), 1.0);
    }
    
    let N: vec3<f32> = normalize(input.normal);
    let V: vec3<f32> = normalize(input.viewDir);
    let L: vec3<f32> = normalize(light.sunDir.xyz);
    let H: vec3<f32> = normalize(V + L);
    
    let cosTheta: f32 = max(dot(V, N), 0.0);
    let fresnel: f32 = pow(1.0 - cosTheta, render.fresnelPower);
    
    let reflectDir: vec3<f32> = reflect(-V, N);
    let refractDir: vec3<f32> = refract(-V, N, 1.0 / 1.33);
    
    var envReflect: vec3<f32>;
    let skyT: f32 = clamp(reflectDir.y * 0.5 + 0.5, 0.0, 1.0);
    let skyLow: vec3<f32> = vec3<f32>(0.85, 0.9, 0.95);
    let skyHigh: vec3<f32> = vec3<f32>(0.3, 0.5, 0.85);
    envReflect = mix(skyLow, skyHigh, pow(skyT, 0.6));
    
    let sunAng: f32 = dot(normalize(reflectDir), L);
    envReflect += vec3<f32>(1.0, 0.95, 0.8) * smoothstep(0.995, 0.9998, sunAng) * 3.0;
    envReflect += vec3<f32>(1.0, 0.8, 0.5) * pow(max(sunAng, 0.0), 32.0) * 0.5;
    
    let cloudUV: vec2<f32> = reflectDir.xz / max(reflectDir.y, 0.05) * 0.5;
    let clouds: f32 = sin(cloudUV.x * 2.0 + render.time * 0.05) * sin(cloudUV.y * 2.5) * 0.5 + 0.5;
    envReflect = mix(envReflect, vec3<f32>(1.0), clouds * 0.15 * skyT);
    
    let refractUV: vec2<f32> = input.uv + refractDir.xz * render.refractStrength * 0.05;
    let subsurfaceT: f32 = sample_terrain(refractUV);
    let depth: f32 = max(input.waterDepth, 0.0);
    let depthT: f32 = 1.0 - exp(-depth * 2.0);
    
    let shallowC: vec3<f32> = light.waterColor.rgb;
    let deepC: vec3<f32> = light.deepColor.rgb;
    let waterBody: vec3<f32> = mix(shallowC, deepC, depthT);
    
    let cuv: vec2<f32> = refractUV * 30.0;
    let c1: f32 = sin(cuv.x + render.time * 2.0) * sin(cuv.y - render.time * 1.5);
    let c2: f32 = sin(cuv.x * 0.7 - render.time * 1.2 + 1.0) * sin(cuv.y * 0.8 + render.time * 1.8 + 0.5);
    let caustics: f32 = (c1 + c2) * 0.25 + 0.5;
    
    let subsurfaceColor: vec3<f32> = waterBody * (0.7 + 0.5 * caustics);
    let bottomColor: vec3<f32> = mix(vec3<f32>(0.35, 0.3, 0.2), vec3<f32>(0.2, 0.15, 0.1), smoothstep(0.0, 0.5, subsurfaceT));
    let bottomBlend: f32 = exp(-depth * 3.0);
    let refractedColor: vec3<f32> = mix(subsurfaceColor, bottomColor, bottomBlend * 0.6);
    
    let color: vec3<f32> = mix(refractedColor, envReflect, fresnel);
    
    let spec: f32 = pow(max(dot(N, H), 0.0), 256.0);
    let specColor: vec3<f32> = light.sunColor.rgb * spec * 2.5;
    
    let diff: f32 = max(dot(N, L), 0.0) * 0.3;
    let diffColor: vec3<f32> = shallowC * diff * light.sunColor.rgb;
    let ambient: vec3<f32> = shallowC * light.ambientColor.rgb * 0.2;
    
    var final_color: vec3<f32> = color + specColor + diffColor + ambient;
    
    let grad: vec2<f32> = sample_normal(input.uv);
    let foam: f32 = smoothstep(1.5, 3.0, abs(grad.x + grad.y) * render.heightAmplify);
    final_color = mix(final_color, vec3<f32>(0.95, 0.97, 1.0), foam * 0.4);
    
    if render.showWireframe > 0.5 {
        let grid: vec2<f32> = abs(fract(input.uv * 64.0) - 0.5);
        let line: f32 = min(grid.x, grid.y);
        let alpha: f32 = smoothstep(0.0, 0.02, line);
        final_color = mix(vec3<f32>(1.0, 0.5, 0.0), final_color, alpha);
    }
    
    return vec4<f32>(pow(final_color, vec3<f32>(1.0 / 2.2)), 1.0);
}

struct SkyVSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) rayDir: vec3<f32>
};

@vertex
fn vs_skybox(@builtin(vertex_index) vid: u32) -> SkyVSOut {
    var pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let p: vec2<f32> = pos[vid];
    let rayClip: vec4<f32> = vec4<f32>(p, 0.9999, 1.0);
    let rayEye: vec4<f32> = camera.invViewProj * rayClip;
    let rayDir: vec3<f32> = normalize(rayEye.xyz / rayEye.w - camera.viewPos.xyz);
    
    var out: SkyVSOut;
    out.position = vec4<f32>(p, 0.9999, 1.0);
    out.rayDir = rayDir;
    return out;
}

@fragment
fn fs_skybox(input: SkyVSOut) -> @location(0) vec4<f32> {
    let dir: vec3<f32> = normalize(input.rayDir);
    let L: vec3<f32> = normalize(light.sunDir.xyz);
    
    let skyT: f32 = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let horizon: vec3<f32> = vec3<f32>(0.85, 0.9, 0.95);
    let zenith: vec3<f32> = vec3<f32>(0.3, 0.5, 0.85);
    var sky: vec3<f32> = mix(horizon, zenith, pow(skyT, 0.6));
    
    let sunAng: f32 = dot(dir, L);
    sky += vec3<f32>(1.0, 0.98, 0.9) * smoothstep(0.995, 0.9998, sunAng) * 3.0;
    sky += vec3<f32>(1.0, 0.8, 0.5) * pow(max(sunAng, 0.0), 32.0) * 0.5;
    
    let cuv: vec2<f32> = dir.xz / max(dir.y, 0.05) * 0.5;
    let c: f32 = sin(cuv.x * 2.0 + render.time * 0.05) * sin(cuv.y * 2.5) * 0.5 + 0.5;
    sky = mix(sky, vec3<f32>(1.0), c * 0.3 * smoothstep(0.3, 0.5, 1.0 - skyT));
    
    return vec4<f32>(pow(sky, vec3<f32>(1.0 / 2.2)), 1.0);
}
`;
