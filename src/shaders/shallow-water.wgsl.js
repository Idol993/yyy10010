export const shallowWaterWgsl = /* wgsl */ `

const GRID_SIZE: u32 = 1024u;
const GRID_SIZE_F: f32 = 1024.0;
const HALO: u32 = 2u;
const INV_GRID: f32 = 1.0 / GRID_SIZE_F;

struct SimParams {
    gravity: f32,
    h0: f32,
    viscosity: f32,
    cfl: f32,
    dt: f32,
    dx: f32,
    time: f32,
    pad: f32
};

struct InteractionParams {
    posX: f32,
    posY: f32,
    radius: f32,
    strength: f32,
    mode: u32,
    active: u32,
    pad0: u32,
    pad1: u32
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<uniform> interaction: InteractionParams;

@group(1) @binding(0) var<storage, read_write> h_buf: array<f32>;
@group(1) @binding(1) var<storage, read_write> hu_buf: array<f32>;
@group(1) @binding(2) var<storage, read_write> hv_buf: array<f32>;
@group(1) @binding(3) var<storage, read_write> hp_buf: array<f32>;
@group(1) @binding(4) var<storage, read_write> hup_buf: array<f32>;
@group(1) @binding(5) var<storage, read_write> hvp_buf: array<f32>;
@group(1) @binding(6) var<storage, read_write> terrain_buf: array<f32>;
@group(1) @binding(7) var<storage, read_write> obstacle_buf: array<f32>;
@group(1) @binding(8) var<storage, read_write> height_buf: array<f32>;
@group(1) @binding(9) var<storage, read_write> normal_buf: array<vec2<f32>>;

fn idx(x: u32, y: u32) -> u32 {
    return y * GRID_SIZE + x;
}

fn safe_idx(x: i32, y: i32) -> u32 {
    let cx: u32 = u32(clamp(x, 0, i32(GRID_SIZE) - 1));
    let cy: u32 = u32(clamp(y, 0, i32(GRID_SIZE) - 1));
    return cy * GRID_SIZE + cx;
}

fn sample_field(field: ptr<storage, array<f32>, read_write>, x: i32, y: i32) -> f32 {
    return (*field)[safe_idx(x, y)];
}

fn sample_with_boundary(field: ptr<storage, array<f32>, read_write>, x: i32, y: i32) -> f32 {
    let cx: u32 = u32(clamp(x, 0, i32(GRID_SIZE) - 1));
    let cy: u32 = u32(clamp(y, 0, i32(GRID_SIZE) - 1));
    let obs: f32 = obstacle_buf[cy * GRID_SIZE + cx];
    
    if obs > 0.5 {
        let rx: i32 = 2 * i32(cx) - x;
        let ry: i32 = 2 * i32(cy) - y;
        let rx2: u32 = u32(clamp(rx, 0, i32(GRID_SIZE) - 1));
        let ry2: u32 = u32(clamp(ry, 0, i32(GRID_SIZE) - 1));
        return (*field)[ry2 * GRID_SIZE + rx2] * 0.5;
    }
    
    return (*field)[safe_idx(x, y)];
}

fn sample_bilinear(field: ptr<storage, array<f32>, read_write>, fx: f32, fy: f32) -> f32 {
    let x: f32 = clamp(fx, 0.0, GRID_SIZE_F - 1.001);
    let y: f32 = clamp(fy, 0.0, GRID_SIZE_F - 1.001);
    
    let x0: u32 = u32(floor(x));
    let y0: u32 = u32(floor(y));
    let x1: u32 = min(x0 + 1u, GRID_SIZE - 1u);
    let y1: u32 = min(y0 + 1u, GRID_SIZE - 1u);
    
    let tx: f32 = x - f32(x0);
    let ty: f32 = y - f32(y0);
    
    let v00: f32 = (*field)[y0 * GRID_SIZE + x0];
    let v10: f32 = (*field)[y0 * GRID_SIZE + x1];
    let v01: f32 = (*field)[y1 * GRID_SIZE + x0];
    let v11: f32 = (*field)[y1 * GRID_SIZE + x1];
    
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

fn sample_trilinear(field: ptr<storage, array<f32>, read_write>, field_prev: ptr<storage, array<f32>, read_write>, fx: f32, fy: f32, t: f32) -> f32 {
    return mix(sample_bilinear(field_prev, fx, fy), sample_bilinear(field, fx, fy), clamp(t, 0.0, 1.0));
}

@compute @workgroup_size(16, 16)
fn initialize_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if x >= GRID_SIZE || y >= GRID_SIZE { return; }
    
    let i: u32 = idx(x, y);
    let xf: f32 = f32(x) * INV_GRID;
    let yf: f32 = f32(y) * INV_GRID;
    
    let dx: f32 = xf - 0.5;
    let dy: f32 = yf - 0.5;
    let dist: f32 = sqrt(dx * dx + dy * dy);
    
    var terrain_h: f32 = params.h0 * 0.12 * exp(-dist * dist * 8.0);
    terrain_h += params.h0 * 0.04 * (sin(xf * 12.566) * 0.5 + 0.5) * exp(-abs(yf - 0.7) * 5.0);
    terrain_h += params.h0 * 0.03 * sin(xf * 25.0 + yf * 18.0) * 0.5 + 0.5;
    
    terrain_buf[i] = terrain_h;
    obstacle_buf[i] = select(0.0, 1.0, terrain_h > params.h0 * 0.5);
    h_buf[i] = max(params.h0 - terrain_h, 0.05);
    hu_buf[i] = 0.0;
    hv_buf[i] = 0.0;
    hp_buf[i] = h_buf[i];
    hup_buf[i] = 0.0;
    hvp_buf[i] = 0.0;
    height_buf[i] = h_buf[i] + terrain_h;
    normal_buf[i] = vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn apply_boundary_conditions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if x >= GRID_SIZE || y >= GRID_SIZE { return; }
    let i: u32 = idx(x, y);
    
    let obs: f32 = obstacle_buf[i];
    if obs > 0.5 {
        h_buf[i] = 0.0;
        hu_buf[i] = 0.0;
        hv_buf[i] = 0.0;
        hp_buf[i] = 0.0;
        hup_buf[i] = 0.0;
        hvp_buf[i] = 0.0;
        return;
    }
    
    if x < HALO || x >= GRID_SIZE - HALO || y < HALO || y >= GRID_SIZE - HALO {
        let ref_x: u32 = clamp(x, HALO, GRID_SIZE - HALO - 1u);
        let ref_y: u32 = clamp(y, HALO, GRID_SIZE - HALO - 1u);
        let ri: u32 = ref_y * GRID_SIZE + ref_x;
        h_buf[i] = h_buf[ri];
        hu_buf[i] = -hu_buf[ri];
        hv_buf[i] = -hv_buf[ri];
    }
}

@compute @workgroup_size(16, 16)
fn maccormack_predictor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if x < 1u || x >= GRID_SIZE - 1u || y < 1u || y >= GRID_SIZE - 1u { return; }
    
    let i: u32 = idx(x, y);
    if obstacle_buf[i] > 0.5 { return; }
    
    let ix: i32 = i32(x);
    let iy: i32 = i32(y);
    let dt: f32 = params.dt;
    let dx: f32 = params.dx;
    let g: f32 = params.gravity;
    let inv_dx: f32 = 1.0 / dx;
    
    let h_l: f32 = sample_with_boundary(&h_buf, ix - 1, iy);
    let h_r: f32 = sample_with_boundary(&h_buf, ix + 1, iy);
    let h_d: f32 = sample_with_boundary(&h_buf, ix, iy - 1);
    let h_u: f32 = sample_with_boundary(&h_buf, ix, iy + 1);
    let h_c: f32 = h_buf[i];
    
    let hu_l: f32 = sample_with_boundary(&hu_buf, ix - 1, iy);
    let hu_r: f32 = sample_with_boundary(&hu_buf, ix + 1, iy);
    let hu_d: f32 = sample_with_boundary(&hu_buf, ix, iy - 1);
    let hu_u: f32 = sample_with_boundary(&hu_buf, ix, iy + 1);
    let hu_c: f32 = hu_buf[i];
    
    let hv_l: f32 = sample_with_boundary(&hv_buf, ix - 1, iy);
    let hv_r: f32 = sample_with_boundary(&hv_buf, ix + 1, iy);
    let hv_d: f32 = sample_with_boundary(&hv_buf, ix, iy - 1);
    let hv_u: f32 = sample_with_boundary(&hv_buf, ix, iy + 1);
    let hv_c: f32 = hv_buf[i];
    
    let fhu_l: f32 = select(hu_l * hu_l / h_l + 0.5 * g * h_l * h_l, 0.0, h_l < 0.001);
    let fhu_r: f32 = select(hu_r * hu_r / h_r + 0.5 * g * h_r * h_r, 0.0, h_r < 0.001);
    let fhv_d: f32 = select(hv_d * hv_d / h_d + 0.5 * g * h_d * h_d, 0.0, h_d < 0.001);
    let fhv_u: f32 = select(hv_u * hv_u / h_u + 0.5 * g * h_u * h_u, 0.0, h_u < 0.001);
    let fhu_d: f32 = select(hu_d * hv_d / h_d, 0.0, h_d < 0.001);
    let fhu_u: f32 = select(hu_u * hv_u / h_u, 0.0, h_u < 0.001);
    let fhv_l: f32 = select(hu_l * hv_l / h_l, 0.0, h_l < 0.001);
    let fhv_r: f32 = select(hu_r * hv_r / h_r, 0.0, h_r < 0.001);
    
    let div_h: f32 = (hu_r - hu_l) * 0.5 * inv_dx + (hv_u - hv_d) * 0.5 * inv_dx;
    let div_hu: f32 = (fhu_r - fhu_l) * 0.5 * inv_dx + (fhu_u - fhu_d) * 0.5 * inv_dx;
    let div_hv: f32 = (fhv_r - fhv_l) * 0.5 * inv_dx + (fhv_u - fhv_d) * 0.5 * inv_dx;
    
    let gh_l: f32 = terrain_buf[safe_idx(ix - 1, iy)];
    let gh_r: f32 = terrain_buf[safe_idx(ix + 1, iy)];
    let gh_d: f32 = terrain_buf[safe_idx(ix, iy - 1)];
    let gh_u: f32 = terrain_buf[safe_idx(ix, iy + 1)];
    let slope_x: f32 = g * h_c * (gh_r - gh_l) * 0.5 * inv_dx;
    let slope_y: f32 = g * h_c * (gh_u - gh_d) * 0.5 * inv_dx;
    
    let visc: f32 = params.viscosity;
    let laplacian: f32 = inv_dx * inv_dx;
    let lap_h: f32 = (h_l + h_r + h_d + h_u - 4.0 * h_c) * laplacian;
    let lap_hu: f32 = (hu_l + hu_r + hu_d + hu_u - 4.0 * hu_c) * laplacian;
    let lap_hv: f32 = (hv_l + hv_r + hv_d + hv_u - 4.0 * hv_c) * laplacian;
    
    hp_buf[i] = max(h_c + dt * (-div_h + visc * lap_h), 0.01);
    hup_buf[i] = hu_c + dt * (-div_hu - slope_x + visc * lap_hu);
    hvp_buf[i] = hv_c + dt * (-div_hv - slope_y + visc * lap_hv);
}

@compute @workgroup_size(16, 16)
fn maccormack_corrector(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if x < 1u || x >= GRID_SIZE - 1u || y < 1u || y >= GRID_SIZE - 1u { return; }
    
    let i: u32 = idx(x, y);
    if obstacle_buf[i] > 0.5 { return; }
    
    let ix: i32 = i32(x);
    let iy: i32 = i32(y);
    let dt: f32 = params.dt;
    let dx: f32 = params.dx;
    let g: f32 = params.gravity;
    let inv_dx: f32 = 1.0 / dx;
    
    let h_l: f32 = sample_with_boundary(&hp_buf, ix - 1, iy);
    let h_r: f32 = sample_with_boundary(&hp_buf, ix + 1, iy);
    let h_d: f32 = sample_with_boundary(&hp_buf, ix, iy - 1);
    let h_u: f32 = sample_with_boundary(&hp_buf, ix, iy + 1);
    let h_c: f32 = hp_buf[i];
    
    let hu_l: f32 = sample_with_boundary(&hup_buf, ix - 1, iy);
    let hu_r: f32 = sample_with_boundary(&hup_buf, ix + 1, iy);
    let hu_d: f32 = sample_with_boundary(&hup_buf, ix, iy - 1);
    let hu_u: f32 = sample_with_boundary(&hup_buf, ix, iy + 1);
    let hu_c: f32 = hup_buf[i];
    
    let hv_l: f32 = sample_with_boundary(&hvp_buf, ix - 1, iy);
    let hv_r: f32 = sample_with_boundary(&hvp_buf, ix + 1, iy);
    let hv_d: f32 = sample_with_boundary(&hvp_buf, ix, iy - 1);
    let hv_u: f32 = sample_with_boundary(&hvp_buf, ix, iy + 1);
    let hv_c: f32 = hvp_buf[i];
    
    let fhu_l: f32 = select(hu_l * hu_l / h_l + 0.5 * g * h_l * h_l, 0.0, h_l < 0.001);
    let fhu_r: f32 = select(hu_r * hu_r / h_r + 0.5 * g * h_r * h_r, 0.0, h_r < 0.001);
    let fhv_d: f32 = select(hv_d * hv_d / h_d + 0.5 * g * h_d * h_d, 0.0, h_d < 0.001);
    let fhv_u: f32 = select(hv_u * hv_u / h_u + 0.5 * g * h_u * h_u, 0.0, h_u < 0.001);
    let fhu_d: f32 = select(hu_d * hv_d / h_d, 0.0, h_d < 0.001);
    let fhu_u: f32 = select(hu_u * hv_u / h_u, 0.0, h_u < 0.001);
    let fhv_l: f32 = select(hu_l * hv_l / h_l, 0.0, h_l < 0.001);
    let fhv_r: f32 = select(hu_r * hv_r / h_r, 0.0, h_r < 0.001);
    
    let div_h: f32 = (hu_r - hu_l) * 0.5 * inv_dx + (hv_u - hv_d) * 0.5 * inv_dx;
    let div_hu: f32 = (fhu_r - fhu_l) * 0.5 * inv_dx + (fhu_u - fhu_d) * 0.5 * inv_dx;
    let div_hv: f32 = (fhv_r - fhv_l) * 0.5 * inv_dx + (fhv_u - fhv_d) * 0.5 * inv_dx;
    
    let gh_l: f32 = terrain_buf[safe_idx(ix - 1, iy)];
    let gh_r: f32 = terrain_buf[safe_idx(ix + 1, iy)];
    let gh_d: f32 = terrain_buf[safe_idx(ix, iy - 1)];
    let gh_u: f32 = terrain_buf[safe_idx(ix, iy + 1)];
    let slope_x: f32 = g * h_c * (gh_r - gh_l) * 0.5 * inv_dx;
    let slope_y: f32 = g * h_c * (gh_u - gh_d) * 0.5 * inv_dx;
    
    let visc: f32 = params.viscosity;
    let laplacian: f32 = inv_dx * inv_dx;
    let lap_h: f32 = (h_l + h_r + h_d + h_u - 4.0 * h_c) * laplacian;
    let lap_hu: f32 = (hu_l + hu_r + hu_d + hu_u - 4.0 * hu_c) * laplacian;
    let lap_hv: f32 = (hv_l + hv_r + hv_d + hv_u - 4.0 * hv_c) * laplacian;
    
    let dh_dt_c: f32 = -div_h + visc * lap_h;
    let dhu_dt_c: f32 = -div_hu - slope_x + visc * lap_hu;
    let dhv_dt_c: f32 = -div_hv - slope_y + visc * lap_hv;
    
    let h_old: f32 = h_buf[i];
    let hu_old: f32 = hu_buf[i];
    let hv_old: f32 = hv_buf[i];
    
    h_buf[i] = max(0.5 * (h_old + h_c + dt * dh_dt_c), 0.01);
    hu_buf[i] = 0.5 * (hu_old + hu_c + dt * dhu_dt_c);
    hv_buf[i] = 0.5 * (hv_old + hv_c + dt * dhv_dt_c);
}

@compute @workgroup_size(16, 16)
fn compute_height_and_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if x >= GRID_SIZE || y >= GRID_SIZE { return; }
    
    let i: u32 = idx(x, y);
    let terr: f32 = terrain_buf[i];
    let water_h: f32 = h_buf[i];
    
    height_buf[i] = water_h + terr;
    
    let xl: u32 = select(x - 1u, 0u, x > 0u);
    let xr: u32 = select(x + 1u, GRID_SIZE - 1u, x < GRID_SIZE - 1u);
    let yl: u32 = select(y - 1u, 0u, y > 0u);
    let yu: u32 = select(y + 1u, GRID_SIZE - 1u, y < GRID_SIZE - 1u);
    
    let hxl: f32 = height_buf[yl * GRID_SIZE + xl];
    let hxr: f32 = height_buf[y * GRID_SIZE + xr];
    let hyl: f32 = height_buf[yl * GRID_SIZE + x];
    let hyu: f32 = height_buf[yu * GRID_SIZE + x];
    
    normal_buf[i] = vec2<f32>(
        (hxr - hxl) * 0.5 * GRID_SIZE_F,
        (hyu - hyl) * 0.5 * GRID_SIZE_F
    );
}

@compute @workgroup_size(16, 16)
fn apply_interaction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if x >= GRID_SIZE || y >= GRID_SIZE { return; }
    if interaction.active == 0u { return; }
    
    let i: u32 = idx(x, y);
    let xf: f32 = f32(x) * INV_GRID;
    let yf: f32 = f32(y) * INV_GRID;
    
    let ddx: f32 = xf - interaction.posX;
    let ddy: f32 = yf - interaction.posY;
    let dist_sq: f32 = ddx * ddx + ddy * ddy;
    let radius_sq: f32 = interaction.radius * interaction.radius;
    
    if dist_sq > radius_sq * 4.0 { return; }
    
    let falloff: f32 = exp(-dist_sq / (radius_sq * 0.25));
    let strength: f32 = interaction.strength * falloff;
    
    switch interaction.mode {
        case 0u: {
            if obstacle_buf[i] < 0.5 {
                h_buf[i] = max(h_buf[i] + strength * 0.3, 0.01);
                hu_buf[i] += strength * ddx * 20.0;
                hv_buf[i] += strength * ddy * 20.0;
            }
        }
        case 1u: {
            let new_terr: f32 = clamp(terrain_buf[i] + strength * 0.1, 0.0, params.h0 * 0.9);
            terrain_buf[i] = new_terr;
            obstacle_buf[i] = select(0.0, 1.0, new_terr > params.h0 * 0.55);
            h_buf[i] = max(params.h0 - new_terr, 0.05);
        }
        case 2u: {
            terrain_buf[i] = params.h0 * 0.8 * falloff + terrain_buf[i] * (1.0 - falloff);
            obstacle_buf[i] = select(0.0, 1.0, terrain_buf[i] > params.h0 * 0.55);
            h_buf[i] = max(params.h0 - terrain_buf[i], 0.05);
            if obstacle_buf[i] > 0.5 {
                hu_buf[i] = 0.0;
                hv_buf[i] = 0.0;
            }
        }
        default: { }
    }
}
`;
