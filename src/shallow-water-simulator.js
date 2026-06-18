import { shallowWaterWgsl } from './shaders/shallow-water.wgsl.js';
import { GPUMemoryPool } from './memory-pool.js';

export class ShallowWaterSimulator {
    constructor(device) {
        this.device = device;
        this.gridSize = 1024;
        this.worldSize = 100;
        this.pool = new GPUMemoryPool(device, 512);

        this.params = {
            gravity: 9.81,
            h0: 1.0,
            viscosity: 0.002,
            cfl: 0.4,
            dt: 0.001,
            dx: 1.0 / 1024,
            time: 0
        };

        this.interaction = {
            posX: 0.5, posY: 0.5, radius: 0.03, strength: 0.5,
            mode: 0, active: 0, direction: 1.0
        };

        this.buffers = {};
        this.pipelines = {};
        this.bindGroups = {};
        this._initialized = false;
    }

    async init() {
        this._createBuffers();
        this._createPipelines();
        this._createBindGroups();
        this._initializeState();
        this._initialized = true;
        return this;
    }

    _createBuffers() {
        const device = this.device;
        const N = this.gridSize * this.gridSize;
        const f4 = 4;
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

        this.buffers.params = device.createBuffer({
            size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.buffers.interaction = device.createBuffer({
            size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.buffers.h = this.pool.allocate(N * f4, usage);
        this.buffers.hu = this.pool.allocate(N * f4, usage);
        this.buffers.hv = this.pool.allocate(N * f4, usage);
        this.buffers.hp = this.pool.allocate(N * f4, usage);
        this.buffers.hup = this.pool.allocate(N * f4, usage);
        this.buffers.hvp = this.pool.allocate(N * f4, usage);
        this.buffers.terrain = this.pool.allocate(N * f4, usage);
        this.buffers.obstacle = this.pool.allocate(N * f4, usage);
        this.buffers.height = this.pool.allocate(N * f4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX);
        this.buffers.normal = this.pool.allocate(N * 2 * f4, usage);
    }

    _createPipelines() {
        const device = this.device;
        const module = device.createShaderModule({ code: shallowWaterWgsl, label: 'SimShader' });

        const entries = ['initialize_state', 'apply_boundary_conditions',
            'maccormack_predictor', 'maccormack_corrector',
            'compute_height_only', 'compute_normals', 'apply_interaction'];

        for (const entry of entries) {
            this.pipelines[entry] = device.createComputePipeline({
                layout: 'auto',
                compute: { module, entryPoint: entry },
                label: `Pipeline_${entry}`
            });
        }

        this._shaderModule = module;
    }

    _createBindGroups() {
        const device = this.device;

        this.bindGroups.params = device.createBindGroup({
            layout: this.pipelines.initialize_state.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.params } },
                { binding: 1, resource: { buffer: this.buffers.interaction } }
            ]
        });

        this.bindGroups.state = device.createBindGroup({
            layout: this.pipelines.initialize_state.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.h } },
                { binding: 1, resource: { buffer: this.buffers.hu } },
                { binding: 2, resource: { buffer: this.buffers.hv } },
                { binding: 3, resource: { buffer: this.buffers.hp } },
                { binding: 4, resource: { buffer: this.buffers.hup } },
                { binding: 5, resource: { buffer: this.buffers.hvp } },
                { binding: 6, resource: { buffer: this.buffers.terrain } },
                { binding: 7, resource: { buffer: this.buffers.obstacle } },
                { binding: 8, resource: { buffer: this.buffers.height } },
                { binding: 9, resource: { buffer: this.buffers.normal } }
            ]
        });
    }

    _initializeState() {
        this._updateParamsBuffer();
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.initialize_state);
        pass.setBindGroup(0, this.bindGroups.params);
        pass.setBindGroup(1, this.bindGroups.state);
        pass.dispatchWorkgroups(Math.ceil(this.gridSize / 16), Math.ceil(this.gridSize / 16));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    _updateParamsBuffer() {
        const maxSpeed = Math.sqrt(this.params.gravity * this.params.h0);
        const dx = this.worldSize / this.gridSize;
        this.params.dx = dx;
        this.params.dt = Math.min(this.params.cfl * dx / maxSpeed, 0.005);

        const data = new Float32Array([
            this.params.gravity, this.params.h0, this.params.viscosity, this.params.cfl,
            this.params.dt, this.params.dx, this.params.time, 0
        ]);
        this.device.queue.writeBuffer(this.buffers.params, 0, data);
    }

    _updateInteractionBuffer() {
        const data = new ArrayBuffer(32);
        const dv = new DataView(data);
        dv.setFloat32(0, this.interaction.posX, true);
        dv.setFloat32(4, this.interaction.posY, true);
        dv.setFloat32(8, this.interaction.radius, true);
        dv.setFloat32(12, this.interaction.strength, true);
        dv.setUint32(16, this.interaction.mode, true);
        dv.setUint32(20, this.interaction.active, true);
        dv.setFloat32(24, this.interaction.direction, true);
        this.device.queue.writeBuffer(this.buffers.interaction, 0, data);
    }

    setInteraction(normX, normY, radius, strength, mode, active, direction = 1.0) {
        this.interaction.posX = normX;
        this.interaction.posY = normY;
        this.interaction.radius = radius;
        this.interaction.strength = strength;
        this.interaction.mode = mode;
        this.interaction.active = active ? 1 : 0;
        this.interaction.direction = direction;
    }

    step(substeps = 2) {
        this._updateParamsBuffer();
        this._updateInteractionBuffer();

        const wg = Math.ceil(this.gridSize / 16);

        for (let s = 0; s < substeps; s++) {
            const encoder = this.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setBindGroup(0, this.bindGroups.params);
            pass.setBindGroup(1, this.bindGroups.state);

            if (s === 0 && this.interaction.active) {
                pass.setPipeline(this.pipelines.apply_interaction);
                pass.dispatchWorkgroups(wg, wg);
            }

            pass.setPipeline(this.pipelines.apply_boundary_conditions);
            pass.dispatchWorkgroups(wg, wg);

            pass.setPipeline(this.pipelines.maccormack_predictor);
            pass.dispatchWorkgroups(wg, wg);

            pass.setPipeline(this.pipelines.apply_boundary_conditions);
            pass.dispatchWorkgroups(wg, wg);

            pass.setPipeline(this.pipelines.maccormack_corrector);
            pass.dispatchWorkgroups(wg, wg);

            pass.end();
            this.device.queue.submit([encoder.finish()]);
        }

        const enc2 = this.device.createCommandEncoder();
        const p2 = enc2.beginComputePass();
        p2.setBindGroup(0, this.bindGroups.params);
        p2.setBindGroup(1, this.bindGroups.state);
        p2.setPipeline(this.pipelines.compute_height_only);
        p2.dispatchWorkgroups(wg, wg);
        p2.setPipeline(this.pipelines.compute_normals);
        p2.dispatchWorkgroups(wg, wg);
        p2.end();
        this.device.queue.submit([enc2.finish()]);

        this.params.time += this.params.dt * substeps;
        this.interaction.active = 0;
    }

    reset() {
        this.params.time = 0;
        this._initializeState();
    }

    createSplash(normX, normY, strength = 1.0, radius = 0.05) {
        this.setInteraction(normX, normY, radius, strength, 0, true, 1.0);
    }

    getBuffersForRender() {
        return {
            height: this.buffers.height,
            normal: this.buffers.normal,
            terrain: this.buffers.terrain,
            obstacle: this.buffers.obstacle
        };
    }

    getSimBindGroupLayout() {
        return this.pipelines.initialize_state.getBindGroupLayout(1);
    }

    getStats() {
        return this.pool.getStats();
    }

    setParam(name, value) {
        if (name in this.params) this.params[name] = value;
    }
}
