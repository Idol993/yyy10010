import { waterRendererWgsl } from './shaders/water-renderer.wgsl.js';

export class WaterRenderer {
    constructor(device, context, simulator) {
        this.device = device;
        this.context = context;
        this.simulator = simulator;
        this.renderSize = 256;

        this.renderParams = {
            heightAmplify: 3.0, time: 0, fresnelPower: 1.5,
            refractStrength: 0.7, showWireframe: 0
        };

        this.camera = {
            position: [0, 25, 50], target: [0, 0, 0], up: [0, 1, 0],
            fov: Math.PI / 4, near: 0.1, far: 500,
            yaw: 0, pitch: -0.35, distance: 60
        };

        this.light = {
            sunDir: [0.5, 0.8, 0.3],
            sunColor: [1.0, 0.98, 0.9, 1.0],
            ambientColor: [0.15, 0.2, 0.3, 1.0],
            waterColor: [0.1, 0.5, 0.7, 1.0],
            deepColor: [0.02, 0.08, 0.15, 1.0]
        };

        this.buffers = {};
        this.pipelines = {};
        this.bindGroups = {};
        this.depthTexture = null;
        this.msaaTexture = null;
        this._canvasWidth = 0;
        this._canvasHeight = 0;
    }

    async init() {
        this._createBuffers();
        this._createPipelines();
        this._createBindGroups();
        this._updateCamera();
        return this;
    }

    _createBuffers() {
        const device = this.device;
        const camSize = 4 * 16 + 4 * 16 + 16 + 16 + 8 + 8;
        this.buffers.camera = device.createBuffer({
            size: camSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.buffers.render = device.createBuffer({
            size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.buffers.light = device.createBuffer({
            size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    _createPipelines() {
        const device = this.device;
        const module = device.createShaderModule({ code: waterRendererWgsl, label: 'RenderShader' });

        const bgl0 = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });

        const simBuffers = this.simulator.getBuffersForRender();
        const bgl1 = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
            ]
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bgl0, bgl1]
        });

        this.pipelines.water = device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: { module, entryPoint: 'vs_water' },
            fragment: {
                module, entryPoint: 'fs_water',
                targets: [{ format: this.context.getPreferredFormat ? this.context.getPreferredFormat() : navigator.gpu.getPreferredCanvasFormat() }]
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: true,
                depthCompare: 'less'
            },
            multisample: { count: 4 }
        });

        this.pipelines.skybox = device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: { module, entryPoint: 'vs_skybox' },
            fragment: {
                module, entryPoint: 'fs_skybox',
                targets: [{ format: this.context.getPreferredFormat ? this.context.getPreferredFormat() : navigator.gpu.getPreferredCanvasFormat() }]
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: false,
                depthCompare: 'less-equal'
            },
            multisample: { count: 4 }
        });

        this._bgl0 = bgl0;
        this._bgl1 = bgl1;
    }

    _createBindGroups() {
        const device = this.device;
        const simBufs = this.simulator.getBuffersForRender();

        this.bindGroups.uniforms = device.createBindGroup({
            layout: this._bgl0,
            entries: [
                { binding: 0, resource: { buffer: this.buffers.camera } },
                { binding: 1, resource: { buffer: this.buffers.render } },
                { binding: 2, resource: { buffer: this.buffers.light } }
            ]
        });

        this.bindGroups.simData = device.createBindGroup({
            layout: this._bgl1,
            entries: [
                { binding: 0, resource: { buffer: simBufs.height } },
                { binding: 1, resource: { buffer: simBufs.normal } },
                { binding: 2, resource: { buffer: simBufs.terrain } },
                { binding: 3, resource: { buffer: simBufs.obstacle } }
            ]
        });
    }

    _ensureRenderTextures(width, height) {
        if (this._canvasWidth === width && this._canvasHeight === height) return;
        this._canvasWidth = width;
        this._canvasHeight = height;

        if (this.depthTexture) this.depthTexture.destroy();
        if (this.msaaTexture) this.msaaTexture.destroy();

        const fmt = navigator.gpu.getPreferredCanvasFormat();
        this.depthTexture = this.device.createTexture({
            size: [width, height], format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: 4
        });
        this.msaaTexture = this.device.createTexture({
            size: [width, height], format: fmt,
            usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: 4
        });
    }

    _updateCamera() {
        const { yaw, pitch, distance, target, up } = this.camera;
        const cx = target[0] + Math.sin(yaw) * Math.cos(pitch) * distance;
        const cy = target[1] + Math.sin(pitch) * distance;
        const cz = target[2] + Math.cos(yaw) * Math.cos(pitch) * distance;
        this.camera.position = [cx, cy, cz];

        const view = this._lookAt([cx, cy, cz], target, up);
        const aspect = this._canvasWidth / Math.max(this._canvasHeight, 1);
        const proj = this._perspective(this.camera.fov, aspect, this.camera.near, this.camera.far);
        const viewProj = this._mat4Mul(proj, view);
        const invViewProj = this._mat4Inverse(viewProj);

        const data = new Float32Array(4 * 16 + 4 * 16 + 4 + 4 + 2 + 2);
        let o = 0;
        data.set(viewProj, o); o += 16;
        data.set(invViewProj, o); o += 16;
        data.set([cx, cy, cz, 1.0], o); o += 4;
        const dir = this._normalize([target[0] - cx, target[1] - cy, target[2] - cz]);
        data.set([dir[0], dir[1], dir[2], 0.0], o); o += 4;
        data.set([this.camera.near, this.camera.far], o);
        this.device.queue.writeBuffer(this.buffers.camera, 0, data);
    }

    _updateRender() {
        const ws = this.simulator.worldSize;
        const data = new Float32Array([
            ws, ws, this.renderParams.heightAmplify, this.renderParams.time,
            this.renderParams.fresnelPower, this.renderParams.refractStrength,
            this.renderParams.showWireframe, 0
        ]);
        this.device.queue.writeBuffer(this.buffers.render, 0, data);
    }

    _updateLight() {
        const dir = this._normalize(this.light.sunDir);
        const data = new Float32Array([
            dir[0], dir[1], dir[2], 0,
            ...this.light.sunColor,
            ...this.light.ambientColor,
            ...this.light.waterColor,
            ...this.light.deepColor
        ]);
        this.device.queue.writeBuffer(this.buffers.light, 0, data);
    }

    rotateCamera(dx, dy) {
        this.camera.yaw += dx * 0.01;
        this.camera.pitch = Math.max(-1.47, Math.min(1.47, this.camera.pitch + dy * 0.01));
    }

    zoomCamera(delta) {
        this.camera.distance = Math.max(10, Math.min(200, this.camera.distance * (1 + delta * 0.001)));
    }

    render(canvasTexture) {
        const w = this.context.canvas.width;
        const h = this.context.canvas.height;
        this._ensureRenderTextures(w, h);

        this._updateCamera();
        this._updateRender();
        this._updateLight();

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.msaaTexture.createView(),
                resolveTarget: canvasTexture,
                clearValue: { r: 0.05, g: 0.08, b: 0.15, a: 1.0 },
                loadOp: 'clear', storeOp: 'store'
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store'
            }
        });

        pass.setPipeline(this.pipelines.skybox);
        pass.setBindGroup(0, this.bindGroups.uniforms);
        pass.setBindGroup(1, this.bindGroups.simData);
        pass.draw(3);

        pass.setPipeline(this.pipelines.water);
        pass.setBindGroup(0, this.bindGroups.uniforms);
        pass.setBindGroup(1, this.bindGroups.simData);
        const faceCount = (this.renderSize - 1) * (this.renderSize - 1);
        pass.draw(faceCount * 6);

        pass.end();
        this.device.queue.submit([encoder.finish()]);

        this.renderParams.time += 0.016;
    }

    setRenderParam(name, value) {
        if (name in this.renderParams) this.renderParams[name] = value;
    }

    _perspective(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0, 0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, (2 * far * near) * nf, 0
        ]);
    }

    _lookAt(eye, target, up) {
        const z = this._normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
        const x = this._normalize(this._cross(up, z));
        const y = this._cross(z, x);
        return new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -this._dot(x, eye), -this._dot(y, eye), -this._dot(z, eye), 1
        ]);
    }

    _mat4Mul(a, b) {
        const o = new Float32Array(16);
        for (let i = 0; i < 4; i++)
            for (let j = 0; j < 4; j++) {
                o[i * 4 + j] = 0;
                for (let k = 0; k < 4; k++)
                    o[i * 4 + j] += a[k * 4 + j] * b[i * 4 + k];
            }
        return o;
    }

    _mat4Inverse(m) {
        const inv = new Float32Array(16);
        inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
        inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
        inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
        inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
        inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
        inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
        inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
        inv[13] = m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
        inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
        inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
        inv[10] = m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
        inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
        inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
        inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
        inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
        inv[15] = m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];

        let det = m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12];
        if (det === 0) return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
        det = 1.0 / det;
        for (let i = 0; i < 16; i++) inv[i] *= det;
        return inv;
    }

    _cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
    _dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
    _normalize(v) {
        const l = Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        return l === 0 ? [0,0,0] : [v[0]/l, v[1]/l, v[2]/l];
    }
}
