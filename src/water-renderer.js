import { waterRendererWgsl } from './shaders/water-renderer.wgsl.js';

export class WaterRenderer {
    constructor(device, gpuContext, simulator) {
        this.device = device;
        this.gpuContext = gpuContext;
        this.simulator = simulator;
        this.canvas = gpuContext.canvas;
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
        this._createShaderModule();
        this._createPipelines();
        this._createBindGroups();
        return this;
    }

    _createBuffers() {
        const device = this.device;

        this.buffers.camera = device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.buffers.render = device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.buffers.light = device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    _createShaderModule() {
        this.shaderModule = this.device.createShaderModule({
            code: waterRendererWgsl,
            label: 'RenderShader'
        });
    }

    _createPipelines() {
        const device = this.device;
        const module = this.shaderModule;
        const targetFmt = navigator.gpu.getPreferredCanvasFormat();

        const depthStencilState = {
            format: 'depth24plus',
            depthWriteEnabled: true,
            depthCompare: 'less'
        };
        const multiSampleState = { count: 4 };
        const primitiveState = { topology: 'triangle-list', cullMode: 'none' };

        this.pipelines.water = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module, entryPoint: 'vs_water' },
            fragment: {
                module, entryPoint: 'fs_water',
                targets: [{ format: targetFmt }]
            },
            primitive: primitiveState,
            depthStencil: depthStencilState,
            multisample: multiSampleState,
            label: 'WaterPipeline'
        });

        this.pipelines.skybox = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module, entryPoint: 'vs_skybox' },
            fragment: {
                module, entryPoint: 'fs_skybox',
                targets: [{ format: targetFmt }]
            },
            primitive: primitiveState,
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: false,
                depthCompare: 'less-equal'
            },
            multisample: multiSampleState,
            label: 'SkyboxPipeline'
        });
    }

    _createBindGroups() {
        const device = this.device;
        const simBufs = this.simulator.getBuffersForRender();

        this.bindGroups.waterUniforms = device.createBindGroup({
            layout: this.pipelines.water.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.camera } },
                { binding: 1, resource: { buffer: this.buffers.render } },
                { binding: 2, resource: { buffer: this.buffers.light } }
            ],
            label: 'WaterUniformBG'
        });

        this.bindGroups.waterSim = device.createBindGroup({
            layout: this.pipelines.water.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: simBufs.height } },
                { binding: 1, resource: { buffer: simBufs.normal } },
                { binding: 2, resource: { buffer: simBufs.terrain } },
                { binding: 3, resource: { buffer: simBufs.obstacle } }
            ],
            label: 'WaterSimBG'
        });

        this.bindGroups.skyboxUniforms = device.createBindGroup({
            layout: this.pipelines.skybox.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.camera } },
                { binding: 1, resource: { buffer: this.buffers.render } },
                { binding: 2, resource: { buffer: this.buffers.light } }
            ],
            label: 'SkyboxUniformBG'
        });

        const skyboxBGL1 = this.pipelines.skybox.getBindGroupLayout(1);
        if (skyboxBGL1) {
            const skyboxEntries = [];
            const skyInfo = skyboxBGL1.entries || [];
            for (const e of skyInfo) {
                if (e.binding === 0) skyboxEntries.push({ binding: 0, resource: { buffer: simBufs.height } });
                if (e.binding === 1) skyboxEntries.push({ binding: 1, resource: { buffer: simBufs.normal } });
                if (e.binding === 2) skyboxEntries.push({ binding: 2, resource: { buffer: simBufs.terrain } });
                if (e.binding === 3) skyboxEntries.push({ binding: 3, resource: { buffer: simBufs.obstacle } });
            }
            if (skyboxEntries.length > 0) {
                this.bindGroups.skyboxSim = device.createBindGroup({
                    layout: skyboxBGL1,
                    entries: skyboxEntries,
                    label: 'SkyboxSimBG'
                });
            }
        }
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

        const data = new Float32Array(64);
        data.set(viewProj, 0);
        data.set(invViewProj, 16);
        data.set([cx, cy, cz, 1.0], 32);
        const dir = this._normalize([target[0] - cx, target[1] - cy, target[2] - cz]);
        data.set([dir[0], dir[1], dir[2], 0.0], 36);
        data.set([this.camera.near, this.camera.far, 0, 0], 40);
        this.device.queue.writeBuffer(this.buffers.camera, 0, data);
    }

    _updateRender() {
        const ws = this.simulator.worldSize;
        const data = new Float32Array(64);
        data[0] = ws;
        data[1] = ws;
        data[2] = this.renderParams.heightAmplify;
        data[3] = this.renderParams.time;
        data[4] = this.renderParams.fresnelPower;
        data[5] = this.renderParams.refractStrength;
        data[6] = this.renderParams.showWireframe;
        data[7] = 0;
        data[8] = 0;
        this.device.queue.writeBuffer(this.buffers.render, 0, data);
    }

    _updateLight() {
        const dir = this._normalize(this.light.sunDir);
        const data = new Float32Array(64);
        data.set([dir[0], dir[1], dir[2], 0], 0);
        data.set(this.light.sunColor, 4);
        data.set(this.light.ambientColor, 8);
        data.set(this.light.waterColor, 12);
        data.set(this.light.deepColor, 16);
        this.device.queue.writeBuffer(this.buffers.light, 0, data);
    }

    rotateCamera(dx, dy) {
        this.camera.yaw += dx * 0.01;
        this.camera.pitch = Math.max(-1.47, Math.min(1.47, this.camera.pitch + dy * 0.01));
    }

    zoomCamera(delta) {
        this.camera.distance = Math.max(10, Math.min(200, this.camera.distance * (1 + delta * 0.001)));
    }

    render(canvasView) {
        const w = this.canvas.width;
        const h = this.canvas.height;
        this._ensureRenderTextures(w, h);

        this._updateCamera();
        this._updateRender();
        this._updateLight();

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.msaaTexture.createView(),
                resolveTarget: canvasView,
                clearValue: { r: 0.05, g: 0.08, b: 0.15, a: 1.0 },
                loadOp: 'clear', storeOp: 'store'
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store'
            },
            label: 'MainPass'
        });

        pass.setPipeline(this.pipelines.skybox);
        pass.setBindGroup(0, this.bindGroups.skyboxUniforms);
        if (this.bindGroups.skyboxSim) {
            pass.setBindGroup(1, this.bindGroups.skyboxSim);
        }
        pass.draw(3);

        pass.setPipeline(this.pipelines.water);
        pass.setBindGroup(0, this.bindGroups.waterUniforms);
        pass.setBindGroup(1, this.bindGroups.waterSim);
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
