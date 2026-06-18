import { ShallowWaterSimulator } from './shallow-water-simulator.js';
import { WaterRenderer } from './water-renderer.js';
import { InputController } from './input-controller.js';

class FluidSimulationApp {
    constructor() {
        this.canvas = document.getElementById('gpu-canvas');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.loadingStatus = document.getElementById('loading-status');
        
        this.device = null;
        this.gpuContext = null;
        this.simulator = null;
        this.renderer = null;
        this.input = null;
        
        this.isPaused = false;
        this.frameCount = 0;
        this.fps = 0;
        this.lastFpsTime = performance.now();
        this.frameTimes = [];
        this.computeTimes = [];
        this.renderTimes = [];
    }

    async init() {
        try {
            this._updateLoading('正在检查WebGPU支持...');
            
            if (!navigator.gpu) {
                throw new Error('当前浏览器不支持WebGPU。请使用Chrome 113+或Edge 113+。');
            }

            this._updateLoading('请求GPU适配器...');
            const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
            if (!adapter) throw new Error('无法获取GPU适配器。请更新显卡驱动。');
            
            const info = await adapter.requestAdapterInfo?.() || adapter.info || {};
            this._updateLoading(`GPU: ${info.description || info.device || '检测中...'}`);

            this._updateLoading('创建WebGPU设备...');
            this.device = await adapter.requestDevice();
            this.device.lost.then(info => console.error('WebGPU设备丢失:', info.message));

            this._updateLoading('配置画布...');
            this.gpuContext = this.canvas.getContext('webgpu');
            const format = navigator.gpu.getPreferredCanvasFormat();
            
            const dpr = Math.min(window.devicePixelRatio, 2);
            this.canvas.width = Math.floor(window.innerWidth * dpr);
            this.canvas.height = Math.floor(window.innerHeight * dpr);
            
            this.gpuContext.configure({
                device: this.device,
                format,
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
                alphaMode: 'premultiplied'
            });

            this._updateLoading('创建浅水方程求解器 (1024×1024网格)...');
            this.simulator = new ShallowWaterSimulator(this.device);
            await this.simulator.init();

            this._updateLoading('创建水面渲染管线 (256×256网格 + MSAA)...');
            this.renderer = new WaterRenderer(this.device, this.gpuContext, this.simulator);
            await this.renderer.init();

            this._updateLoading('初始化交互控制器...');
            this.input = new InputController(this.canvas, this.simulator, this.renderer);

            this._setupUI();
            
            window.addEventListener('resize', () => {
                const d = Math.min(window.devicePixelRatio, 2);
                this.canvas.width = Math.floor(window.innerWidth * d);
                this.canvas.height = Math.floor(window.innerHeight * d);
                this.gpuContext.configure({
                    device: this.device, format,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT,
                    alphaMode: 'premultiplied'
                });
            });

            this._updateLoading('启动模拟循环...');
            await new Promise(r => setTimeout(r, 200));
            this._hideLoading();
            this._start();

            console.log('✅ 流体模拟器初始化成功！');

        } catch (error) {
            console.error('初始化失败:', error);
            this._showError(error.message);
        }
    }

    _setupUI() {
        const bind = (id, valueId, callback) => {
            const el = document.getElementById(id);
            const vEl = document.getElementById(valueId);
            if (!el) return;
            el.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                if (vEl) vEl.textContent = Number.isInteger(v) ? v : v.toFixed(v < 0.01 ? 4 : v < 1 ? 2 : 1);
                callback(v);
            });
        };

        bind('gravity', 'gravity-value', v => this.simulator.setParam('gravity', v));
        bind('h0', 'h0-value', v => this.simulator.setParam('h0', v));
        bind('viscosity', 'viscosity-value', v => this.simulator.setParam('viscosity', v));
        bind('cfl', 'cfl-value', v => this.simulator.setParam('cfl', v));
        bind('amplify', 'amplify-value', v => this.renderer.setRenderParam('heightAmplify', v));
        bind('fresnel', 'fresnel-value', v => this.renderer.setRenderParam('fresnelPower', v));
        bind('refract', 'refract-value', v => this.renderer.setRenderParam('refractStrength', v));

        document.getElementById('show-wireframe')?.addEventListener('change', (e) => {
            this.renderer.setRenderParam('showWireframe', e.target.checked ? 1 : 0);
        });

        document.querySelectorAll('input[name="mode"]').forEach(r => {
            r.addEventListener('change', (e) => this.input.setMode(e.target.value));
        });

        bind('brush-size', 'brush-size-value', v => this.input.setBrushSize(v));
        bind('brush-strength', 'brush-strength-value', v => this.input.setBrushStrength(v));

        document.getElementById('btn-pause')?.addEventListener('click', function() {
            this.isPaused = !this.isPaused;
            this.textContent = this.isPaused ? '继续' : '暂停';
        }.bind(this));

        document.getElementById('btn-reset')?.addEventListener('click', () => this.simulator.reset());

        document.getElementById('btn-splash')?.addEventListener('click', () => {
            this.simulator.createSplash(0.5, 0.5, 2.0, 0.08);
            setTimeout(() => this.simulator.createSplash(0.3, 0.4, 1.5, 0.06), 100);
            setTimeout(() => this.simulator.createSplash(0.7, 0.6, 1.5, 0.06), 200);
        });
    }

    _start() {
        this._running = true;
        this._loop();
    }

    _loop() {
        if (!this._running) return;
        
        const frameStart = performance.now();
        
        if (!this.isPaused) {
            const computeStart = performance.now();
            this.simulator.step(2);
            this.computeTimes.push(performance.now() - computeStart);
        }
        
        const renderStart = performance.now();
        const canvasTexture = this.gpuContext.getCurrentTexture().createView();
        this.renderer.render(canvasTexture);
        this.renderTimes.push(performance.now() - renderStart);
        
        this.frameTimes.push(performance.now() - frameStart);
        this.frameCount++;
        
        if (performance.now() - this.lastFpsTime >= 500) {
            const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
            const avgFrame = avg(this.frameTimes);
            this.fps = Math.round(1000 / avgFrame);
            
            document.getElementById('fps').textContent = this.fps;
            document.getElementById('frame-time').textContent = avgFrame.toFixed(1);
            document.getElementById('compute-time').textContent = avg(this.computeTimes).toFixed(1);
            document.getElementById('render-time').textContent = avg(this.renderTimes).toFixed(1);
            
            const stats = this.simulator.getStats();
            document.getElementById('mem-usage').textContent = stats.usedMB;
            document.getElementById('mem-total').textContent = stats.totalMB;
            
            this.frameTimes = [];
            this.computeTimes = [];
            this.renderTimes = [];
            this.lastFpsTime = performance.now();
        }
        
        requestAnimationFrame(() => this._loop());
    }

    _updateLoading(status) { if (this.loadingStatus) this.loadingStatus.textContent = status; }
    _hideLoading() { if (this.loadingOverlay) this.loadingOverlay.classList.add('hidden'); }

    _showError(message) {
        if (!this.loadingOverlay) return;
        this.loadingOverlay.innerHTML = `
            <div class="loading-content">
                <h2 style="color:#ff5252;-webkit-text-fill-color:#ff5252;">初始化失败</h2>
                <p style="color:#ff8a80;margin-top:16px;">${message}</p>
                <div style="margin-top:24px;padding:16px;background:rgba(255,82,82,0.1);border-radius:8px;text-align:left;">
                    <p style="font-size:12px;color:#90a0b0;line-height:1.8;">
                        <strong style="color:#4fc3f7;">解决方案：</strong><br>
                        1. 使用 Chrome 113+ 或 Edge 113+ 浏览器<br>
                        2. 地址栏输入 chrome://flags/ 搜索 "WebGPU" 设为 Enabled<br>
                        3. 重启浏览器后重试<br>
                        4. 确保显卡驱动已更新到最新版本
                    </p>
                </div>
            </div>`;
    }
}

window.addEventListener('DOMContentLoaded', () => new FluidSimulationApp().init());
