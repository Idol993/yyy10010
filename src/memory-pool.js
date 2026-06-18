export class GPUMemoryPool {
    constructor(device, initialMB = 256) {
        this.device = device;
        this.initialBytes = initialMB * 1024 * 1024;
        this.totalBytes = 0;
        this.usedBytes = 0;
        this.pools = new Map();
        this.freeBlocks = [];
        this.allocations = new Map();
        this._initPools();
    }

    _initPools() {
        const sizes = [
            64, 256, 1024, 4096, 16384, 65536,
            262144, 1048576, 4194304, 16777216, 67108864
        ];
        for (const size of sizes) {
            this.pools.set(size, []);
        }
    }

    _getPoolSize(size) {
        const sizes = Array.from(this.pools.keys()).sort((a, b) => a - b);
        for (const s of sizes) {
            if (s >= size) return s;
        }
        return Math.pow(2, Math.ceil(Math.log2(size)));
    }

    allocate(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
        const poolSize = this._getPoolSize(size);
        const pool = this.pools.get(poolSize);
        
        let buffer;
        if (pool && pool.length > 0) {
            buffer = pool.pop();
        } else {
            const alignedSize = Math.max(poolSize, 256);
            buffer = this.device.createBuffer({
                size: alignedSize,
                usage,
                mappedAtCreation: false
            });
            this.totalBytes += alignedSize;
        }
        
        this.usedBytes += poolSize;
        this.allocations.set(buffer, { poolSize, originalSize: size, usage });
        return buffer;
    }

    allocateTexture(width, height, format = 'r32float', usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT) {
        const key = `tex_${width}_${height}_${format}`;
        const poolSize = width * height * 4 * 2;
        
        if (!this.pools.has(key)) {
            this.pools.set(key, []);
        }
        const pool = this.pools.get(key);
        
        let texture;
        if (pool.length > 0) {
            texture = pool.pop();
        } else {
            texture = this.device.createTexture({
                size: [width, height],
                format,
                usage,
                dimension: '2d'
            });
            this.totalBytes += poolSize;
        }
        
        this.usedBytes += poolSize;
        this.allocations.set(texture, { key, poolSize, isTexture: true });
        return texture;
    }

    deallocate(resource) {
        const info = this.allocations.get(resource);
        if (!info) return;
        
        this.usedBytes -= info.poolSize;
        this.allocations.delete(resource);
        
        if (info.isTexture) {
            const pool = this.pools.get(info.key);
            if (pool) pool.push(resource);
        } else {
            const pool = this.pools.get(info.poolSize);
            if (pool) pool.push(resource);
        }
    }

    allocateWithData(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE) {
        const size = data.byteLength;
        const buffer = this.allocate(size, usage);
        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }

    async readBuffer(buffer, offset = 0, size = buffer.size) {
        const staging = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, offset, staging, 0, size);
        this.device.queue.submit([encoder.finish()]);
        
        await staging.mapAsync(GPUMapMode.READ);
        const result = new Uint8Array(staging.getMappedRange()).slice(0);
        staging.unmap();
        staging.destroy();
        
        return result;
    }

    async copyBufferToBufferAsync(src, dst, size) {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(src, 0, dst, 0, size);
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    getStats() {
        return {
            usedMB: (this.usedBytes / 1024 / 1024).toFixed(2),
            totalMB: (this.totalBytes / 1024 / 1024).toFixed(2),
            allocationCount: this.allocations.size,
            fragmentation: this.totalBytes > 0 ? ((this.totalBytes - this.usedBytes) / this.totalBytes * 100).toFixed(1) : 0
        };
    }

    destroy() {
        for (const [key, pool] of this.pools) {
            for (const resource of pool) {
                if (typeof resource.destroy === 'function') {
                    resource.destroy();
                }
            }
        }
        for (const [resource] of this.allocations) {
            if (typeof resource.destroy === 'function') {
                resource.destroy();
            }
        }
        this.pools.clear();
        this.allocations.clear();
        this.totalBytes = 0;
        this.usedBytes = 0;
    }
}
