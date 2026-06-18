import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 8080;
const HOST = 'localhost';

const MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.mjs': 'application/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.wgsl': 'text/plain',
    '.map': 'application/json'
};

const server = http.createServer((req, res) => {
    let filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
    
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    
    fs.stat(filePath, (err, stats) => {
        if (err || !stats.isFile()) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end('404 Not Found');
            return;
        }
        
        res.writeHead(200, {
            'Content-Type': contentType,
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp'
        });
        
        const readStream = fs.createReadStream(filePath);
        readStream.pipe(res);
        
        readStream.on('error', (streamErr) => {
            console.error('Stream error:', streamErr);
            res.writeHead(500);
            res.end('Server Error');
        });
    });
});

server.listen(PORT, HOST, () => {
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🌊 WebGPU 浅水方程流体模拟器 启动成功!                      ║
║                                                              ║
║   📡 本地服务器地址:  http://${HOST}:${PORT}                    ║
║                                                              ║
║   🖥️  推荐使用:                                              ║
║      • Chrome 113+ (推荐)                                   ║
║      • Edge 113+                                            ║
║      • Safari 17+                                           ║
║                                                              ║
║   ⚙️  如WebGPU不可用, 请在浏览器地址栏输入:                    ║
║      chrome://flags/ → 启用 WebGPU                           ║
║                                                              ║
║   🎮 操作说明:                                              ║
║      • 左键拖拽: 生成波浪 / 修改地形 / 放置障碍物            ║
║      • 右键拖拽: 旋转相机                                   ║
║      • 滚轮: 缩放视图                                       ║
║                                                              ║
║   按 Ctrl+C 停止服务器                                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
`);
});

server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`❌ 端口 ${PORT} 已被占用, 请尝试其他端口: node server.js`);
        process.exit(1);
    }
    console.error('服务器错误:', err);
});

process.on('SIGINT', () => {
    console.log('\n\n👋 服务器已停止');
    server.close();
    process.exit(0);
});
