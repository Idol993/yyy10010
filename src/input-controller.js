export class InputController {
    constructor(canvas, simulator, renderer) {
        this.canvas = canvas;
        this.simulator = simulator;
        this.renderer = renderer;
        
        this.mode = 'wave';
        this.brushSize = 30;
        this.brushStrength = 0.5;
        this.isDragging = false;
        this.lastX = 0;
        this.lastY = 0;
        this.isRightDragging = false;
        this.mouseX = 0;
        this.mouseY = 0;
        this.shiftHeld = false;
        
        this._bindEvents();
    }

    _bindEvents() {
        this.canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
        this.canvas.addEventListener('mouseup', (e) => this._onMouseUp(e));
        this.canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => this._onMouseLeave());
        this.canvas.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
        
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        window.addEventListener('keydown', (e) => { if (e.key === 'Shift') this.shiftHeld = true; });
        window.addEventListener('keyup', (e) => { if (e.key === 'Shift') this.shiftHeld = false; });
        
        this.canvas.addEventListener('touchstart', (e) => this._onTouchStart(e), { passive: false });
        this.canvas.addEventListener('touchmove', (e) => this._onTouchMove(e), { passive: false });
        this.canvas.addEventListener('touchend', (e) => this._onTouchEnd(e));
    }

    _onMouseDown(e) {
        if (e.button === 0) {
            this.isDragging = true;
            this.lastX = e.clientX;
            this.lastY = e.clientY;
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
            this._applyInteraction(e.clientX, e.clientY, true);
        } else if (e.button === 2) {
            this.isRightDragging = true;
            this.lastX = e.clientX;
            this.lastY = e.clientY;
        }
    }

    _onMouseUp(e) {
        if (e.button === 0) {
            this.isDragging = false;
        } else if (e.button === 2) {
            this.isRightDragging = false;
        }
    }

    _onMouseMove(e) {
        this.mouseX = e.clientX;
        this.mouseY = e.clientY;
        
        if (this.isDragging) {
            this._applyInteraction(e.clientX, e.clientY, true);
        }
        
        if (this.isRightDragging) {
            const dx = e.clientX - this.lastX;
            const dy = e.clientY - this.lastY;
            this.renderer.rotateCamera(dx, dy);
            this.lastX = e.clientX;
            this.lastY = e.clientY;
        }
    }

    _onMouseLeave() {
        this.isDragging = false;
        this.isRightDragging = false;
    }

    _onWheel(e) {
        e.preventDefault();
        this.renderer.zoomCamera(e.deltaY);
    }

    _onTouchStart(e) {
        e.preventDefault();
        if (e.touches.length === 1) {
            const touch = e.touches[0];
            this.isDragging = true;
            this.lastX = touch.clientX;
            this.lastY = touch.clientY;
            this.mouseX = touch.clientX;
            this.mouseY = touch.clientY;
            this._applyInteraction(touch.clientX, touch.clientY, true);
        } else if (e.touches.length === 2) {
            this._lastPinchDist = this._pinchDistance(e.touches);
        }
    }

    _onTouchMove(e) {
        e.preventDefault();
        if (e.touches.length === 1 && this.isDragging) {
            const touch = e.touches[0];
            this.mouseX = touch.clientX;
            this.mouseY = touch.clientY;
            this._applyInteraction(touch.clientX, touch.clientY, true);
            
            const dx = touch.clientX - this.lastX;
            const dy = touch.clientY - this.lastY;
            if (Math.abs(dx) + Math.abs(dy) > 20) {
                this.renderer.rotateCamera(dx * 0.5, dy * 0.5);
                this.lastX = touch.clientX;
                this.lastY = touch.clientY;
            }
        } else if (e.touches.length === 2) {
            const dist = this._pinchDistance(e.touches);
            if (this._lastPinchDist) {
                const delta = this._lastPinchDist - dist;
                this.renderer.zoomCamera(delta * 0.5);
            }
            this._lastPinchDist = dist;
        }
    }

    _onTouchEnd(e) {
        this.isDragging = false;
        this._lastPinchDist = null;
    }

    _pinchDistance(touches) {
        const dx = touches[0].clientX - touches[1].clientX;
        const dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    _applyInteraction(clientX, clientY, active = false) {
        const rect = this.canvas.getBoundingClientRect();
        const normX = (clientX - rect.left) / rect.width;
        const normY = 1.0 - (clientY - rect.top) / rect.height;
        
        const radius = this.brushSize / this.simulator.gridSize;
        const strength = this.brushStrength;
        
        let mode = 0;
        if (this.mode === 'wave') mode = 0;
        else if (this.mode === 'terrain') mode = 1;
        else if (this.mode === 'obstacle') mode = 2;
        
        let direction = 1.0;
        if (this.mode === 'terrain' && this.shiftHeld) {
            direction = -1.0;
        }
        
        this.simulator.setInteraction(
            Math.max(0.01, Math.min(0.99, normX)),
            Math.max(0.01, Math.min(0.99, normY)),
            radius,
            strength,
            mode,
            active,
            direction
        );
    }

    setMode(mode) {
        this.mode = mode;
    }

    setBrushSize(size) {
        this.brushSize = size;
    }

    setBrushStrength(strength) {
        this.brushStrength = strength;
    }
}
