function rosslerControls(rossler) {
    this.rossler = rossler;
    this.listeners = [];

    this.chain = [];
    this.button = null;

    var sigma = this.bind('#sigma', '#sigma-label', function(value) {
        return rossler.params.sigma = value;
    })(rossler.params.sigma);
    var beta = this.bind('#beta', '#beta-label', function(value) {
        return rossler.params.beta = value;
    })(rossler.params.beta);
    var rho = this.bind('#rho', '#rho-label', function(value) {
        return rossler.params.sigma = value;
    })(rossler.params.rho);

    this.set_sigma = function(value) {
        rossler.params.sigma = value;
        sigma(value);
    };
    this.set_beta = function(value) {
        rossler.params.beta = value;
        beta(value);
    };
    this.set_rho = function(value) {
        rossler.params.rho = value;
        rho(value);
    };

    this.set_length = this.bind('#length', '#length-label', function(value) {
        var length = Math.pow(2, parseFloat(value));
        rossler.length = length;
        return length;
    })({
        input: Math.log(rossler.length) * Math.LOG2E,
        label: rossler.length
    });

    var canvas = rossler.gl.canvas;
    canvas.addEventListener('mousedown', function(e) {
        if (e.buttons) {
            this.button = e.buttons & 4 ? 'middle' : 'left';
            this.push({x: e.pageX, y: e.pageY});
        }
    }.bind(this));
    canvas.addEventListener('mousemove', function(e) {
        e.preventDefault();
        if (this.button) {
            this.push({x: e.pageX, y: e.pageY});
            var shift = e.shiftKey;
            var delta = null;
            if (this.button === 'middle') {
                /* Translate */
                delta = this.delta(1 / 40);
                if (shift)
                    this.rossler.display.translation[2] += delta.y;
                else
                    this.rossler.display.translation[0] += -delta.x;
                this.rossler.display.translation[1] += delta.y;
            } else {
                this.rossler.display.rotationd[0] = 0;
                this.rossler.display.rotationd[1] = 0;
                this.rossler.display.rotationd[2] = 0;
                delta = this.delta(1 / 20);
                if (e.shift)
                    rossler.display.rotation[1] += -delta.x;
                else
                    rossler.display.rotation[2] += delta.x;
                rossler.display.rotation[0] += delta.y;
            }
        }
    }.bind(this));
    canvas.addEventListener('mouseup', function(e) {
        e.preventDefault();
        this.push({x: e.pageX, y: e.pageY});
        if (this.button != 'middle') {
            var delta = this.delta(1 / 20);
            if (e.shift)
                rossler.display.rotationd[1] = -delta.x;
            else
                rossler.display.rotationd[2] = delta.x;
            rossler.display.rotationd[0] = delta.y;
        }
        this.chain.length = 0;
        this.button = null;
    }.bind(this));

    canvas.addEventListener('DOMMouseScroll', function(e) {
        e.preventDefault();
        this.rossler.display.scale *= e.detail > 0 ? 0.95 : 1.1;
    }.bind(this));
    canvas.addEventListener('mousewheel', function(e) {
        e.preventDefault();
        this.rossler.display.scale *= e.wheelDelta < 0 ? 0.95 : 1.1;
    }.bind(this));

    window.addEventListener('keypress', function(e) {
        if (e.which == 'a'.charCodeAt(0))
            this.add();
        else if (e.which == 'c'.charCodeAt(0))
            this.clone();
        else if (e.which == 'C'.charCodeAt(0))
            this.clear();
        else if (e.which == ' '.charCodeAt(0))
            this.pause();
        else if (e.which == 'h'.charCodeAt(0))
            this.rossler.display.draw_heads = !this.rossler.display.draw_heads;
        else if (e.which == 'd'.charCodeAt(0))
            this.rossler.display.damping = !this.rossler.display.damping;
        else if (e.which == '['.charCodeAt(0) && rossler.length > 4)
            this.set_length({
                input: Math.log(rossler.length /= 2) * Math.LOG2E,
                label: rossler.length
            });
        else if (e.which == ']'.charCodeAt(0) && rossler.length < 32768)
            this.set_length({
                input: Math.log(rossler.length *= 2) * Math.LOG2E,
                label: rossler.length
            });
   }.bind(this));

    window.addEventListener('touchmove', function(e) {
        e.preventDefault();
        this.push({x: e.touches[0].clientX, y: e.touches[0].clientY});
        var delta = this.delta(1 / 20);
        this.rossler.display.rotationd[0] = 0;
        this.rossler.display.rotationd[1] = 0;
        this.rossler.display.rotationd[2] = 0;
        this.rossler.display.rotation[2] += delta.x;
        this.rossler.display.rotation[0] += delta.y;
    }.bind(this));
    window.addEventListener('touchend', function(e) {
        var delta = this.delta(1 / 10); // small for more playfulness
        if (delta.x || delta.y) {
            this.rossler.display.rotationd[2] = delta.x;
            this.rossler.display.rotationd[0] = delta.y;
        } else {
            this.add();
        }
    }.bind(this));
}

rosslerControls.prototype.push = function(e) {
    e.t = Date.now();
    this.chain.push(e);
};

rosslerControls.prototype.delta = function(scale) {
    scale /= 1000;
    var stop = this.chain.length - 1;
    var start = stop;
    var dt = 0;
    while (dt === 0 && stop > 1) {
        stop--;
        dt = (this.chain[start].t - this.chain[stop].t) / 1000;
    }
    if (dt === 0)
        return {x: 0, y: 0}; // no delta!
    return {
        x: (this.chain[stop].x - this.chain[start].x) * scale / dt,
        y: (this.chain[stop].y - this.chain[start].y) * scale / dt
    };
};

rosslerControls.prototype.add = function() {
    this.rossler.add(Rossler.generate());
    for (var n = 0; n < this.listeners.length; n++)
        this.listeners[n]();
};

rosslerControls.prototype.clone = function() {
    var i = Math.floor(Math.random() * this.rossler.solutions.length);
    var s = this.rossler.solutions[i].slice(0);
    s[0] += (Math.random() - 0.5) / 10000;
    s[1] += (Math.random() - 0.5) / 10000;
    s[2] += (Math.random() - 0.5) / 10000;
    this.rossler.add(s);
    for (var n = 0; n < this.listeners.length; n++)
        this.listeners[n]();
};

rosslerControls.prototype.clear = function() {
    this.rossler.empty();
    for (var n = 0; n < this.listeners.length; n++)
        this.listeners[n]();
};

rosslerControls.prototype.pause = function() {
    this.rossler.params.paused = !this.rossler.params.paused;
};

rosslerControls.prototype.bind = function(input_selector, label_selector, f) {
    var input = document.querySelector(input_selector);
    var label = document.querySelector(label_selector);
    var handler = function(e) {
        label.textContent = f(parseFloat(input.value));
    };
    input.addEventListener('input', handler);
    input.addEventListener('change', handler);
    return function self(value) {
        if (typeof value === 'number') {
            input.value = value;
            label.textContent = value;
        } else {
            input.value = value.input;
            label.textContent = value.label;
        }
        return self;
    };
};
