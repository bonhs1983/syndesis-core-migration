// Particle system for enhanced visual effects
class ParticleSystem {
    constructor(container) {
        this.container = container;
        this.particles = [];
        this.maxParticles = 50;
        this.init();
    }

    init() {
        for (let i = 0; i < this.maxParticles; i++) {
            this.createParticle();
        }
        this.animate();
    }

    createParticle() {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random positioning
        const x = Math.random() * window.innerWidth;
        const y = window.innerHeight + 10;
        const size = Math.random() * 4 + 2;
        const duration = Math.random() * 20 + 10;
        const delay = Math.random() * 5;
        
        particle.style.left = x + 'px';
        particle.style.top = y + 'px';
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        particle.style.animationDuration = duration + 's';
        particle.style.animationDelay = delay + 's';
        
        this.container.appendChild(particle);
        this.particles.push(particle);
        
        // Remove particle after animation
        setTimeout(() => {
            if (particle.parentNode) {
                particle.parentNode.removeChild(particle);
                this.particles = this.particles.filter(p => p !== particle);
                this.createParticle(); // Create a new one
            }
        }, (duration + delay) * 1000);
    }

    animate() {
        // Additional animation logic if needed
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize particles when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const particleContainer = document.querySelector('.particles');
    if (particleContainer) {
        new ParticleSystem(particleContainer);
    }
});

// Logo animation effects
function initLogoEffects() {
    const logo = document.querySelector('.logo');
    if (logo) {
        logo.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.1) rotate(5deg)';
        });
        
        logo.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1) rotate(0deg)';
        });
    }
}

// Enhanced button effects
function initButtonEffects() {
    const buttons = document.querySelectorAll('.btn-enhanced');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px) scale(1.05)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
        
        button.addEventListener('mousedown', function() {
            this.style.transform = 'translateY(-1px) scale(0.98)';
        });
        
        button.addEventListener('mouseup', function() {
            this.style.transform = 'translateY(-3px) scale(1.05)';
        });
    });
}

// Initialize all effects
document.addEventListener('DOMContentLoaded', function() {
    initLogoEffects();
    initButtonEffects();
    
    // Add floating elements
    createFloatingElements();
});

function createFloatingElements() {
    const hero = document.querySelector('.hero-enhanced');
    if (!hero) return;
    
    const symbols = ['🧠', '⚡', '🔮', '✨', '🚀', '💫'];
    
    for (let i = 0; i < 6; i++) {
        const element = document.createElement('div');
        element.className = 'floating-element';
        element.innerHTML = symbols[i % symbols.length];
        element.style.fontSize = '2rem';
        element.style.opacity = '0.6';
        element.style.pointerEvents = 'none';
        
        // Random positioning
        const x = Math.random() * 80 + 10; // 10-90%
        const y = Math.random() * 80 + 10; // 10-90%
        
        element.style.left = x + '%';
        element.style.top = y + '%';
        element.style.animationDelay = (i * 2) + 's';
        
        hero.appendChild(element);
    }
}