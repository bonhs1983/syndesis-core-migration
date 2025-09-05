// Main JavaScript file for the FractalAI Training Pipeline

class PipelineDashboard {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupAutoRefresh();
    }

    setupEventListeners() {
        // Handle form submissions with loading states
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    const originalText = submitBtn.innerHTML;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
                    submitBtn.disabled = true;
                    
                    // Re-enable after 3 seconds (in case of issues)
                    setTimeout(() => {
                        submitBtn.innerHTML = originalText;
                        submitBtn.disabled = false;
                    }, 3000);
                }
            });
        });

        // Handle navigation highlighting
        this.highlightCurrentNav();
    }

    setupAutoRefresh() {
        // Auto-refresh dashboard stats every 30 seconds
        if (window.location.pathname === '/') {
            setInterval(() => {
                this.refreshDashboardStats();
            }, 30000);
        }
    }

    highlightCurrentNav() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            const linkPath = new URL(link.href).pathname;
            if (linkPath === currentPath) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    refreshDashboardStats() {
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                // Update stats cards
                const statsCards = document.querySelectorAll('.card-body h2');
                if (statsCards.length >= 3) {
                    statsCards[0].textContent = data.total_interactions;
                    statsCards[1].textContent = data.unprocessed_interactions;
                    statsCards[2].textContent = data.training_jobs;
                }
            })
            .catch(error => {
                console.error('Error refreshing stats:', error);
            });
    }

    showNotification(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            <i data-feather="${type === 'error' ? 'alert-circle' : 'check-circle'}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Replace feather icons
        feather.replace();
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Utility functions
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        dashboard.showNotification('Copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy: ', err);
        dashboard.showNotification('Failed to copy to clipboard', 'error');
    });
}

function downloadFile(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new PipelineDashboard();
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

// Handle page visibility change to pause/resume auto-refresh
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('Page hidden - pausing auto-refresh');
    } else {
        console.log('Page visible - resuming auto-refresh');
        if (window.dashboard && window.location.pathname === '/') {
            window.dashboard.refreshDashboardStats();
        }
    }
});
