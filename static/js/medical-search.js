// Medical Search Functionality
class MedicalRiskApp {
    constructor() {
        this.currentProfile = null;
        this.init();
    }

    async init() {
        await this.loadHealthProfile();
        this.setupEventListeners();
        this.setupCitySuggestions();
    }

    async loadHealthProfile() {
        try {
            // In real implementation, this would fetch from server
            this.currentProfile = this.getDefaultProfile();
            this.updateProfileSummary();
        } catch (error) {
            console.error('Error loading health profile:', error);
        }
    }

    getDefaultProfile() {
        return {
            age: 30,
            bmi: 22.5,
            smoker: false,
            former_smoker: false,
            outdoor_worker: false,
            regular_exercise: true,
            urban_resident: true,
            // Medical conditions would be loaded from server
        };
    }

    updateProfileSummary() {
        const summaryEl = document.getElementById('profileSummary');
        if (!summaryEl) return;

        const profile = this.currentProfile;
        let summary = '';

        if (profile.age) {
            summary += `<strong>Age:</strong> ${profile.age} years<br>`;
        }

        // Add health conditions summary
        const conditions = this.getActiveConditions(profile);
        if (conditions.length > 0) {
            summary += `<strong>Conditions:</strong> ${conditions.join(', ')}`;
        } else {
            summary += '<em>No specific health conditions reported</em>';
        }

        summaryEl.innerHTML = summary;
    }

    getActiveConditions(profile) {
        const medicalConditions = [
            'asthma', 'copd', 'heart_disease', 'hypertension', 'diabetes',
            'pregnancy', 'child_under_5', 'child_5_12', 'elderly_65_75', 'elderly_over_75'
        ];

        return medicalConditions.filter(condition => profile[condition]);
    }

    setupEventListeners() {
        const form = document.getElementById('medicalSearchForm');
        if (form) {
            form.addEventListener('submit', (e) => this.handleSearch(e));
        }

        // City suggestion clicks
        document.querySelectorAll('.suggestion-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                document.getElementById('cityInput').value = tag.textContent;
                this.handleSearch(new Event('submit'));
            });
        });
    }

    setupCitySuggestions() {
        const input = document.getElementById('cityInput');
        const suggestions = document.getElementById('citySuggestions');

        if (input && suggestions) {
            input.addEventListener('input', () => {
                const value = input.value.toLowerCase();
                suggestions.style.display = value ? 'block' : 'none';
            });

            // Hide suggestions when clicking outside
            document.addEventListener('click', (e) => {
                if (!input.contains(e.target) && !suggestions.contains(e.target)) {
                    suggestions.style.display = 'none';
                }
            });
        }
    }

    async handleSearch(e) {
        e.preventDefault();
        
        const cityInput = document.getElementById('cityInput');
        const city = cityInput.value.trim();

        if (!city) {
            alert('Please enter a city name');
            return;
        }

        this.showLoading(true);
        
        try {
            const response = await this.fetchMedicalRiskAssessment(city);
            this.displayMedicalResults(response, city);
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async fetchMedicalRiskAssessment(city) {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                city: city,
                profile: this.currentProfile
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Assessment failed');
        }

        return await response.json();
    }

    displayMedicalResults(data, city) {
        this.updateRiskMeter(data.risk_score, data.risk_level);
        this.updateVulnerabilityAnalysis(data.user_vulnerabilities);
        this.updateMedicalAdvice(data.advice);
        this.updatePollutionData(data.pollution_data, data.source_info);
        
        document.getElementById('resultCity').textContent = city;
        document.getElementById('medicalResults').style.display = 'block';
    }

    updateRiskMeter(score, level) {
        const gauge = document.getElementById('gaugeFillMedical');
        const scoreEl = document.getElementById('riskScoreMedical');
        const badge = document.getElementById('riskBadge');

        if (gauge) gauge.style.width = `${Math.min(score * 10, 100)}%`;
        if (scoreEl) scoreEl.textContent = score.toFixed(1);
        if (badge) {
            badge.textContent = level;
            badge.setAttribute('data-level', level.toLowerCase().replace(' ', '-'));
        }
    }

    updateVulnerabilityAnalysis(vulnerabilities) {
        const grid = document.getElementById('vulnerabilityGrid');
        if (!grid) return;

        if (!vulnerabilities || vulnerabilities.length === 0) {
            grid.innerHTML = '<div class="vulnerability-item">No specific vulnerabilities identified</div>';
            return;
        }

        grid.innerHTML = vulnerabilities.map(vuln => 
            `<div class="vulnerability-item">${vuln}</div>`
        ).join('');
    }

    updateMedicalAdvice(advice) {
        const list = document.getElementById('medicalAdviceList');
        if (!list) return;

        if (!advice || advice.length === 0) {
            list.innerHTML = '<div class="advice-item">No specific recommendations at this time</div>';
            return;
        }

        list.innerHTML = advice.map(item => 
            `<div class="advice-item">${item}</div>`
        ).join('');
    }

    updatePollutionData(pollution, sourceInfo) {
        const grid = document.getElementById('pollutionGridMedical');
        const sourceEl = document.getElementById('dataSource');

        if (grid) {
            const pollutants = [
                { name: 'PM2.5', value: pollution.PM2_5 || pollution['PM2.5'], unit: 'μg/m³' },
                { name: 'PM10', value: pollution.PM10, unit: 'μg/m³' },
                { name: 'Ozone', value: pollution.O3, unit: 'μg/m³' },
                { name: 'CO', value: pollution.CO, unit: 'μg/m³' },
                { name: 'NO₂', value: pollution.NO2, unit: 'μg/m³' },
                { name: 'SO₂', value: pollution.SO2, unit: 'μg/m³' }
            ];

            grid.innerHTML = pollutants.map(p => `
                <div class="pollutant-item-medical">
                    <div class="pollutant-name">${p.name}</div>
                    <div class="pollutant-value-medical">${p.value || 0}</div>
                    <div class="pollutant-unit">${p.unit}</div>
                </div>
            `).join('');
        }

        if (sourceEl && sourceInfo) {
            sourceEl.innerHTML = `Data source: ${sourceInfo.name} • ${new Date(sourceInfo.timestamp).toLocaleString()}`;
        }
    }

    showLoading(show) {
        const loading = document.getElementById('loadingMedical');
        const form = document.getElementById('medicalSearchForm');
        
        if (loading) loading.style.display = show ? 'block' : 'none';
        if (form) form.style.display = show ? 'none' : 'block';
    }

    showError(message) {
        alert(`Medical assessment error: ${message}`);
    }
}

// Global functions for HTML onclick handlers
function newAssessment() {
    document.getElementById('medicalResults').style.display = 'none';
    document.getElementById('medicalSearchForm').style.display = 'block';
    document.getElementById('cityInput').value = '';
}

function skipProfile() {
    if (confirm('You can set up your health profile later. Continue with default assessment?')) {
        window.location.href = '/search';
    }
}

// Initialize app when DOM loads
document.addEventListener('DOMContentLoaded', () => {
    new MedicalRiskApp();
});