document.addEventListener('DOMContentLoaded', () => {
    const indicators = [
        "SMA", "EMA", "RSI", "Upper Band", "Lower Band", "ATR",
        "Stochastic %K", "Stochastic %D", "OBV", "CCI", "VWAP",
        "MACD", "MACD Signal", "MACD Hist", "ADX", "ADX+DI", "ADX-DI",
        "MFI", "TRIX", "TRIX Signal", "TSI", "TSI Signal", "Williams %R", "Ultimate Oscillator"
    ];

    const checkboxContainer = document.getElementById('checkboxContainer');

    indicators.forEach((indicator, index) => {
        const checkboxId = `btncheck${index + 1}`;

        // Create the input checkbox element
        const inputElement = document.createElement('input');
        inputElement.type = 'checkbox';
        inputElement.className = 'btn-check indicator-toggle';
        inputElement.id = checkboxId;
        inputElement.setAttribute('data-indicator', indicator);
        inputElement.autocomplete = 'off';
        inputElement.checked = true;

        // Create the label element
        const labelElement = document.createElement('label');
        labelElement.className = 'btn btn-outline-primary btn-sm';
        labelElement.setAttribute('for', checkboxId);
        labelElement.textContent = indicator;

        // Append the checkbox and label to the container
        checkboxContainer.appendChild(inputElement);
        checkboxContainer.appendChild(labelElement);
    });

    loadIndicatorSettings();
    fetchDataAndUpdateChart(); // Initial fetch and update
    startAutoReload(); // Start the auto-reload logic

    document.querySelectorAll('.indicator-toggle').forEach(checkbox => {
        checkbox.addEventListener('change', (event) => {
            const indicatorName = event.target.dataset.indicator;
            saveIndicatorSettings();
            applyIndicatorSettings();
        });
    });
});

function saveIndicatorSettings() {
    const indicatorSettings = {};
    document.querySelectorAll('.indicator-toggle').forEach(checkbox => {
        indicatorSettings[checkbox.dataset.indicator] = checkbox.checked;
    });
    localStorage.setItem('indicatorSettings', JSON.stringify(indicatorSettings));
}

function loadIndicatorSettings() {
    const indicatorSettings = JSON.parse(localStorage.getItem('indicatorSettings'));
    if (indicatorSettings) {
        document.querySelectorAll('.indicator-toggle').forEach(checkbox => {
            const indicator = checkbox.dataset.indicator;
            if (indicatorSettings.hasOwnProperty(indicator)) {
                checkbox.checked = indicatorSettings[indicator];
                applyIndicatorSettings();
            }
        });
    }
}

function applyIndicatorSettings() {
    const indicatorSettings = JSON.parse(localStorage.getItem('indicatorSettings'));
    if (indicatorSettings && chart) {
        chart.series.forEach(series => {
            if (indicatorSettings.hasOwnProperty(series.options.id)) {
                console.log(`Applying visibility for series: ${series.options.id}`);
                series.setVisible(indicatorSettings[series.options.id], false);
            }
        });
        chart.redraw();
    }
}

function flashScreen(colorClass) {
    document.body.classList.add(colorClass);
    setTimeout(() => {
        document.body.classList.remove(colorClass);
    }, 10000); // 10 repetitions of 500ms animation
}

function startAutoReload() {
    setInterval(() => {
        if (document.visibilityState === 'visible') {
            fetchDataAndUpdateChart();
        }
    }, 10000);
}

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        fetchDataAndUpdateChart();
    }
});
