// The base coin for which to display self-contained data, like price, volume, and volatility

// e.g. `bitcoin`
let baseCoinId = document.getElementById('coinDropdownButton').textContent.at(-1).toLowerCase();

// e.g. `$BTC`
let baseCoinSymbol = document.getElementById('coinDropdownButton').textContent;

// The target coin for which to display the base coin's correlation data
const TARGET_COIN_ID = 'bitcoin';

const PATH_PREFIX = '../../data/generated';

const coinDropdownItems = document.querySelectorAll('.dropdown-item');

async function fetchLocalData(dataFilePath) {
    try {
        const response = await fetch(dataFilePath);
        return await response.json();
    } catch (error) {
        console.error("Failed to fetch local data for", dataFilePath);
        throw new Error(error);
    }
}

function renderAllCharts() {
     makeMonteCarloChart();
     makeCombinedChart();
     makeStlChart();
     makeFFTChart();
     makeStlProphetForecast();
     makePriceSwingsChart();
     makePriceSwingsHeatmap();
     makePriceSwingsScatterPlot();
     makeWaveletTransformChart();
     makeBestDaysToBuyHeatmap();
}

coinDropdownItems.forEach(item => {
    item.addEventListener('click', async (event) => {
        event.preventDefault();

        baseCoinId = event.target.getAttribute('data-coin-id');
        baseCoinSymbol = event.target.textContent;
        document.getElementById('coinDropdownButton').textContent = baseCoinSymbol;

        renderAllCharts();
    });
})
