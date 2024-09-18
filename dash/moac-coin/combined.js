/**
 * Make a unified chart for the coin, showing price, volume, volatility, and correlation data
 *
 * @returns {Promise<void>}
 */
async function makeCombinedChart() {
    const priceData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-price-data.json`);
    const volumeData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-volume-data.json`);
    const volatilityData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-volatility-data.json`);
    const priceCorrData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-price-corr.json`);
    const volumeCorrData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-volume-corr.json`);

    const CHART_STYLE = {
        chart: {
            type: 'line',
            backgroundColor: '#1a1a1a',
            zoomType: 'x',
        },
        xAxis: {
            type: 'datetime',
            title: {
                text: 'Date',
                style: {color: 'white'}
            },
            labels: {style: {color: 'white'}}
        },
        legend: {
            enabled: true,
            itemStyle: {color: '#fff'}
        },
        plotOptions: {
            line: {enableMouseTracking: true},
            series: {
                turboThreshold: 0,
                marker: {
                    enabled: false,
                    lineWidth: 1,
                    lineColor: '#ffffff'
                }
            }
        }
    };

    Highcharts.chart('priceVolContainer', {
        ...CHART_STYLE,
        title: {
            text: `${baseCoinSymbol.toUpperCase()} | Price, Volume, Volatility, and Correlations`,
            style: {color: '#e3dddd'}
        },
        yAxis: [
            {
                title: {
                    text: 'Price (USD)',
                    style: {color: '#75a1fa'}
                },
                labels: {
                    style: {color: '#75a1fa'}
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0,
                opposite: true
            }, {
                title: {
                    text: 'Volume',
                    style: {color: '#00ff00'}
                },
                labels: {
                    style: {color: '#00ff00'}
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0
            },
            {
                title: {
                    text: 'Volatility (%)',
                    style: {color: '#de8ece'}
                },
                labels: {
                    style: {color: '#de8ece'}
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0,
                opposite: true
            },
            {
                title: {
                    text: 'Price Corr Coefficient',
                    style: {color: 'orange'}
                },
                labels: {
                    style: {color: 'orange'}
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0,
                opposite: true
            },
            {
                title: {
                    text: 'Vol Corr Coefficient',
                    style: {color: '#ff3361'}
                },
                labels: {
                    style: {color: '#ff3361'}
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0,
                opposite: true
            },
        ],
        series: [
            {
                name: 'Price',
                type: 'line',
                yAxis: 0,
                data: priceData.map(item => [item[0], item[1]]),
                color: '#75a1fa'
            }, {
                name: 'Volume',
                type: 'column',
                yAxis: 1,
                data: volumeData.map(item => [item[0], item[1]]),
                color: '#00ff00'
            },
            {
                name: 'Volatility',
                type: 'line',
                yAxis: 2,
                data: processVolatilityData(volatilityData).find(item => item.name.includes(baseCoinId)).data,
                color: '#de8ece'
            }, {
                name: 'Price Corr w/ ' + TARGET_COIN_ID.toUpperCase(),
                type: 'line',
                yAxis: 3,
                data: processCorrData(priceCorrData)[0].data,
                color: 'gold'
            }, {
                name: 'Vol Corr w/ ' + TARGET_COIN_ID.toUpperCase(),
                type: 'line',
                yAxis: 4,
                data: processCorrData(volumeCorrData)[0].data,
                color: '#ff3361'
            },
        ]
    });
}

function processVolatilityData(data) {
    return Object.keys(data).map(key => ({
        name: `${key} (volatility: ${data[key].volatility.toFixed(2)}%)`,
        data: data[key].returns_data.map(item => [item[0], item[1] * 100])
    }));
}

/**
 * Process correlation data for the coin
 *
 * @param data {any} - The correlation data
 * @returns {any[]} - The processed data
 */
function processCorrData(data) {
    const seriesMap = new Map();

    data.data.forEach(dayData => {
        dayData.correlations.forEach(coinData => {
            if (coinData.coin !== TARGET_COIN_ID) {
                return;
            }

            if (!seriesMap.has(coinData.coin)) {
                seriesMap.set(coinData.coin, {name: coinData.coin, data: []});
            }
            const date = new Date(dayData.end_date).getTime();
            seriesMap.get(coinData.coin).data.push({x: date, y: coinData.correlation});
        });
    });

    return Array.from(seriesMap.values());
}
