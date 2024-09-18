async function makeStlChart() {
    const stlData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-stl-prophet-forecast.json`);
    const timestamps = stlData.timestamps;
    const priceData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-price-data.json`);
    const titleText = `${baseCoinId.toUpperCase()} Price Seasonality via Seasonal-Trend Decomposition (LOESS)`;


    Highcharts.chart('stlContainer', {
        chart: {
            type: 'line',
            zoomType: 'x',
            backgroundColor: '#1a1a1a'
        },
        title: {
            text: titleText,
            style: {color: '#e3dddd'}
        },
        subtitle: {
            text: '<b>Trend</b>: Longer-term progression, ignoring short-term fluctuations. ' +
                '<b>Seasonal</b>: Regular pattern in the data that repeats over a specific period. ' +
                '<b>Residual</b>: Noise and irregularities that cannot be explained otherwise.',
            style: {color: '#e3dddd'}
        },
        xAxis: {
            type: 'datetime',
            title: {
                text: 'Date',
                style: {color: '#fff'}
            },
            labels: {
                style: {color: '#fff'}
            },
        },
        yAxis: {
            title: {
                text: 'Price',
                style: {color: '#fff'}
            },
            labels: {
                style: {color: '#fff'}
            },
            gridLineWidth: 0,
            minorGridLineWidth: 0
        },
        series: [{
            name: 'Original Price',
            data: priceData.map((item, index) => [item[0], item[1]]),
            tooltip: {
                valueDecimals: 5
            }
        }, {
            name: 'Trend',
            data: stlData.trend.map((value, index) => [timestamps[index], value]),
            tooltip: {
                valueDecimals: 5
            }
        }, {
            name: 'Seasonal',
            data: stlData.seasonal.map((value, index) => [timestamps[index], value]),
            tooltip: {
                valueDecimals: 5
            }
        }, {
            name: 'Residual',
            data: stlData.residual.map((value, index) => [timestamps[index], value]),
            tooltip: {
                valueDecimals: 5
            }
        }],
        legend: {
            itemStyle: {color: '#fff'}
        },
        plotOptions: {
            series: {
                marker: {
                    enabled: false,
                },
            },
        }
    });
}

// makeStlChart()
