async function makeStlProphetForecast() {
    const response = await fetch(`${PATH_PREFIX}/${baseCoinId}-stl-prophet-forecast-daily.json`);
    const data = await response.json();

    const series = [];

    // Process each forecast configuration
    Object.keys(data).forEach((config, index) => {
        const forecasts = data[config].map(item => ({
            x: new Date(item.ds).getTime(),
            y: item.yhat
        }));

        // Add forecast line series
        series.push({
            name: `Config ${index + 1}: (${config})`,
            type: 'line',
            data: forecasts.map(point => [point.x, point.y]),
            tooltip: {
                valueDecimals: 2
            },
            turboThreshold: 0 // Enable turbo mode for large data sets
        });
    });

    Highcharts.chart('stlProphetForecast', {
        chart: {
            backgroundColor: '#1a1a1a',
            zoomType: 'xy',
        },
        title: {
            text: `${baseCoinId.toUpperCase()} Price Forecast with STL Decomposition and Prophet`,
            style: {color: '#e3dddd'}
        },
        subtitle: {
            text: `${Object.keys(data).length} forecasts were generated using a combination of different parameters.`,
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
                text: 'Price (USD)',
                style: {color: '#fff'}
            },
            labels: {
                style: {color: '#fff'}
            },
            gridLineWidth: 0,
            minorGridLineWidth: 0,
        },
        legend: {
            enabled: true,
            itemStyle: {color: '#fff'},
            layout: 'vertical',
            verticalAlign: 'middle',
            align: 'right',
            width: 100,
            itemWidth: 100,
            navigation: {
                activeColor: '#3E576F',
                animation: true,
                arrowSize: 12,
                inactiveColor: '#CCC',
                style: {
                    fontWeight: 'bold',
                    color: '#333',
                    fontSize: '12px'
                }
            }
        },
        plotOptions: {
            series: {
                marker: {
                    enabled: false,
                    lineWidth: 1,
                    lineColor: '#ffffff'
                }
            }
        },
        series: series,
    });
}
