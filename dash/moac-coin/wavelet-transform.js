async function makeWaveletTransformChart() {
    const waveletTransformData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-wavelet-transform.json`);
    const maxLength = Math.max(...waveletTransformData.coefficients.map(level => level.length));
    const waveletSeries = waveletTransformData.coefficients.map((level, index) => ({
        name: `Wavelet Level ${index + 1}`,
        data: interpolateArray(level, maxLength).map((coef, idx) => [idx, coef]),
        zIndex: 1,
        yAxis: 0,
        type: 'line',
        dashStyle: 'Solid'
    }));

    Highcharts.chart('waveletTransformContainer', {
        chart: {
            type: 'spline',
            backgroundColor: '#1a1a1a',
            zoomType: 'x',
        },
        title: {
            text: `${baseCoinId.toUpperCase()} Wavelet Transform Coefficients`,
            style: {color: '#e3dddd'}
        },
        subtitle: {
            text: 'Visualizing multi-resolution information in price data. Higher levels capture trends, lower levels capture details.',
            style: {color: '#e3dddd'}
        },
        xAxis: {
            title: {
                text: 'Coefficient Index',
                style: {color: '#fff'}
            },
            labels: {
                style: {color: '#fff'}
            },
        },
        yAxis: [{ // First yAxis
            title: {
                text: 'Wavelet Coefficients',
                style: {color: '#fff'}
            },
            labels: {
                style: {color: '#fff'}
            },
            gridLineWidth: 0,
            minorGridLineWidth: 0
        }],
        legend: {
            // layout: 'vertical',
            // align: 'right',
            // verticalAlign: 'middle',
            itemStyle: {color: '#fff'}
        },
        series: [...waveletSeries],
        responsive: {
            rules: [{
                condition: {
                    maxWidth: 500
                },
                chartOptions: {
                    legend: {
                        layout: 'horizontal',
                        align: 'center',
                        verticalAlign: 'bottom'
                    }
                }
            }]
        },
        plotOptions: {
            series: {
                marker: {
                    enabled: false,
                },
                lineWidth: 4
            },
        }
    });
}

function interpolateArray(data, newLength) {
    const interpolated = [];
    const factor = (data.length - 1) / (newLength - 1);
    for (let i = 0; i < newLength; i++) {
        const index = i * factor;
        const lowerIndex = Math.floor(index);
        const upperIndex = Math.ceil(index);
        const weight = index - lowerIndex;
        interpolated.push(data[lowerIndex] * (1 - weight) + (data[upperIndex] ? data[upperIndex] * weight : 0));
    }
    return interpolated;
}

// makeWaveletTransformChart()
