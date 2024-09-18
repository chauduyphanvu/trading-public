async function makeMonteCarloChart() {
    const data = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-monte-carlo-results.json`);
    const aggregatedPaths = data.aggregated_path_counter.map(item => item.path);
    const top10Paths = data.top_10_paths;

    Highcharts.chart('monteCarloContainer', {
        chart: {
            type: 'line',
            backgroundColor: '#1a1a1a',
            zoomType: 'x'
        },
        title: {
            text: `${baseCoinSymbol.toUpperCase()} | Monte Carlo Simulation Of Possible Price Paths`,
            style: {color: 'white'}
        },
        subtitle: {
            text: 'Top 10 Paths',
            style: {color: 'white'}
        },
        xAxis: {
            title: {
                text: 'Steps',
                style: {color: 'white'}
            },
            labels: {style: {color: 'white'}},
        },
        yAxis: {
            title: {
                text: 'Price',
                style: {color: 'white'}
            },
            labels: {style: {color: 'white'}},
            gridLineWidth: 0,
            minorGridLineWidth: 0
        },
        series: [
            {
                name: 'Simulated Paths',
                data: aggregatedPaths,
                type: 'line',
                color: 'rgba(0, 0, 255, 0.1)',
                enableMouseTracking: false
            },
            ...top10Paths.map((path, index) => ({
                name: `Top Path ${index + 1}`,
                data: path,
                type: 'line',
                color: Highcharts.getOptions().colors[index]
            }))
        ],
        plotOptions: {
            series: {
                turboThreshold: 0,
                marker: {
                    enabled: false,
                    lineWidth: 1,
                    lineColor: '#ffffff'
                }
            }
        }
    });
}
