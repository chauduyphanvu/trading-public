async function makePriceSwingsHeatmap() {
    const data = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-price-swings.json`);
    const swings = data.swings;
    const swingsThreshold = data['threshold'];
    const heatmapData = swings.map((swing, index) => {
        return [index, 0, swing.change]; // Heatmap x, y, value
    });

    Highcharts.chart('priceSwingsHeatmapContainer', {
        chart: {
            type: 'heatmap',
            backgroundColor: '#1a1a1a',
        },
        title: {
            text: `${baseCoinId.toUpperCase()} Price Swings`,
            style: {color: '#e3dddd'}
        },
        subtitle: {
            text: `Heatmap of day-to-day price swings greater than ${swingsThreshold}%`,
            style: {color: '#e3dddd'}
        },
        xAxis: {
            categories: swings.map(swing =>
                `${new Date(swing.from).toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                    // hour: '2-digit',
                    // minute: '2-digit'
                })} 
        to 
        ${new Date(swing.to).toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                    // hour: '2-digit',
                    // minute: '2-digit'
                })}`),
            labels: {
                rotation: -45,
                style: {color: '#fff'}
            }
        },

        yAxis: {
            categories: ['Change'],
            title: null,
            style: {color: '#fff'},
            labels: {
                style: {color: '#fff'}
            }
        },
        colorAxis: {
            min: Math.min(...heatmapData.map(item => item[2])),
            max: Math.max(...heatmapData.map(item => item[2])),
            minColor: 'white',
            maxColor: 'rgb(79,222,86)',
            labels: {
                style: {
                    color: '#ffffff' // Makes color axis labels white
                }
            },
        },
        legend: {
            align: 'right',
            layout: 'vertical',
            margin: 0,
            verticalAlign: 'top',
            y: 25,
            symbolHeight: 280
        },
        tooltip: {
            formatter: function () {
                return `<strong>${this.series.xAxis.categories[this.point.x]}</strong><br>Change: <strong>${this.point.value.toFixed(2)}%</strong>`;
            }
        },
        series: [{
            name: 'Price Change',
            borderWidth: 1,
            data: heatmapData,
            dataLabels: {
                enabled: true,
                color: '#000000',
                format: '{point.value:.2f}' // Format the label to display value with two decimal places
            }
        }]
    });
}

// makePriceSwingsHeatmap()
