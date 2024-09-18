async function makePriceSwingsScatterPlot() {
    const data = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-price-swings.json`)
    const swingsThreshold = data['threshold'];
    const scatterData = data.swings.map(swing => {
        return {
            x: (swing.from + swing.to) / 2, // Average timestamp for the x-value
            y: swing.change, // Change for the y-value
            from: swing.from,
            to: swing.to
        };
    });

    // Creating the scatter plot chart
    Highcharts.chart('priceSwingsScatterContainer', {
        chart: {
            type: 'scatter',
            zoomType: 'xy',
            backgroundColor: '#1a1a1a',
        },
        title: {
            text: `${baseCoinSymbol} Price Swings`,
            style: {color: '#e3dddd'}
        },
        subtitle: {
            text: `Scatter plot of day-to-day price swings greater than ${swingsThreshold}%`,
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
                text: 'Price Change (%)',
                style: {color: '#fff'}
            },
            gridLineWidth: 0,
            minorGridLineWidth: 0,
            labels: {
                style: {color: '#fff'}
            },

        },
        legend: {
            enabled: false
        },
        tooltip: {
            formatter: function () {
                return `<b>Date:</b> ${Highcharts.dateFormat('%e %b %Y, %H:%M', new Date(this.point.x))}<br>
                        <b>Change:</b> ${this.point.y.toFixed(2)}%<br>
                        <b>From:</b> ${Highcharts.dateFormat('%e %b %Y, %H:%M', new Date(this.point.from))}<br>
                        <b>To:</b> ${Highcharts.dateFormat('%e %b %Y, %H:%M', new Date(this.point.to))}`;
            }
        },
        series: [{
            name: 'Price Change',
            data: scatterData,
            marker: {
                radius: 5
            }
        }]
    });

}

// makePriceSwingsScatterPlot()