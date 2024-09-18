let percentageChart;

function makePercentageGraph(timespan) {
    // Aggregate data based on the selected timespan.
    let processedData = getDataForTimespan(timespan);
    let allDates = [];
    let seriesData = [];

    // Collect all unique dates from the aggregated data.
    Object.keys(processedData).forEach(holder => {
        Object.keys(processedData[holder]).forEach(date => {
            if (!allDates.includes(date)) {
                allDates.push(date);
            }
        });
    });

    allDates.sort(); // Sort dates to maintain chronological order.

    // Prepare series data for each holder based on aggregated values.
    Object.keys(processedData).forEach(holder => {
        let data = allDates.map(date => {
            return processedData[holder][date] || null; // Use aggregated value or null if no data.
        });

        seriesData.push({
            name: holder,
            data: data,
            originalData: data.slice() // Store a copy of original data
        });
    });

    percentageChart = Highcharts.chart('streamGraphContainer', {
        chart: {
            type: 'area',
            backgroundColor: '#1a1a1a',
            zoomType: 'x',
        },
        title: {
            useHTML: true,
            text: 'WIF Whale Watcher',
            align: 'left',
            style: {
                color: '#fff',
                fontSize: '24px'
            }
        },
        subtitle: {
            text: 'Percentage of tokens held by the top 40 addresses',
            align: 'left',
            style: {
                color: '#e3dddd',
                fontSize: '18px'
            }
        },
        xAxis: {
            categories: allDates.map(date => new Date(date).toISOString().substring(0, 10)),
            labels: {
                formatter: function () {
                    return Highcharts.dateFormat('%Y-%m-%d', new Date(this.value));
                },
                style: {
                    color: '#fff'
                }
            },
        },
        yAxis: {
            labels: {
                format: '{value}%',
                style: {
                    color: '#fff'
                }
            },
            title: {
                enabled: false
            },
            gridLineWidth: 0, // Removes horizontal grid lines
            minorGridLineWidth: 0, // Removes minor horizontal grid lines if any
        },
        legend: {
            itemStyle: {
                color: '#fff'
            }
        },
        plotOptions: {
            area: {
                point: {
                    events: {
                        click: function () {
                            window.open('https://solscan.io/account/' + this.series.name, '_blank');
                        }
                    }
                },

                stacking: 'percent',
                marker: {
                    enabled: false,
                    lineWidth: 1,
                    lineColor: '#ffffff'
                }
            }
        },
        series: seriesData
    });
}

document.getElementById('streamShowAll').addEventListener('click', function () {
    if (this.checked) {
        showAllSeries();
    }
});

document.getElementById('streamShowNone').addEventListener('click', function () {
    if (this.checked) {
        showNoSeries();
    }
});

document.getElementById('streamShowFirstHalf').addEventListener('click', function () {
    if (this.checked) {
        const halfIndex = Math.ceil(percentageChart.series.length / 2);
        const firstHalfSeries = Array.from({length: halfIndex}, (_, i) => i);

        updateStreamGraphSeriesData(firstHalfSeries);
    }
});

document.getElementById('streamShowLastHalf').addEventListener('click', function () {
    if (this.checked) {
        const halfIndex = Math.ceil(percentageChart.series.length / 2);
        const lastHalfSeries = Array.from({length: percentageChart.series.length - halfIndex}, (_, i) => i + halfIndex);

        updateStreamGraphSeriesData(lastHalfSeries);
    }
});

function showAllSeries() {
    percentageChart.series.forEach(series => {
        series.update({
            visible: true,
            data: series.options.originalData // Ensure the original data is used
        }, false); // false to prevent immediate redraw
    });
    percentageChart.redraw(); // Redraw the chart once after all updates
}

function showNoSeries() {
    percentageChart.series.forEach(series => {
        series.update({
            visible: false
        }, false); // false to prevent immediate redraw
    });
    percentageChart.redraw(); // Redraw the chart once after all updates
}

function updateStreamGraphSeriesData(visibleSeries) {
    percentageChart.series.forEach((series, index) => {
        if (visibleSeries.includes(index)) {
            series.update({
                visible: true,
                data: series.options.originalData // Use previously stored original data
            }, false); // false to prevent immediate redraw
        } else {
            series.update({
                visible: false
            }, false); // false to prevent immediate redraw
        }
    });
    percentageChart.redraw(); // Redraw the chart once after all updates
}

/**
 * Resets the dataset visibility buttons to "All" and clears any holder filters.
 * This function ensures that when a new timeframe is selected, the graph displays all data.
 */
function resetDatasetPercentageGraphBtns() {
    // Check if the 'streamShowAll' radio button is not already checked
    if (!document.getElementById('streamShowAll').checked) {
        document.getElementById('streamShowAll').checked = true; // Reset to show all datasets
    }
}

/**
 * Attach event listeners to the timeframe buttons to update the percentage graph
 * according to the selected timeframe, and reset the dataset buttons if necessary.
 */
document.getElementById('btn15MStream').addEventListener('click', () => {
    resetDatasetPercentageGraphBtns();
    makePercentageGraph('15M');  // Call the percentage graph function with the '15M' timeframe
});

document.getElementById('btn1HStream').addEventListener('click', () => {
    resetDatasetPercentageGraphBtns();
    makePercentageGraph('1H');  // Call the percentage graph function with the '1H' timeframe
});

document.getElementById('btn2HStream').addEventListener('click', () => {
    resetDatasetPercentageGraphBtns();
    makePercentageGraph('2H');  // Call the percentage graph function with the '2H' timeframe
});

document.getElementById('btn5HStream').addEventListener('click', () => {
    resetDatasetPercentageGraphBtns();
    makePercentageGraph('5H');  // Call the percentage graph function with the '5H' timeframe
});

document.getElementById('btn1DStream').addEventListener('click', () => {
    resetDatasetPercentageGraphBtns();
    makePercentageGraph('1D');  // Call the percentage graph function with the '1D' timeframe
});
