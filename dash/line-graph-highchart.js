let chart;

function makeHighChartLineGraph(timespan) {
    let processedData = getDataForTimespan(timespan);
    let allDates = [];
    let seriesData = [];

    Object.keys(processedData).forEach(holder => {
        Object.keys(processedData[holder]).forEach(date => {
            if (!allDates.includes(date)) {
                allDates.push(date);
            }
        });
    });

    allDates = Array.from(new Set(allDates)).sort();

    Object.keys(processedData).forEach((holder, index) => {
        let data = allDates.map(date => {
            let value = processedData[holder][date];
            return [new Date(date).getTime(), value ? value : null];
        });

        seriesData.push({
            name: holder,
            data: data,
            originalData: data.slice() // Store a copy of original data
        });
    });

    // Check if chart exists and update data, else create new chart
    if (chart) {
        chart.update({
            series: seriesData
        });
    } else {
        chart = Highcharts.chart('container', {
            chart: {
                type: 'line',
                backgroundColor: '#1a1a1a',
                zoomType: 'x',
            },
            title: {
                text: 'MOTHER Whale Watcher',
                style: {
                    color: '#fff',
                    fontSize: '24px'
                }
            },
            subtitle: {
                text: 'Number of tokens held by the top 40 addresses',
                style: {
                    color: '#e3dddd',
                    fontSize: '18px'
                }
            },
            xAxis: {
                type: 'datetime',
                title: {
                    text: 'Date',
                    style: {
                        color: 'white'
                    }
                },
                labels: {
                    style: {
                        color: '#fff'
                    }
                }
            },
            yAxis: {
                title: {
                    text: 'Quantity',
                    style: {
                        color: 'white'
                    }
                },
                labels: {
                    style: {
                        color: '#fff'
                    }
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
                series: {
                    point: {
                        events: {
                            click: function () {
                                window.open('https://solscan.io/account/' + this.series.name, '_blank');
                            }
                        }
                    },
                    label: {
                        connectorAllowed: false
                    },
                    lineWidth: 3,  // Increase this value to make the line thicker
                    marker: {
                        enabled: false,  // Enabling markers, adjust as necessary
                    },
                    pointStart: Date.UTC(2010, 0, 1),
                    pointInterval: 24 * 3600 * 1000 // one day
                }
            },

            series: seriesData,
            // responsive: {
            //     rules: [{
            //         condition: {
            //             maxWidth: 500
            //         },
            //         chartOptions: {
            //             legend: {
            //                 layout: 'horizontal',
            //                 align: 'center',
            //                 verticalAlign: 'bottom'
            //             }
            //         }
            //     }]
            // }
        });
    }
}

// function makeStlWalletSizesChart() {
//     fetch('../data/generated/wif_holders_data-stl-wallet-size.json')  // Replace with the actual path to your STL JSON file
//         .then(response => response.json())
//         .then(data => {

//             stlChart = Highcharts.chart('stl-container', {
//                 chart: {
//                     type: 'line',
//                     backgroundColor: '#1a1a1a',
//                     zoomType: 'x',
//                 },
//                 title: {
//                     text: 'STL Seasonal Components',
//                     style: {
//                         color: '#fff',
//                         fontSize: '24px'
//                     }
//                 },
//                 xAxis: {
//                     type: 'datetime',
//                     title: {
//                         text: 'Date',
//                         style: {
//                             color: 'white'
//                         }
//                     },
//                     labels: {
//                         style: {
//                             color: '#fff'
//                         }
//                     }
//                 },
//                 yAxis: {
//                     title: {
//                         text: 'Normalized Seasonal Component',
//                         style: {
//                             color: 'white'
//                         }
//                     },
//                     labels: {
//                         style: {
//                             color: '#fff'
//                         }
//                     },
//                     gridLineWidth: 0,
//                     minorGridLineWidth: 0,
//                 },
//                 legend: {
//                     itemStyle: {
//                         color: '#fff'
//                     }
//                 },
//                 plotOptions: {
//                     series: {
//                         label: {
//                             connectorAllowed: false
//                         },
//                         lineWidth: 3,
//                         marker: {
//                             enabled: false,
//                         },
//                         pointStart: Date.UTC(2010, 0, 1),
//                         pointInterval: 24 * 3600 * 1000 // one day
//                     }
//                 },
//                 series: data.series,
//             });

//         });
// }

document.getElementById('btn15M').addEventListener('click', function () {
    makeHighChartLineGraph('15M');
});

document.getElementById('btn1H').addEventListener('click', function () {
    makeHighChartLineGraph('1H');
});

document.getElementById('btn2H').addEventListener('click', function () {
    makeHighChartLineGraph('2H');
});

document.getElementById('btn5H').addEventListener('click', function () {
    makeHighChartLineGraph('5H');
});

document.getElementById('btn1D').addEventListener('click', function () {
    makeHighChartLineGraph('1D');
});

document.getElementById('showFirstHalf').addEventListener('click', function () {
    if (chart) {
        const halfIndex = Math.ceil(chart.series.length / 2);
        const firstHalfSeries = Array.from({ length: halfIndex }, (_, i) => i);
        updateSeriesData(firstHalfSeries);
    }
});

document.getElementById('showLastHalf').addEventListener('click', function () {
    if (chart) {
        const halfIndex = Math.ceil(chart.series.length / 2);
        const lastHalfSeries = Array.from({ length: chart.series.length - halfIndex }, (_, i) => i + halfIndex);
        updateSeriesData(lastHalfSeries);
    }
});

document.getElementById('selectAll').addEventListener('click', function () {
    if (chart) {
        const allSeries = Array.from({ length: chart.series.length }, (_, i) => i);
        updateSeriesData(allSeries);
    }
});

document.getElementById('selectNone').addEventListener('click', function () {
    if (chart) {
        updateSeriesData([]); // Pass an empty array to hide all series
    }
});

function updateSeriesData(visibleSeries) {
    chart.series.forEach((series, index) => {
        series.update({
            visible: visibleSeries.includes(index)
        }, false); // false to prevent immediate redraw
    });
    chart.redraw(); // Redraw the chart once after all updates
}
