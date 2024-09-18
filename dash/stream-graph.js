function makeStreamGraph(jsonData) {
    let allDates = [];
    let seriesData = [];

    Object.keys(jsonData).forEach(holder => {
        allDates = allDates.concat(Object.keys(jsonData[holder]));
    });

    allDates = Array.from(new Set(allDates)).sort();  // Ensure the dates are unique and sorted

    Object.keys(jsonData).forEach((holder, index) => {
        let dataSeries = new Array(allDates.length).fill(0);  // Initialize with zeros
        seriesData.push({name: holder, data: dataSeries});

        // Fill in the data
        Object.entries(jsonData[holder]).forEach(([date, value]) => {
            let dateIndex = allDates.indexOf(date);
            if (dateIndex !== -1) {
                seriesData[index].data[dateIndex] = parseFloat(value.replace(/,/g, '')) || 0;
            }
        });
    });

    myChart = Highcharts.chart('container', {
        chart: {
            type: 'streamgraph',
            marginBottom: 30,
            zoomType: 'x'
        },
        title: {
            floating: true,
            align: 'left',
            text: '$MYRO Whale Watcher',
            maxFontSize: 40
        },
        subtitle: {
            floating: true,
            align: 'left',
            y: 30,
            text: 'Number of tokens held by the top 40 holders',
            maxFontSize: 30
        },
        xAxis: {
            categories: allDates,
            type: 'category',
            crosshair: true,
            labels: {
                align: 'left',  // Adjust text alignment if necessary
                reserveSpace: true,
                rotation: -45,  // Try different angles
                step: 6,
                formatter: function () {
                    return Highcharts.dateFormat('%Y-%m-%d', new Date(this.value));
                }
            },
            lineWidth: 0,
            margin: 20,
            tickWidth: 0
        },
        yAxis: {
            visible: false,
            startOnTick: false,
            endOnTick: false
        },
        legend: {
            enabled: false,
        },
        plotOptions: {
            series: {
                label: {
                    minFontSize: 5,
                    maxFontSize: 15,
                    style: {
                        color: 'rgba(255,255,255,0.75)'
                    }
                }
            }
        },
        series: seriesData,
        exporting: {
            sourceWidth: 1600,
            sourceHeight: 1200
        }
    });

    document.getElementById('streamShowAll').addEventListener('click', function () {
        myChart.series.forEach(function (series) {
            series.show();
        });
    });

    document.getElementById('streamShowNone').addEventListener('click', function () {
        myChart.series.forEach(function (series) {
            series.hide();
        });
    });

    document.getElementById('streamShowFirstHalf').addEventListener('click', function () {
        const halfPoint = Math.ceil(myChart.series.length / 2);
        myChart.series.forEach((series, index) => {
            if (index < halfPoint) {
                series.show();
            } else {
                series.hide();
            }
        });
    });

    document.getElementById('streamShowLastHalf').addEventListener('click', function () {
        const halfPoint = Math.ceil(myChart.series.length / 2);
        myChart.series.forEach((series, index) => {
            if (index >= halfPoint) {
                series.show();
            } else {
                series.hide();
            }
        });
    });
}
