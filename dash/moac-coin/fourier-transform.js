async function makeFFTChart(customFreq = null) {
    try {
        const fftResponse = await fetch(`${PATH_PREFIX}/${baseCoinId}-fft-data.json`);
        const fftData = await fftResponse.json();
        const priceResponse = await fetch(`${PATH_PREFIX}/${baseCoinId}-price-data.json`);
        const priceData = await priceResponse.json();
        const cycles = fftData.cycles_by_frequency;

        const colors = ['#7cb5ec', '#434348', '#90ed7d', '#f7a35c', '#8085e9', '#f15c80', '#e4d354', '#2b908f', '#f45b5b', '#91e8e1'];
        const frequencyKeys = customFreq ? [customFreq] : Object.keys(cycles);
        const seriesData = frequencyKeys.map((frequency, index) => ({
            name: `Frequency ${frequency}`,
            data: cycles[frequency].map(item => ({
                x: Date.parse(item.start_date),
                x2: Date.parse(item.conclusion_date),
                y: frequencyKeys.indexOf(frequency),
                description: `Magnitude: ${item.magnitude}, Phase: ${item.phase}, Period: ${item.period} days`
            })),
            color: colors[index % colors.length],
            pointWidth: 2000,
            tooltip: {
                pointFormatter: function () {
                    return this.description;
                }
            }
        }));

        // Convert price data and add to the series
        const priceSeries = {
            name: 'Price',
            data: priceData.map(item => [item[0], item[1]]),
            type: 'line',
            color: '#FFD700',
            yAxis: 1,
            tooltip: {
                valueDecimals: 2
            }
        };

        seriesData.push(priceSeries);

        Highcharts.chart('fftContainer', {
            chart: {
                type: 'xrange',
                backgroundColor: '#1a1a1a',
                zoomType: 'x',
            },
            plotOptions: {
                series: {
                    pointPadding: 0.1, // Reduces the padding inside the category; adjust as needed
                    groupPadding: 0.999, // Reduces the padding between categories; adjust as needed
                },
                xrange: {
                    maxPointWidth: 2000,
                }
            },
            title: {
                text: `$${baseCoinId.toUpperCase()} Fourier Transform Cycles`,
                style: {color: '#e3dddd'}
            },
            subtitle: {
                text: "The chart shows the dominant price cycles as detected by the Fourier Transform algorithm. The x-axis represents the date range of each cycle.",
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
            yAxis: [{
                title: {
                    text: 'Frequency',
                    style: {color: '#fff'}
                },
                labels: {
                    style: {color: '#fff'}
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0,
                reversed: true
            }, {
                title: {
                    text: 'Price',
                    style: {color: '#FFD700'}
                },
                labels: {
                    style: {color: '#FFD700'}
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0,
                opposite: true
            }],
            series: seriesData,
            tooltip: {
                shared: true,
                positioner: function (labelWidth, labelHeight) {
                    return {
                        x: this.chart.plotWidth / 2 - labelWidth / 2,
                        y: this.chart.plotHeight / 2 - labelHeight / 2
                    };
                },
            },
            legend: {
                enabled: true,
                itemStyle: {color: '#fff'}
            },
            credits: {
                enabled: false
            }
        });

    } catch (error) {
        console.error('Error loading the data: ', error);
    }
}
