const s3Url = "https://username.s3.us-east-2.amazonaws.com/trading/gme-ta-trading-signals.json.gz";
let lastProcessedSignalTime = null;
const SIGNAL_THRESHOLD_SECONDS = 60; // Define how old a signal can be to still trigger a flash
let chart;

function fetchData() {
    // Append a timestamp to the URL to prevent caching
    const uniqueUrl = `${s3Url}?t=${new Date().getTime()}`;

    console.log(`Fetching price and TA data from ${uniqueUrl}...`);
    return fetch(uniqueUrl)
        .then(response => response.arrayBuffer())
        .then(buffer => {
            console.log('Data fetched successfully. Decompressing...');
            const decompressed = new TextDecoder("utf-8").decode(pako.ungzip(buffer));
            return JSON.parse(decompressed);
        })
        .catch(error => {
            console.error(`Error fetching price and TA data: ${error}`);
            throw error;
        });
}

function isNewSignal(signalTime) {
    const currentTime = new Date().getTime();
    const signalAge = (currentTime - signalTime) / 1000; // Age in seconds
    return signalAge <= SIGNAL_THRESHOLD_SECONDS;
}

function determineLatestSignal(data) {
    const lastBuySignalIndex = data.buy_signals.length - 1;
    const lastSellSignalIndex = data.sell_signals.length - 1;

    const lastBuySignalTime = lastBuySignalIndex >= 0 ? new Date(data.buy_signals[lastBuySignalIndex]).getTime() : null;
    const lastSellSignalTime = lastSellSignalIndex >= 0 ? new Date(data.sell_signals[lastSellSignalIndex]).getTime() : null;

    if (lastBuySignalTime && (!lastSellSignalTime || lastBuySignalTime > lastSellSignalTime)) {
        return {
            latestSignal: {
                x: lastBuySignalTime,
                y: data.buy_prices[lastBuySignalIndex],
                text: `Buy: $${data.buy_prices[lastBuySignalIndex].toFixed(2)}`
            },
            latestSignalColor: 'rgba(0, 128, 0, 0.75)',
            latestSignalTime: lastBuySignalTime
        };
    } else if (lastSellSignalTime) {
        return {
            latestSignal: {
                x: lastSellSignalTime,
                y: data.sell_prices[lastSellSignalIndex],
                text: `Sell: $${data.sell_prices[lastSellSignalIndex].toFixed(2)}`
            },
            latestSignalColor: 'rgba(255, 0, 0, 0.75)',
            latestSignalTime: lastSellSignalTime
        };
    } else {
        // If there are no buy or sell signals, return a default value or handle it accordingly
        return {
            latestSignal: null,
            latestSignalColor: null,
            latestSignalTime: null
        };
    }
}

function checkAndUpdateLastProcessedSignal(latestSignalTime, latestSignalColor) {
    if (isNewSignal(latestSignalTime) && (lastProcessedSignalTime === null || latestSignalTime > lastProcessedSignalTime)) {
        const flashColor = latestSignalColor === 'rgba(0, 128, 0, 0.75)' ? 'flash-green' : 'flash-red';
        flashScreen(flashColor);
        lastProcessedSignalTime = latestSignalTime;
    }
}

function prepareSeries(data, indicatorSettings) {
    const series = [
        {
            id: 'Price',
            name: 'Price',
            data: data.price.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            lineWidth: 2,
            yAxis: 0
        }
    ];

    if (indicatorSettings['SMA'] !== false && data.SMA) {
        series.push({
            id: 'SMA',
            name: 'SMA',
            data: data.SMA.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 0
        });
    }

    if (indicatorSettings['EMA'] !== false && data.EMA) {
        series.push({
            id: 'EMA',
            name: 'EMA',
            data: data.EMA.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 0
        });
    }

    if (indicatorSettings['RSI'] !== false && data.RSI) {
        series.push({
            id: 'RSI',
            name: 'RSI',
            data: data.RSI.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['Upper Band'] !== false && data.upper_band) {
        series.push({
            id: 'Upper Band',
            name: 'Upper Band',
            data: data.upper_band.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 0,
        });
    }

    if (indicatorSettings['Lower Band'] !== false && data.lower_band) {
        series.push({
            id: 'Lower Band',
            name: 'Lower Band',
            data: data.lower_band.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 0,
        });
    }

    if (indicatorSettings['ATR'] !== false && data.ATR) {
        series.push({
            id: 'ATR',
            name: 'ATR',
            data: data.ATR.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['Stochastic %K'] !== false && data.Stoch_K) {
        series.push({
            id: 'Stochastic %K',
            name: 'Stochastic %K',
            data: data.Stoch_K.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['Stochastic %D'] !== false && data.Stoch_D) {
        series.push({
            id: 'Stochastic %D',
            name: 'Stochastic %D',
            data: data.Stoch_D.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['OBV'] !== false && data.OBV) {
        series.push({
            id: 'OBV',
            name: 'OBV',
            data: data.OBV.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['CCI'] !== false && data.CCI) {
        series.push({
            id: 'CCI',
            name: 'CCI',
            data: data.CCI.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['VWAP'] !== false && data.VWAP) {
        series.push({
            id: 'VWAP',
            name: 'VWAP',
            data: data.VWAP.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 0,
        });
    }

    if (indicatorSettings['MACD'] !== false && data.MACD) {
        series.push({
            id: 'MACD',
            name: 'MACD',
            data: data.MACD.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 2
        });
    }

    if (indicatorSettings['MACD Signal'] !== false && data.MACD_Signal) {
        series.push({
            id: 'MACD Signal',
            name: 'MACD Signal',
            data: data.MACD_Signal.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 2
        });
    }

    if (indicatorSettings['MACD Hist'] !== false && data.MACD_Hist) {
        series.push({
            id: 'MACD Hist',
            name: 'MACD Hist',
            data: data.MACD_Hist.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 2
        });
    }

    if (indicatorSettings['ADX'] !== false && data.ADX) {
        series.push({
            id: 'ADX',
            name: 'ADX',
            data: data.ADX.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['ADX+DI'] !== false && data['ADX+DI']) {
        series.push({
            id: 'ADX+DI',
            name: 'ADX+DI',
            data: data['ADX+DI'].map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['ADX-DI'] !== false && data['ADX-DI']) {
        series.push({
            id: 'ADX-DI',
            name: 'ADX-DI',
            data: data['ADX-DI'].map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['MFI'] !== false && data.MFI) {
        series.push({
            id: 'MFI',
            name: 'MFI',
            data: data.MFI.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['TRIX'] !== false && data.TRIX) {
        series.push({
            id: 'TRIX',
            name: 'TRIX',
            data: data.TRIX.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['TRIX Signal'] !== false && data.TRIX_signal) {
        series.push({
            id: 'TRIX Signal',
            name: 'TRIX Signal',
            data: data.TRIX_signal.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['TSI'] !== false && data.TSI) {
        series.push({
            id: 'TSI',
            name: 'TSI',
            data: data.TSI.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['TSI Signal'] !== false && data.TSI_signal) {
        series.push({
            id: 'TSI Signal',
            name: 'TSI Signal',
            data: data.TSI_signal.map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['Williams %R'] !== false && data['Williams %R']) {
        series.push({
            id: 'Williams %R',
            name: 'Williams %R',
            data: data['Williams %R'].map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    if (indicatorSettings['Ultimate Oscillator'] !== false && data['Ultimate Oscillator']) {
        series.push({
            id: 'Ultimate Oscillator',
            name: 'Ultimate Oscillator',
            data: data['Ultimate Oscillator'].map((p, i) => [new Date(data.timestamp[i]).getTime(), p]),
            tooltip: {
                valueDecimals: 2
            },
            yAxis: 1
        });
    }

    series.push({
        type: 'scatter',
        id: 'Buy Signal',
        name: 'Buy Signal',
        data: data.buy_signals.map((timestamp, i) => [new Date(timestamp).getTime(), data.buy_prices[i]]),
        marker: {
            symbol: 'triangle',
            fillColor: 'green',
            lineColor: 'white',
            lineWidth: 2,
            radius: 6
        },
        tooltip: {
            valueDecimals: 2
        },
        yAxis: 0
    });

    series.push({
        type: 'scatter',
        id: 'Sell Signal',
        name: 'Sell Signal',
        data: data.sell_signals.map((timestamp, i) => [new Date(timestamp).getTime(), data.sell_prices[i]]),
        marker: {
            symbol: 'triangle-down',
            fillColor: 'red',
            lineColor: 'white',
            lineWidth: 2,
            radius: 6
        },
        tooltip: {
            valueDecimals: 2
        },
        yAxis: 0
    });

    return series;
}

function updateChart(data) {
    Highcharts.setOptions({
        time: {
            timezone: 'America/New_York'
        }
    });

    const { latestSignal, latestSignalColor, latestSignalTime } = determineLatestSignal(data);
    checkAndUpdateLastProcessedSignal(latestSignalTime, latestSignalColor);

    const indicatorSettings = JSON.parse(localStorage.getItem('indicatorSettings')) || {};
    const series = prepareSeries(data, indicatorSettings);

    const latestPriceIndex = data.price.length - 1;
    const latestPriceTime = new Date(data.timestamp[latestPriceIndex]).getTime();
    const latestPriceValue = data.price[latestPriceIndex];
    const latestPricePoint = {
        x: latestPriceTime,
        y: 20 + latestPriceValue, // Move the text a bit higher than the price
        text: `Price: $${latestPriceValue.toFixed(2)}`,
    };

    if (chart) {
        console.log('Updating existing chart...');
        series.forEach((newSeries) => {
            const existingSeries = chart.get(newSeries.id);
            if (existingSeries) {
                console.log(`Updating existing series: ${newSeries.id}`);
                existingSeries.setData(newSeries.data, false);
            } else {
                console.log(`Adding new series: ${newSeries.id}`);
                chart.addSeries(newSeries, false);
            }
        });

        chart.redraw();
    } else {
        console.log('Creating new chart...');
        chart = Highcharts.stockChart('container', {
            chart: {
                backgroundColor: '#1a1a1a',
                zoomType: 'xy',
            },
            rangeSelector: {
                selected: 1
            },
            title: {
                text: `$${data.symbol.toUpperCase()} | Technical Indicators & Signals`,
                style: {
                    color: 'white'
                }
            },
            subtitle: {
                text: `Last price at ${data.timestamp[latestPriceIndex]}. Indicators used to generate signals: ${data.indicators.join(', ')}.`,
                style: {
                    color: 'white'
                }
            },
            xAxis: {
                type: 'datetime',
                labels: {
                    style: {
                        color: 'white'
                    },
                },
            },
            yAxis: [{
                title: {
                    text: '',
                    style: {
                        color: 'white',
                        enabled: false,
                    }
                },
                height: '60%',
                lineWidth: 1,
                labels: {
                    style: {
                        color: 'white'
                    }
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0
            }, {
                title: {
                    text: '',
                    style: {
                        color: 'white'
                    }
                },
                top: '50%',
                height: '55%',
                offset: 0,
                lineWidth: 1,
                labels: {
                    style: {
                        color: 'white'
                    }
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0
            }, {
                title: {
                    text: '',
                    style: {
                        color: 'white'
                    }
                },
                top: '70%',
                height: '30%',
                offset: 0,
                lineWidth: 1,
                labels: {
                    style: {
                        color: 'white'
                    }
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0
            }],
            series: series,
            annotations: [
                {
                    labels: [
                        {
                            point: latestSignal,
                            text: latestSignal?.text,
                            backgroundColor: latestSignalColor,
                            style: {
                                color: 'white'
                            }
                        },
                        {
                            point: latestPricePoint,
                            text: latestPricePoint?.text,
                            backgroundColor: 'rgba(120,202,217,0.75)',
                            style: {
                                color: 'black'
                            }
                        }
                    ]
                }
            ]
        });

        // Apply indicator settings after creating the chart
        applyIndicatorSettings();
    }
}

function fetchDataAndUpdateChart() {
    console.log('Fetching data and updating chart...');
    fetchData()
        .then(data => {
            updateChart(data);
        });
}
