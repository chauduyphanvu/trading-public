async function makePriceSwingsChart() {
    const priceData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-price-data.json`);
    const marketCapData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-market-cap-data.json`);
    const swingsData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-price-swings.json`);
    const swingsThreshold = swingsData['threshold'];
    const priceSwings = swingsData['swings'];

    Highcharts.chart('priceSwingsContainer', {
        chart: {
            type: 'line',
            backgroundColor: '#1a1a1a',
            zoomType: 'x',
            panning: true,
            panKey: 'shift',
            scrollablePlotArea: {
                minWidth: 600
            }
        },
        title: {
            text: `$${baseCoinId.toUpperCase()} Price & Market Cap w/ Daily Swings (> ${swingsThreshold}%) Annotated`,
            style: {color: '#e3dddd'}
        },
        subtitle: {
            text: 'Swings are calculated as the percentage change in price from the previous day. Positive swings are annotated in green, while negative swings are annotated in red.',
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
            }
        },
        yAxis: [
            {
                title: {
                    text: 'Price (USD)',
                    style: {color: 'white'}
                },
                labels: {
                    style: {color: '#fff'}
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0,
            },
            {
                title: {
                    text: 'Market Cap (USD)',
                    style: {color: 'white'}
                },
                labels: {
                    style: {color: '#dd00ff'}
                },
                gridLineWidth: 0,
                minorGridLineWidth: 0,
                opposite: true  // This sets up the secondary axis on the right
            }
        ],
        legend: {
            enabled: true,
            itemStyle: {
                color: '#e3dddd'
            }
        },
        series: [
            {
                name: 'Price',
                data: priceData,
                zIndex: 1,
                yAxis: 0
            },
                {
                name: 'Market Cap',
                data: marketCapData,
                zIndex: 1,
                yAxis: 1,
                color: '#dd00ff'
            }
        ],
        annotations: priceSwings.map(swing => {
            const isPositive = swing.change > 0;
            return {
                labels: [{
                    point: {
                        x: (swing.from + swing.to) / 2,
                        xAxis: 0,
                        y: 0,
                        yAxis: 0
                    },
                    text: `${swing.change.toFixed(2)}%`
                }],
                shapes: [{
                    type: 'path',
                    points: [{
                        x: swing.from,
                        xAxis: 0,
                        y: 0,
                        yAxis: 0
                    }, {
                        x: swing.to,
                        xAxis: 0,
                        y: 0,
                        yAxis: 0
                    }],
                    stroke: isPositive ? 'green' : 'red',
                    strokeWidth: 2000
                }]
            };
        })
    });
}

async function makeBestDaysToBuyHeatmap() {
    const swingsData = await fetchLocalData(`${PATH_PREFIX}/${baseCoinId}-price-swings.json`);
    const bestDaysData = swingsData['average_week_day_returns'];
    const probabilitiesData = swingsData['week_day_probabilities'];
    const isCrypto = ["bitcoin", "shiba-inu", "myro", "dogecoin", "solana", "dogwifcoin", "bonk", "gme"].includes(baseCoinId);

    // Define the natural order for days and weeks
    const stockDaysOrder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];
    const cryptoDaysOrder = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const weeksOrder = ['1st', '2nd', '3rd', '4th', '5th'];

    // Sort the days based on the type of asset
    const sortedDays = isCrypto ? cryptoDaysOrder : stockDaysOrder;

    // Filter out days that are not in the dataset
    const days = sortedDays.filter(day => Object.keys(bestDaysData).some(key => key.includes(day)));

    const heatmapData = [];

    // Calculate today's day of the week and week of the month
    const today = new Date();
    const dayOfWeek = today.toLocaleString('en-US', {weekday: 'long'});
    const dayOfMonth = today.getDate();
    const weekOfMonth = Math.ceil(dayOfMonth / 7);

    let todayCoordinates = null;

    weeksOrder.forEach((week, weekIndex) => {
        days.forEach((day, dayIndex) => {
            const key = `(${weekIndex + 1}, '${day}')`;
            if (bestDaysData[key] !== undefined) {
                const avgValue = bestDaysData[key];
                const probability = probabilitiesData[key];
                const color = avgValue >= 0 ? '#6ad96a' : '#ee3131';
                heatmapData.push([dayIndex, weekIndex, avgValue, probability, color]);

                // Check if this is today's cell
                if (weekIndex + 1 === weekOfMonth && day === dayOfWeek) {
                    todayCoordinates = [dayIndex, weekIndex];
                }
            }
        });
    });

    Highcharts.chart('bestBuyDaysContainer', {
        chart: {
            type: 'heatmap',
            backgroundColor: '#1a1a1a'
        },
        title: {
            text: `${baseCoinSymbol} | When to buy and sell for the best returns`,
            style: {color: '#e3dddd'}
        },
        subtitle: {
            text: 'Cells are colored based on average daily returns, with green indicating positive returns and red indicating negative returns. Probability of positive returns is shown in the tooltip.. Important: A day\'s outlook is based on data from the previous day (7 PM to 7 PM).',
            style: {color: '#e3dddd'}
        },
        xAxis: {
            categories: days,
            title: {
                text: 'Day of the Week',
                style: {color: '#fff'}
            },
            labels: {
                style: {color: '#fff'}
            }
        },
        yAxis: {
            categories: weeksOrder,
            title: {
                text: 'Week of the Month',
                style: {color: '#fff'}
            },
            labels: {
                style: {color: '#fff'}
            },
            reversed: true
        },
        colorAxis: {
            min: -10,
            max: 10,
            stops: [
                [0, '#ee3131'],
                [0.5, '#ffffff'],
                [1, '#6ad96a']
            ]
        },
        tooltip: {
            formatter: function () {
                const probability = (this.point.probability * 100).toFixed(2);
                return `<b>${this.series.xAxis.categories[this.point.x]}</b><br>Average Return: <b>${this.point.value.toFixed(2)}%</b><br>Probability of Positive Return: <b>${probability}%</b>`;
            }
        },
        legend: {
            enabled: false
        },
        plotOptions: {
            heatmap: {
                dataLabels: {
                    enabled: true,
                    color: '#000000',
                    formatter: function () {
                        return `${this.point.value.toFixed(2)}%`;
                    }
                }
            },
            series: {
                cursor: 'pointer',
            }
        },
        series: [{
            name: 'Average Daily Returns',
            data: heatmapData.map(item => ({
                x: item[0],
                y: item[1],
                value: item[2],
                probability: item[3],
                color: item[4]
            })),
            borderColor: '#1a1a1a'
        },
            {
                name: 'Today Marker',
                type: 'scatter',
                data: todayCoordinates ? [{
                    x: todayCoordinates[0],
                    y: todayCoordinates[1],
                    marker: {
                        // Calendar
                        symbol: 'url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTAR2oeYWFawx4YGJsqu3t_uWTkcch_NkTXNVPJ1yu7nw&s)',
                        width: 24,
                        height: 24
                    }
                }] : [],
                // color: 'transparent',
                showInLegend: false
            }]
    });
}

makeBestDaysToBuyHeatmap()
makePriceSwingsChart()
