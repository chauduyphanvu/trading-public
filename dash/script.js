const HOLDERS_DATA_URL = '../data/mother_holders_data.json';
let hodlersData = {};

fetch(HOLDERS_DATA_URL)
    .then(response => response.json())
    .then(jsonData => {
        hodlersData = jsonData;

        // Default to larger time frame for rendering responsiveness
        makeHighChartLineGraph('1H');
        makePercentageGraph('1H')
    });
