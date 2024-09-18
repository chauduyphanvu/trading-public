/**
 * Retrieves data for a specified timespan. This function either returns the data as-is
 * for 15-minute intervals or aggregates it based on the specified timespan.
 * 
 * @param {string} timespan - The desired timespan for data retrieval which can be '15M', '1H', '2H', '5H', or '1D'.
 * @returns {Object} The processed data object with holders' data either aggregated or in original form depending on the timespan.
 */
function getDataForTimespan(timespan) {
    // Check if the requested timespan is 15 minutes, which does not require aggregation.
    if (timespan === '15M') {
        let nonAggregatedData = {};

        // Iterate over each holder and parse their timestamps and values.
        Object.keys(hodlersData).forEach(holder => {
            nonAggregatedData[holder] = {};
            Object.keys(hodlersData[holder]).forEach(timestamp => {
                // Convert string values to floats, removing any commas.
                nonAggregatedData[holder][timestamp] = parseFloat(hodlersData[holder][timestamp].replace(/,/g, ''));
            });
        });

        return nonAggregatedData;
    } else {
        return aggregateData(timespan);
    }
}


/**
 * Aggregates data based on the given timespan. This function normalizes timestamps to the nearest
 * desired interval (e.g., hour, two hours, five hours, one day) and then computes averages for each interval.
 * 
 * @param {string} timespan - The timespan to aggregate data over, such as '1H', '2H', '5H', or '1D'.
 * @returns {Object} An object containing aggregated data for each holder.
 */
function aggregateData(timespan) {
    let unitMultiplier;

    // Determine the number of hours to aggregate based on the provided timespan.
    switch (timespan) {
        case '2H': unitMultiplier = 2; break;
        case '5H': unitMultiplier = 5; break;
        case '1D': unitMultiplier = 24; break;
        default: unitMultiplier = 1; // Default to 1 hour if timespan is '1H' or any undefined values.
    }

    let aggregatedData = {};

    // Iterate over each holder and their timestamps.
    Object.keys(hodlersData).forEach(holder => {
        aggregatedData[holder] = {};
        Object.keys(hodlersData[holder]).forEach(timestamp => {
            const date = new Date(timestamp);
            date.setMinutes(0, 0, 0); // Normalize minutes, seconds, milliseconds to 0

            let hour = date.getHours();
            hour -= hour % unitMultiplier; // Round down to nearest aggregation unit

            date.setHours(hour);

            // Format the date string to remove milliseconds and align to the aggregated hour.
            const roundedDateString = date.toISOString().replace(/:\d{2}\.\d{3}Z$/, ':00Z');

            // Initialize or update the aggregation data for this rounded timestamp.
            if (!aggregatedData[holder][roundedDateString]) {
                aggregatedData[holder][roundedDateString] = {
                    sum: parseFloat(hodlersData[holder][timestamp].replace(/,/g, '')),
                    count: 1
                };
            } else {
                aggregatedData[holder][roundedDateString].sum += parseFloat(hodlersData[holder][timestamp].replace(/,/g, ''));
                aggregatedData[holder][roundedDateString].count++;
            }
        });
    });

    // Calculate the average for each aggregation period
    Object.keys(aggregatedData).forEach(holder => {
        Object.keys(aggregatedData[holder]).forEach(date => {
            let data = aggregatedData[holder][date];
            aggregatedData[holder][date] = data.sum / data.count;
        });
    });

    return aggregatedData;
}
