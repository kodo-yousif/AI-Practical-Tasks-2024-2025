let currentData = null;
let currentCenters = null;

// Generate a color palette for the classes
function generateColors(n) {
    const colors = [];
    for (let i = 0; i < n; i++) {
        const hue = (i * 360 / n) % 360;
        colors.push(`hsl(${hue}, 70%, 50%)`);
    }
    return colors;
}

// Update the scatter plot
function updatePlot(data, centers, newPoint = null) {
    const colors = generateColors(centers.length);
    const traces = [];

    // Add traces for each class
    for (let i = 0; i < centers.length; i++) {
        const classPoints = data.points.filter((_, idx) => data.labels[idx] === i);
        traces.push({
            x: classPoints.map(p => p[0]),
            y: classPoints.map(p => p[1]),
            mode: 'markers',
            type: 'scatter',
            name: `Class ${i}`,
            marker: {
                color: colors[i],
                size: 8
            }
        });
    }

    // Add centers
    traces.push({
        x: centers.map(c => c[0]),
        y: centers.map(c => c[1]),
        mode: 'markers',
        type: 'scatter',
        name: 'Centers',
        marker: {
            color: 'black',
            symbol: 'star',
            size: 15
        }
    });

    // Add new point if exists
    if (newPoint) {
        traces.push({
            x: [newPoint[0]],
            y: [newPoint[1]],
            mode: 'markers',
            type: 'scatter',
            name: 'New Point',
            marker: {
                color: 'red',
                symbol: 'x',
                size: 12,
                line: {
                    width: 2,
                    color: 'black'
                }
            }
        });
    }

    const layout = {
        title: 'KNN Visualization',
        hovermode: 'closest',
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' }
    };

    Plotly.newPlot('scatter-plot', traces, layout);
}

// Update statistics display
function updateStats(stats) {
    const traditionalStats = document.getElementById('traditional-stats');
    const samplingStats = document.getElementById('sampling-stats');

    let statsHtml = '';
    Object.entries(stats).forEach(([classId, classStat]) => {
        statsHtml += `
            <div>
                <strong>Class ${classId}:</strong>
                <ul>
                    <li>Count: ${classStat.count}</li>
                    <li>Center: (${classStat.center.map(v => v.toFixed(2)).join(', ')})</li>
                    <li>Std Dev: (${classStat.std.map(v => v.toFixed(2)).join(', ')})</li>
                </ul>
            </div>
        `;
    });

    traditionalStats.innerHTML = statsHtml;
    samplingStats.innerHTML = statsHtml;
}

// Generate new data
async function generateData() {
    try {
        const nSamples = parseInt(document.getElementById('n-samples').value);
        const nClasses = parseInt(document.getElementById('n-classes').value);

        console.log('Sending request with:', { nSamples, nClasses });

        const response = await fetch('http://localhost:8000/api/generate-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                n_samples: nSamples,
                n_classes: nClasses
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const data = await response.json();
        console.log('Received data:', data);
        
        currentData = data;
        currentCenters = data.centers;
        
        updatePlot(data, data.centers);
        updateStats(data.stats);
    } catch (error) {
        console.error('Error generating data:', error);
        alert('Error generating data. Check the console for details.');
    }
}

// Make prediction for a new point
async function predict(point) {
    const k = parseInt(document.getElementById('k-value').value);
    const sampleSize = parseInt(document.getElementById('sample-size').value);

    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            point: point,
            k: k,
            use_sampling: true,
            sample_size: sampleSize
        })
    });

    const result = await response.json();
    
    const predictionInfo = document.getElementById('prediction-info');
    predictionInfo.innerHTML = `
        <strong>Predictions for point (${point.map(v => v.toFixed(2)).join(', ')}):</strong>
        <ul>
            <li>Traditional KNN: Class ${result.traditional_prediction}</li>
            <li>Sampling KNN: Class ${result.sampled_prediction}</li>
        </ul>
    `;

    updatePlot(currentData, currentCenters, point);
}

// Event listeners
document.getElementById('generate-btn').addEventListener('click', generateData);

document.getElementById('scatter-plot').addEventListener('click', function(event) {
    const xaxis = event.target._fullLayout.xaxis;
    const yaxis = event.target._fullLayout.yaxis;
    
    if (!xaxis || !yaxis) return;
    
    const point = [
        xaxis.p2d(event.target.getBoundingClientRect().left + event.offsetX),
        yaxis.p2d(event.target.getBoundingClientRect().top + event.offsetY)
    ];
    
    predict(point);
});

// Initial data generation
generateData(); 