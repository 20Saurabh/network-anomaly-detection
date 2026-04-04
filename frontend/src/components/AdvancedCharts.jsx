import React from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    Filler
);



export const TimeSeriesWithAnomalies = ({ data, anomalies, featureName }) => {
    // Plot feature value over time, highlight anomalies
    const labels = data.map((_, i) => i);
    const values = data.map(row => row[featureName]);

    // Create a point background color array
    const pointColors = anomalies.map(a => a._isAnomaly ? '#ef4444' : 'transparent');
    const pointRadii = anomalies.map(a => a._isAnomaly ? 6 : 0);

    const chartData = {
        labels,
        datasets: [
            {
                label: `${featureName} Value`,
                data: values,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)', // Light fill
                borderWidth: 1.5,
                pointRadius: 0, // No dots
                pointHoverRadius: 6, // Show on hover
                fill: true, // Fill area under line
                tension: 0.2, // Slight curve
                segment: {
                    borderColor: ctx => {
                        // Color segment red if the next point is an anomaly
                        if (ctx.p1 && anomalies[ctx.p1.dataIndex]?._isAnomaly) {
                            return '#ef4444';
                        }
                        return '#6366f1';
                    }
                }
            }
        ]
    };

    const options = {
        responsive: true,
        plugins: {
            title: { display: true, text: `Time Series: ${featureName} (Red = Anomaly)`, color: '#f8fafc' },
            legend: { display: false }
        },
        scales: {
            x: { grids: { display: false }, ticks: { color: '#94a3b8' } },
            y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
        }
    };

    return (
        <div style={{ height: '100%', width: '100%', minHeight: '300px' }}>
            <Line options={options} data={chartData} />
        </div>
    )
}
